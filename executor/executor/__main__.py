"""fx-executor CLI.

    python -m executor check                      # config + broker connectivity, no orders
    python -m executor run-once                   # one cycle
    python -m executor loop --interval 300        # forever (the container's main process)
    python -m executor status
    python -m executor approve --stage practice --id "ml|EURUSD|h1|…" --by Khalid
    python -m executor reset-kill --by Khalid
    python -m executor flatten --by Khalid        # close every fx-executor trade and lock
    python -m executor replay --from 2026-03-01 [--to …] [--cash 100000]   # FX_MODE=sim: run the real engine on history
"""
import argparse
import json
import sys
import time
import traceback

import pandas as pd

from .config import Config
from .engine import Engine


def build(cfg):
    from .signals import MLBridge
    ml = MLBridge(cfg)
    if cfg.mode in ("practice", "live"):
        from .brokers.oanda import OandaBroker
        return Engine(cfg, OandaBroker(cfg.oanda_account, cfg.oanda_token, cfg.mode), ml)
    raise SystemExit("FX_MODE=sim is only for `replay` and tests")


def replay(cfg, start, end, cash):
    from .brokers.sim import SimBroker
    from .signals import MLBridge
    ml = MLBridge(cfg)
    D = ml.C.load_all("h1")
    D = {p: d for p, d in D.items() if p in cfg.universe}
    spreads = {p: ml.expected_spread_px(p) for p in D}
    b = SimBroker(D, spreads, start_cash=cash)
    clock = {"t": None}
    eng = Engine(cfg, b, ml, now_fn=lambda: clock["t"])
    idx = D[sorted(D)[0]].index
    idx = idx[(idx >= pd.Timestamp(start, tz="UTC")) & (idx <= pd.Timestamp(end or idx[-1], tz="UTC"))]
    for t in idx:
        b.set_time(t)
        clock["t"] = t
        eng.run_once()
    trades = eng.st.trades()
    closed = [t for t in trades if t["status"] == "CLOSED"]
    pl = [t["realized_pl"] or 0 for t in closed]
    g, l = sum(x for x in pl if x > 0), -sum(x for x in pl if x < 0)
    out = {"bars": len(idx), "orders": len(trades), "closed": len(closed), "open": len(eng.st.trades("OPEN")),
           "net_pl": round(sum(pl), 2), "pf": round(g / l, 3) if l else None,
           "final_equity": round(b.account().nav, 2), "status": {k: v for k, v in json.loads(
               (cfg.state_dir / cfg.mode / "status.json").read_text()).items() if k in ("dd_pct", "hwm", "kill")}}
    print(json.dumps(out, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="executor")
    ap.add_argument("cmd", choices=["check", "run-once", "loop", "status", "approve", "reset-kill", "flatten", "replay"])
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--stage", choices=["practice", "live"])
    ap.add_argument("--id"); ap.add_argument("--by")
    ap.add_argument("--from", dest="start"); ap.add_argument("--to", dest="end")
    ap.add_argument("--cash", type=float, default=100_000)
    a = ap.parse_args(argv)
    cfg = Config.load()

    if a.cmd == "replay":
        if cfg.mode != "sim":
            raise SystemExit("replay requires FX_MODE=sim")
        replay(cfg, a.start, a.end, a.cash); return 0
    if a.cmd == "status":
        f = cfg.state_dir / cfg.mode / "status.json"
        print(f.read_text() if f.exists() else "no cycle yet"); return 0

    if cfg.mode == "sim" and a.cmd == "loop":
        print("FX_MODE=sim: idle (no broker). Use `python -m executor replay --from …`.", flush=True)
        while True:
            time.sleep(3600)
    eng = build(cfg)
    if a.cmd == "check":
        acct = eng.b.account()
        px = eng.b.prices(cfg.universe[:3])
        print(json.dumps({"config": cfg.public(), "nav": acct.nav, "currency": acct.currency,
                          "prices": {k: [v.bid, v.ask] for k, v in px.items()},
                          "feature_version": eng.ml.feature_version(),
                          "models": [{"id": c["id"], "eligible": eng.eligible(c)} for c in eng.ml.cards()]},
                         indent=2, default=str)); return 0
    if a.cmd in ("approve", "reset-kill", "flatten") and not a.by:
        raise SystemExit("--by <name> is required")
    if a.cmd == "approve":
        if not (a.id and a.stage):
            raise SystemExit("--id and --stage are required")
        ok, info = eng.approve(a.id, a.stage, a.by)
        print(("APPROVED " if ok else "REFUSED ") + json.dumps(info, default=str)); return 0 if ok else 2
    if a.cmd == "reset-kill":
        k = eng.st.get("kill")
        eng.st.set("kill", None)
        eng.st.set("hwm", eng.b.account().nav)               # new high-water mark from here
        eng.audit.log("KILL_RESET", by=a.by, previous=k); eng.audit.alert(f"kill switch reset by {a.by}")
        print("kill switch cleared; high-water mark reset to current equity"); return 0
    if a.cmd == "flatten":
        eng.st.set("kill", {"ts": pd.Timestamp.now(tz="UTC"), "why": f"manual flatten by {a.by}"})
        n = eng.flatten(f"manual flatten by {a.by}")
        eng.audit.alert(f"manual flatten by {a.by}: {n} trade(s) closed, trading locked")
        print(f"closed {n}; locked until reset-kill"); return 0
    if a.cmd == "run-once":
        print(json.dumps(eng.run_once(), indent=2, default=str)); return 0
    # loop
    eng.audit.log("START", config=cfg.public())
    fails = 0
    while True:
        try:
            eng.run_once(); fails = 0
        except Exception as e:
            fails += 1
            eng.audit.log("CYCLE_ERROR", err=str(e), tb=traceback.format_exc()[-1500:])
            if fails in (1, 5, 20):
                eng.audit.alert(f"cycle error ×{fails}: {e}")
        time.sleep(a.interval - (time.time() % a.interval) + 5)   # align to the clock (+5s after the bar closes)


if __name__ == "__main__":
    sys.exit(main())
