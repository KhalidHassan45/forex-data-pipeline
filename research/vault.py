#!/usr/bin/env python3
"""
Locked-vault test — the ONE out-of-sample check a candidate gets on data nobody has looked at.

    python vault.py --list                                   # candidates awaiting the vault
    python vault.py --id "session|GBPUSD|london" --approved-by Khalid
    python vault.py --history                                # every unlock so far

Rules enforced in code:
    * only hypotheses whose latest SCAN status is CANDIDATE
    * direction is frozen from the research window (cannot be flipped)
    * each hypothesis can open the vault ONCE — a retry is refused
    * an approver name is mandatory and logged
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from hypotheses import by_id  # noqa: E402

VAULT_RULES = {"t_min": 2.0, "net_sharpe_min": 0.3, "net_pf_min": 1.10, "min_trades": 30}


def latest(reg, hid, event):
    rows = [r for r in reg if r["id"] == hid and r.get("event") == event]
    return rows[-1] if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id")
    ap.add_argument("--approved-by")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--tf", default="h1")
    a = ap.parse_args()
    reg = C.registry()

    if a.history:
        for r in (r for r in reg if r.get("event", "").startswith("VAULT")):
            print(json.dumps(r, ensure_ascii=False))
        return 0
    if a.list:
        ids = {r["id"] for r in reg if r.get("event") == "SCAN"}
        for hid in sorted(ids):
            s = latest(reg, hid, "SCAN")
            if s["status"] == "CANDIDATE" and not latest(reg, hid, "VAULT"):
                print(f"{hid}\tt={s['t']}\tstab={s['stability']}\tnetSR={s['net_sharpe']}")
        return 0

    if not a.id or not a.approved_by:
        print("refused: --id and --approved-by are both required")
        return 2
    scan = latest(reg, a.id, "SCAN")
    if not scan or scan["status"] != "CANDIDATE":
        print(f"refused: {a.id} is not a CANDIDATE")
        return 2
    if latest(reg, a.id, "VAULT"):
        print(f"refused: {a.id} already used its single vault test")
        return 2

    D = C.load_all(a.tf)
    vs = C.vault_start(D)
    h = by_id(D, sorted(D), a.id)
    V = {p: df[df.index >= vs - pd.Timedelta(days=60)] for p, df in D.items()}  # 60d warm-up
    s = h.fn(V)
    s = s[s.index >= vs]
    st, net = C.evaluate(s, D[h.pair][D[h.pair].index >= vs], h.pair, direction=scan["direction"])
    if st is None:
        print("no exposure in vault window")
        return 1

    fails = []
    if st["t"] < VAULT_RULES["t_min"]: fails.append(f"t={st['t']}")
    if st["net_sharpe"] < VAULT_RULES["net_sharpe_min"]: fails.append(f"netSR={st['net_sharpe']}")
    if st["net_profit_factor"] < VAULT_RULES["net_pf_min"]: fails.append(f"PF={st['net_profit_factor']}")
    if st["net_trades"] < VAULT_RULES["min_trades"]: fails.append(f"trades={st['net_trades']}")
    event = "VAULT_PASS" if not fails else "VAULT_FAIL"
    unlocks = sum(1 for r in reg if r.get("event", "").startswith("VAULT")) + 1

    rec = {"event": event, "id": a.id, "ts": C.now(), "approved_by": a.approved_by,
           "vault_start": str(vs.date()), "direction": scan["direction"], "why": ",".join(fails),
           "unlock_number": unlocks, **{k: st[k] for k in ("t", "stability", "net_sharpe", "net_profit_factor",
                                                           "net_max_dd_pct", "net_trades", "net_cagr_pct")}}
    C.append([{**rec, "event": "VAULT"}, rec])   # "VAULT" marker = used; second row = verdict

    if event == "VAULT_PASS":
        spec = {
            "hypothesis": a.id, "description": h.desc, "pair": h.pair, "params": h.params,
            "direction": "follow" if scan["direction"] > 0 else "fade",
            "research_stats": {k: scan[k] for k in ("t", "stability", "net_sharpe")},
            "vault_stats": rec,
            "paper_trading_plan": {
                "duration": "3 months minimum",
                "risk_per_trade": "0.5% of demo equity",
                "kill_switch": f"stop if drawdown exceeds {round(1.5 * rec['net_max_dd_pct'], 1)}% "
                               "or profit factor < 1.0 after 40 trades",
                "review": "monthly: live vs vault stats",
            },
            "status": "AWAITING_KHALID_APPROVAL_FOR_PAPER_TRADING",
        }
        p = C.RES_DIR / "promoted" / f"{a.id.replace('|', '_')}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec, ensure_ascii=False, indent=2, default=str))
        rec["spec_file"] = str(p)

    print(json.dumps(rec, ensure_ascii=False, indent=2, default=str))
    if unlocks > 10:
        print(f"WARNING: vault opened {unlocks} times — it is losing its value as untouched data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
