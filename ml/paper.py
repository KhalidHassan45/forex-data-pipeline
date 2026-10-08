#!/usr/bin/env python3
"""Layer 6 — shadow paper trading, drift monitoring and the kill switch. NO broker, NO account, NO orders.

    python paper.py approve --id "ml|EURUSD|h1|ab12cd34" --approved-by Khalid   # only after Khalid's explicit OK
    python paper.py monitor            # daily: new signals → ledger of finished trades → drift → kill switch
    python paper.py status             # one line per model

How it works: every day, after the data pipeline adds yesterday's bars, each PAPER_ACTIVE model scores bars it
has never seen. A signal becomes a paper trade (next-bar open, same barriers and costs as the labels); once its
barriers resolve it lands in the ledger. That is a true forward test on data that did not exist at training time.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
import features as F  # noqa: E402
import labels as LB  # noqa: E402
import models as M  # noqa: E402
from vault import predict, safe  # noqa: E402

PAPER = C.ML_DIR / "paper"
PSI_ALERT = 0.25


def cards():
    for f in sorted(C.MODEL_DIR.glob("*/card.json")):
        yield f, json.loads(f.read_text())


def save_card(f, card):
    f.write_text(json.dumps(card, ensure_ascii=False, indent=2, default=str))


def psi(ref, cur, bins=10):
    ref, cur = ref.dropna(), cur.dropna()
    if len(ref) < 200 or len(cur) < 50:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    b = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(((b - a) * np.log(b / a)).sum())


def approve(mid, who):
    f = C.MODEL_DIR / safe(mid) / "card.json"
    if not f.exists():
        print("refused: no such model"); return 2
    card = json.loads(f.read_text())
    if card["status"] != "AWAITING_KHALID_APPROVAL_FOR_PAPER_TRADING":
        print(f"refused: status is {card['status']}"); return 2
    card.update(status="PAPER_ACTIVE", paper_approved_by=who, paper_since=C.now())
    save_card(f, card)
    C.append([{"event": "PAPER_APPROVED", "id": mid, "ts": C.now(), "approved_by": who}])
    p = C.RES_DIR / "promoted" / f"{safe(mid)}.json"
    if p.exists():
        spec = json.loads(p.read_text()); spec["status"] = "PAPER_TRADING_ACTIVE (shadow, no broker)"
        p.write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    print(f"{mid} → PAPER_ACTIVE (shadow ledger; no account connected)")
    return 0


def monitor():
    D = None
    report = []
    for f, card in cards():
        if card["status"] not in ("PAPER_ACTIVE", "PAPER_PAUSED"):
            continue
        mid, pair, tf = card["id"], card["pair"], card["tf"]
        if card["feature_version"] != F.version():
            card["status"] = "PAPER_PAUSED"; card["pause_reason"] = "feature code changed — model needs its exact features"
            save_card(f, card); report.append({"id": mid, "status": card["status"], "alert": card["pause_reason"]}); continue
        D = D or C.load_all(tf)
        F.build(D, tf); LB.build({pair: D[pair]}, tf, card["label_cfg"])
        art = joblib.load(f.parent / "model.joblib")
        since = pd.Timestamp(card["paper_since"]).floor("D")
        X = F.load(pair, tf)
        X = X[X.index >= since]
        d = PAPER / safe(mid); d.mkdir(parents=True, exist_ok=True)
        rep = {"id": mid, "pair": pair, "status": card["status"], "bars_scored": int(len(X))}
        if len(X):
            P, s, size = predict(art, X)
            sig = pd.DataFrame({"side": s, "size": size, "p_long": P["p_long"], "p_short": P["p_short"]})
            sig[sig["side"] != 0].to_parquet(d / "signals.parquet")
            L = LB.load(pair, tf, card["label_cfg"])
            done = L.index.intersection(s.index)                   # bars whose trade outcome is already known
            trades = M.simulate(s.loc[done], L.loc[done], size.loc[done])
            trades.to_parquet(d / "ledger.parquet")
            st = C.clean(C.trade_metrics(trades, start=since))
            last = s[s != 0].tail(1)
            rep.update(ledger=st, open_signal=None if last.empty or last.index[0] in done else
                       {"ts": str(last.index[0]), "side": int(last.iloc[0]), "size": round(float(size[last.index[0]]), 2)})
            # kill switch (from the plan written at vault time)
            vdd = card["vault"]["net_max_dd_pct"]
            if card["status"] == "PAPER_ACTIVE" and (
                    st["net_max_dd_pct"] < 1.5 * vdd or (st["net_trades"] >= 40 and st["net_profit_factor"] < 1.0)):
                card["status"] = "PAPER_PAUSED"
                card["pause_reason"] = f"kill switch: DD={st['net_max_dd_pct']}% PF={st['net_profit_factor']}"
                C.append([{"event": "PAPER_PAUSED", "id": mid, "ts": C.now(), "why": card["pause_reason"]}])
                rep["alert"] = card["pause_reason"]
        # drift: last 60 days vs the research window, on the 15 most important features
        Xall = F.load(pair, tf)
        vs = pd.Timestamp(card["vault"]["vault_start"], tz="UTC")
        ref = Xall[(Xall.index < vs) & (Xall.index >= vs - pd.Timedelta(days=730))]
        cur = Xall[Xall.index >= Xall.index[-1] - pd.Timedelta(days=60)]
        imp = getattr(art["primary"], "feature_importances_", None)
        cols = list(pd.Series(imp, index=art["columns"]).sort_values(ascending=False).index[:15]) if imp is not None else art["columns"][:15]
        ps, drifted = {}, {}
        for c in cols:
            v = psi(ref[c], cur[c])
            ps[c] = round(v, 3)
            # each feature gets its own bar: how far do ordinary 60-day windows INSIDE the research data drift?
            # slow features (monthly trend distance…) wander naturally; only drift beyond their normal range counts
            base = [psi(ref[c], ref[c][(ref.index >= t) & (ref.index < t + pd.Timedelta(days=60))])
                    for t in pd.date_range(ref.index[0], ref.index[-1] - pd.Timedelta(days=60), freq="30D")]
            bar = max(PSI_ALERT, float(np.nanquantile(base, 0.95))) if np.isfinite(base).any() else PSI_ALERT
            if v == v and v > bar:
                drifted[c] = {"psi": round(v, 3), "normal_max": round(bar, 3)}
        rep["drift"] = {"mean_psi": round(float(np.nanmean(list(ps.values()))), 3), "drifted": drifted}
        if drifted:
            rep["retrain_suggested"] = True
        rep["status"] = card["status"]
        save_card(f, card)
        report.append(rep)
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "status.json").write_text(json.dumps({"ts": C.now(), "models": report}, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def status():
    for _, c in cards():
        print(f"{c['id']}\t{c['status']}\t{c.get('pause_reason', '')}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["approve", "monitor", "status"])
    ap.add_argument("--id"); ap.add_argument("--approved-by")
    a = ap.parse_args()
    if a.cmd == "approve":
        if not a.id or not a.approved_by:
            print("refused: --id and --approved-by are both required"); sys.exit(2)
        sys.exit(approve(a.id, a.approved_by))
    sys.exit(monitor() if a.cmd == "monitor" else status())
