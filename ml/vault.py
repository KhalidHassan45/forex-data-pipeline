#!/usr/bin/env python3
"""The ONE test on the locked vault for an ML model — the same vault forex-research uses.

    python vault.py --list
    python vault.py --id "ml|EURUSD|h1|ab12cd34" --approved-by Khalid
    python vault.py --history

Enforced in code: only CANDIDATE models · once per model · approver name mandatory · the final model is fitted
on research data only (labels ending inside the vault are purged) with thresholds and meta model frozen
BEFORE the vault is read · every unlock is also written to the research registry so both labs see the count.
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
from train import dataset  # noqa: E402

VAULT_RULES = {"t_min": 2.0, "net_sharpe_min": 0.3, "net_pf_min": 1.10, "min_trades": 30}


def latest(reg, mid, event):
    rows = [r for r in reg if r.get("id") == mid and r.get("event") == event]
    return rows[-1] if rows else None


def safe(mid):
    return mid.replace("|", "_")


def fit_final(rec, vs):
    """Freeze everything from research data: primary model, thresholds, meta model."""
    cfg = rec["config"]
    pair, tf, stride, q = cfg["pair"], cfg["tf"], cfg.get("stride", 3), cfg["quantile"]
    lcfg = {k: cfg["labels"][k] for k in ("horizon", "pt", "sl")}
    X, L = dataset(pair, tf, lcfg, vs, cfg["excluded"])
    n = len(X)
    cut = int(n * 0.8)
    cut_ts = X.index[cut]
    inner = np.where((np.arange(n) < cut) & (C.utc(L["t1"]) < cut_ts))[0]
    m_in = M.fit_primary(X.iloc[inner[::stride]], L["label"].iloc[inner[::stride]])
    thr = M.thresholds(M.proba(m_in, X.iloc[cut:]), q)
    primary = M.fit_primary(X.iloc[::stride], L["label"].iloc[::stride])
    meta = None
    if cfg["meta"] == "on":
        oof = pd.read_parquet(Path(rec["run_dir"]) / "oof.parquet")
        f = oof[oof["signal"] != 0]
        side = f["signal"]
        out = np.where(side > 0, L.loc[side.index, "ret_long"], L.loc[side.index, "ret_short"]) > 0
        meta = M.fit_meta(X.loc[side.index], f[["p_short", "p_none", "p_long"]], side, pd.Series(out, index=side.index))
    return {"primary": primary, "meta": meta, "thr": thr, "columns": list(X.columns), "lcfg": lcfg}


def predict(art, X):
    X = X[art["columns"]]
    P = M.proba(art["primary"], X)
    s = M.signals(P, art["thr"])
    size = pd.Series(np.where(s != 0, 1.0, 0.0), index=s.index)
    f = s != 0
    if art["meta"] is not None and f.any():
        size[f] = M.meta_size(art["meta"].predict_proba(M.meta_X(X[f], P[f], s[f]))[:, 1])
    return P, s, size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id"); ap.add_argument("--approved-by")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--history", action="store_true")
    a = ap.parse_args()
    reg = C.registry()
    if a.history:
        for r in reg:
            if r.get("event", "").startswith("VAULT_"):
                print(json.dumps(r, ensure_ascii=False))
        return 0
    if a.list:
        for mid in sorted({r["id"] for r in reg if r.get("event") == "TRAIN"}):
            t = latest(reg, mid, "TRAIN")
            if t["status"] == "CANDIDATE" and not latest(reg, mid, "VAULT"):
                print(f"{mid}\ttrades={t['net_trades']}\tSR={t['net_sharpe']}\tDSR={t['dsr']}\tnull={t['null_sharpe']}")
        return 0
    if not a.id or not a.approved_by:
        print("refused: --id and --approved-by are both required"); return 2
    rec = latest(reg, a.id, "TRAIN")
    if not rec or rec["status"] != "CANDIDATE":
        print(f"refused: {a.id} is not a CANDIDATE"); return 2
    if latest(reg, a.id, "VAULT"):
        print(f"refused: {a.id} already used its single vault test"); return 2
    if rec["feature_version"] != F.version():
        print("refused: feature code changed since training — retrain this config first"); return 2

    D = C.load_all(rec["tf"])
    vs = C.vault_start(D)
    art = fit_final(rec, vs)                                     # frozen before the vault is read
    F.build(D, rec["tf"]); LB.build({rec["pair"]: D[rec["pair"]]}, rec["tf"], art["lcfg"])
    Xa = F.load(rec["pair"], rec["tf"])
    La = LB.load(rec["pair"], rec["tf"], art["lcfg"])
    La = La[La.index >= vs]
    Xv = Xa.reindex(La.index)
    P, s, size = predict(art, Xv)
    trades = M.simulate(s, La, size)
    st = C.trade_metrics(trades, start=vs)

    fails = []
    if st["t"] < VAULT_RULES["t_min"]: fails.append(f"t={st['t']}")
    if st["net_sharpe"] < VAULT_RULES["net_sharpe_min"]: fails.append(f"netSR={st['net_sharpe']}")
    if st["net_profit_factor"] < VAULT_RULES["net_pf_min"]: fails.append(f"PF={st['net_profit_factor']}")
    if st["net_trades"] < VAULT_RULES["min_trades"]: fails.append(f"trades={st['net_trades']}")
    event = "VAULT_PASS" if not fails else "VAULT_FAIL"
    unlocks = C.vault_unlocks_total() + 1
    keep = ("t", "net_sharpe", "net_profit_factor", "net_max_dd_pct", "net_trades", "net_cagr_pct", "net_win_rate")
    v = {"event": event, "id": a.id, "ts": C.now(), "approved_by": a.approved_by, "vault_start": str(vs.date()),
         "why": ",".join(fails), "unlock_number": unlocks, "family": "ml", "pair": rec["pair"],
         **{k: st[k] for k in keep}}
    C.append([{**v, "event": "VAULT"}, v])                              # ML registry
    C.append([{**v, "event": "VAULT"}, v], path=C.RES_REGISTRY)          # shared vault accounting

    mdir = C.MODEL_DIR / safe(a.id)
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, mdir / "model.joblib")
    trades.to_parquet(mdir / "vault_trades.parquet")
    status = "AWAITING_KHALID_APPROVAL_FOR_PAPER_TRADING" if event == "VAULT_PASS" else "VAULT_FAILED"
    (mdir / "card.json").write_text(json.dumps({
        "id": a.id, "pair": rec["pair"], "tf": rec["tf"], "status": status, "feature_version": rec["feature_version"],
        "thresholds": art["thr"], "label_cfg": art["lcfg"], "why": rec["why"], "train": {k: rec[k] for k in (
            "net_trades", "net_sharpe", "net_profit_factor", "dsr", "edge_over_null", "stability")},
        "vault": v, "created": C.now()}, ensure_ascii=False, indent=2, default=str))

    if event == "VAULT_PASS":
        spec = {
            "kind": "ml", "hypothesis": a.id, "description": f"نموذج ML — {rec['pair']}: {rec['why']}",
            "pair": rec["pair"], "params": rec["config"]["labels"], "direction": "model",
            "research_stats": {"t": rec["t"], "stability": rec["stability"], "net_sharpe": rec["net_sharpe"],
                               "dsr": rec["dsr"]},
            "vault_stats": v,
            "paper_trading_plan": {
                "duration": "3 months minimum", "risk_per_trade": "0.5% of demo equity",
                "kill_switch": f"stop if drawdown exceeds {round(1.5 * v['net_max_dd_pct'], 1)}% "
                               "or profit factor < 1.0 after 40 trades",
                "review": "monthly: live vs vault stats + feature drift (PSI)"},
            "status": "AWAITING_KHALID_APPROVAL_FOR_PAPER_TRADING",
        }
        p = C.RES_DIR / "promoted" / f"{safe(a.id)}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec, ensure_ascii=False, indent=2, default=str))
        v["spec_file"] = str(p)
    print(json.dumps(v, ensure_ascii=False, indent=2, default=str))
    if unlocks > 10:
        print(f"WARNING: vault opened {unlocks} times by both labs — it is losing its value as untouched data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
