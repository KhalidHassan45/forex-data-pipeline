#!/usr/bin/env python3
"""Train + validate one model per pair on the RESEARCH window only (never the vault).

    python train.py --pairs EURUSD --why "momentum after volatility compression around London open"
    python train.py --pairs GBPUSD --horizon 12 --pt 2 --sl 1 --quantile 0.92 --meta off --why "..."

Pipeline per pair:
  labels (triple-barrier, net of costs) + certified features (leaky columns dropped)
  → walk-forward by year (purged + embargo); thresholds chosen on an inner validation split of TRAIN only
  → optional meta-label model trained only on earlier years' out-of-sample signals
  → trade simulation with the labeller's real fills and costs, plus a ×1.5 cost stress test
  → a NULL run on block-shuffled labels (what does this pipeline "find" in noise?)
  → Deflated Sharpe Ratio using every training run ever registered
  → CANDIDATE only if every ML_RULES gate passes; otherwise REJECTED with reasons.
"""
import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
import cv  # noqa: E402
import features as F  # noqa: E402
import labels as LB  # noqa: E402
import leakage  # noqa: E402
import models as M  # noqa: E402

ML_RULES = {"min_trades": 100, "net_sharpe_min": 0.5, "net_pf_min": 1.10, "dsr_min": 0.95,
            "stability_min": 0.6, "edge_over_null_min": 0.3, "stress_sharpe_min": 0.2}
WEEKLY_BUDGET = 20
MIN_META_SIGNALS = 300


def dataset(pair, tf, lcfg, vs, excluded):
    X = F.load(pair, tf)
    L = LB.load(pair, tf, lcfg)
    L = L[(L.index < vs) & (C.utc(L["t1"]) < vs)]          # no label may touch the vault
    X = X.drop(columns=[c for c in excluded if c in X], errors="ignore").reindex(L.index)
    keep = X.notna().mean(axis=1) >= 0.8                                # drop warm-up rows
    return X[keep], L[keep]


def block_shuffle(y, block=24, seed=123):
    rng = np.random.default_rng(seed)
    n = len(y)
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    order = rng.permutation(len(blocks))
    perm = np.concatenate([blocks[i] for i in order])[:n]
    return pd.Series(y.to_numpy()[perm], index=y.index)


def run_wf(X, L, y, q, use_meta, min_years, stride=3):
    sig_all, size_all, P_all, imp, years, meta_years = [], [], [], [], [], []
    prior = []   # earlier years' out-of-sample signals → training data for the meta model
    t1 = L["t1"]
    for yr, tr, te in cv.walk_forward(X.index, t1, min_train_years=min_years):
        # labels overlap (each spans up to `horizon` bars), so neighbouring rows are near-duplicates:
        # fitting on every `stride`-th row loses little information and is 3× faster on a shared server
        tr_fit = tr[::stride]
        Xtr, ytr = X.iloc[tr_fit], y.iloc[tr_fit]
        cut = int(len(tr) * 0.8)
        cut_ts = X.index[tr[cut]]
        inner = tr[:cut][C.utc(t1.iloc[tr[:cut]]) < cut_ts]
        m_in = M.fit_primary(X.iloc[inner[::stride]], y.iloc[inner[::stride]])
        thr = M.thresholds(M.proba(m_in, X.iloc[tr[cut:]]), q)
        m = M.fit_primary(Xtr, ytr)
        Xte = X.iloc[te]
        P = M.proba(m, Xte)
        s = M.signals(P, thr)
        size = pd.Series(np.where(s != 0, 1.0, 0.0), index=s.index)
        if use_meta:
            used = False
            if prior and sum(len(p[0]) for p in prior) >= MIN_META_SIGNALS:
                PX = pd.concat([p[0] for p in prior]); PP = pd.concat([p[1] for p in prior])
                PS = pd.concat([p[2] for p in prior]); PO = pd.concat([p[3] for p in prior])
                mm = M.fit_meta(PX, PP, PS, PO)
                f = s != 0
                if f.any():
                    pm = mm.predict_proba(M.meta_X(Xte[f], P[f], s[f]))[:, 1]
                    size[f] = M.meta_size(pm)
                used = True
            else:
                size[s != 0] = 0.5                                       # not enough history yet: half size
            meta_years.append({"year": int(yr), "meta_used": used})
        f = s != 0
        side = s[f]
        out = np.where(side > 0, L.loc[side.index, "ret_long"], L.loc[side.index, "ret_short"]) > 0
        prior.append((Xte[f], P[f], side, pd.Series(out, index=side.index)))
        sig_all.append(s); size_all.append(size); P_all.append(P)
        if hasattr(m, "feature_importances_"):
            imp.append(pd.Series(m.feature_importances_, index=X.columns))
        years.append(int(yr))
    if not sig_all:
        return None
    sig, size = pd.concat(sig_all), pd.concat(size_all)
    trades = M.simulate(sig, L, size)
    stress = M.simulate(sig, L, size, extra_cost_mult=0.5)
    met = C.trade_metrics(trades)
    by_year = trades.groupby(pd.DatetimeIndex(trades["exit_ts"]).year)["ret"].sum() if len(trades) else pd.Series(dtype=float)
    return {"metrics": met, "stress": C.trade_metrics(stress), "trades": trades, "P": pd.concat(P_all),
            "sig": sig, "size": size, "years": years, "meta_years": meta_years,
            "by_year": {int(k): round(float(v) * 100, 2) for k, v in by_year.items()},
            "stability": round(float((by_year > 0).mean()), 2) if len(by_year) else 0.0,
            "importance": (pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False) if imp else pd.Series(dtype=float))}


def trials_stats():
    rows = [r for r in C.registry() if r.get("event") == "TRAIN"]
    srs = [r["daily_sr"] for r in rows if isinstance(r.get("daily_sr"), (int, float))]
    return len(rows), (float(np.var(srs)) if len(srs) >= 5 else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--tf", default="h1")
    ap.add_argument("--horizon", type=int, default=LB.DEFAULT["horizon"])
    ap.add_argument("--pt", type=float, default=LB.DEFAULT["pt"])
    ap.add_argument("--sl", type=float, default=LB.DEFAULT["sl"])
    ap.add_argument("--quantile", type=float, default=0.90, help="trade only the top (1-q) most confident bars")
    ap.add_argument("--meta", choices=["on", "off"], default="on")
    ap.add_argument("--min-train-years", type=int, default=3)
    ap.add_argument("--stride", type=int, default=3, help="fit on every n-th bar (overlapping labels)")
    ap.add_argument("--why", required=True, help="economic rationale written BEFORE seeing results")
    a = ap.parse_args()
    if len(a.why.strip()) < 15:
        print("refused: --why must state a real economic rationale"); return 2

    reg = C.registry()
    week = [r for r in reg if r.get("event") == "TRAIN" and pd.Timestamp(r["ts"]) > pd.Timestamp(C.now()) - timedelta(days=7)]
    if len(week) + len(a.pairs) > WEEKLY_BUDGET:
        print(f"refused: weekly budget {WEEKLY_BUDGET} training runs (used {len(week)})"); return 2

    cert = leakage.get_cert(a.tf)
    if not cert:
        print("refused: features not certified — run `ml.sh leakcheck` first"); return 2
    excluded = cert["leaky_columns"]

    pairs = [p.upper() for p in a.pairs]
    D = C.load_all(a.tf)
    vs = C.vault_start(D)
    lcfg = {"horizon": a.horizon, "pt": a.pt, "sl": a.sl}
    F.build(D, a.tf); LB.build({p: D[p] for p in pairs}, a.tf, lcfg)

    stamp = pd.Timestamp(C.now()).strftime("%Y%m%d_%H%M%S")
    batch = C.RUNS_DIR / stamp
    batch.mkdir(parents=True, exist_ok=True)
    results = []
    for pair in pairs:
        cfg = {"pair": pair, "tf": a.tf, "labels": {**LB.DEFAULT, **lcfg}, "features": F.version(),
               "excluded": excluded, "quantile": a.quantile, "meta": a.meta, "min_train_years": a.min_train_years, "stride": a.stride,
               "primary": M.PRIMARY_PARAMS, "meta_params": M.META_PARAMS}
        mid = f"ml|{pair}|{a.tf}|{C.cfg_hash(cfg)}"
        prev = [r for r in reg if r.get("event") == "TRAIN" and r["id"] == mid]
        if prev:
            print(f"skip {mid}: identical config already trained → status {prev[-1]['status']}")
            continue
        X, L = dataset(pair, a.tf, lcfg, vs, excluded)
        if len(X) < 20000:
            print(f"skip {pair}: only {len(X)} research rows"); continue
        R = run_wf(X, L, L["label"], a.quantile, a.meta == "on", a.min_train_years, a.stride)
        N = run_wf(X, L, block_shuffle(L["label"]), a.quantile, False, a.min_train_years, a.stride)
        if R is None:
            print(f"skip {pair}: not enough years for walk-forward"); continue
        n_trials, sr_var = trials_stats()
        met, null = R["metrics"], N["metrics"] if N else {"net_sharpe": 0.0}
        daily = met.get("_daily", pd.Series(dtype=float))
        dsr = C.deflated_sharpe(daily, n_trials + 1, sr_var) if len(daily) else 0.0
        daily_sr = float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() > 0 else 0.0
        edge = round(met["net_sharpe"] - null["net_sharpe"], 3)
        fails = []
        if met["net_trades"] < ML_RULES["min_trades"]: fails.append(f"trades={met['net_trades']}")
        if met["net_sharpe"] < ML_RULES["net_sharpe_min"]: fails.append(f"netSR={met['net_sharpe']}")
        if met["net_profit_factor"] < ML_RULES["net_pf_min"]: fails.append(f"PF={met['net_profit_factor']}")
        if dsr < ML_RULES["dsr_min"]: fails.append(f"DSR={dsr}")
        if R["stability"] < ML_RULES["stability_min"]: fails.append(f"stability={R['stability']}")
        if edge < ML_RULES["edge_over_null_min"]: fails.append(f"edge_over_null={edge}")
        if R["stress"]["net_sharpe"] < ML_RULES["stress_sharpe_min"]: fails.append(f"stressSR={R['stress']['net_sharpe']}")
        status = "CANDIDATE" if not fails else "REJECTED"
        lab_dist = L["label"].value_counts(normalize=True).round(3).to_dict()

        out = batch / pair
        out.mkdir(exist_ok=True)
        R["trades"].to_parquet(out / "trades.parquet")
        pd.concat([R["P"], R["sig"].rename("signal"), R["size"].rename("size")], axis=1).to_parquet(out / "oof.parquet")
        eq = (1 + R["trades"].set_index("exit_ts")["ret"]).cumprod() if len(R["trades"]) else pd.Series(dtype=float)
        rec = {"event": "TRAIN", "id": mid, "ts": C.now(), "pair": pair, "tf": a.tf, "why": a.why, "status": status,
               "fails": ",".join(fails), "run_dir": str(out), "vault_start": str(vs.date()),
               "label_cfg": cfg["labels"], "label_dist": {str(k): v for k, v in lab_dist.items()},
               "feature_version": F.version(), "n_features": int(X.shape[1]), "excluded_features": excluded,
               "quantile": a.quantile, "meta": a.meta, "wf_years": R["years"], "meta_years": R["meta_years"],
               "rows": int(len(X)), **C.clean(met), "stability": R["stability"], "by_year_pct": R["by_year"],
               "stress_sharpe": R["stress"]["net_sharpe"], "null_sharpe": null["net_sharpe"], "edge_over_null": edge,
               "dsr": dsr, "n_trials": n_trials + 1, "daily_sr": round(daily_sr, 5),
               "top_features": {k: round(float(v), 1) for k, v in R["importance"].head(15).items()},
               "config": cfg}
        (out / "summary.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str))
        (out / "equity.json").write_text(json.dumps([[str(i), round(float(v), 5)] for i, v in eq.iloc[::max(1, len(eq) // 260)].items()]))
        C.append([rec])
        _mlflow(rec, out)
        results.append(rec)
        print(f"{mid}  {status}  trades={met['net_trades']} SR={met['net_sharpe']} PF={met['net_profit_factor']} "
              f"DSR={dsr} null={null['net_sharpe']} stab={R['stability']} {('· ' + ','.join(fails)) if fails else ''}")

    (batch / "summary.json").write_text(json.dumps({"run_id": stamp, "ts": C.now(), "rules": ML_RULES,
        "results": [{k: r[k] for k in ("id", "pair", "status", "fails", "net_trades", "net_sharpe",
                                        "net_profit_factor", "dsr", "edge_over_null", "stability")} for r in results]},
        ensure_ascii=False, indent=2))
    (C.ML_DIR / "LATEST").write_text(str(batch))
    return 0


def _mlflow(rec, out):
    import os
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
        mlflow.set_experiment("forex-ml-lab")
        with mlflow.start_run(run_name=rec["id"]):
            mlflow.set_tags({"pair": rec["pair"], "status": rec["status"], "why": rec["why"][:250]})
            mlflow.log_params({"tf": rec["tf"], "quantile": rec["quantile"], "meta": rec["meta"],
                               "features": rec["feature_version"], **{f"lbl_{k}": v for k, v in rec["label_cfg"].items()}})
            mlflow.log_metrics({k: float(rec[k]) for k in ("net_sharpe", "net_profit_factor", "net_max_dd_pct",
                                                           "net_trades", "dsr", "edge_over_null", "stability",
                                                           "stress_sharpe", "null_sharpe")})
            mlflow.log_artifact(str(out / "summary.json"))
    except Exception as e:  # tracking must never break the lab
        print(f"mlflow logging skipped: {e}")


if __name__ == "__main__":
    sys.exit(main())
