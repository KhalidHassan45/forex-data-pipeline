#!/usr/bin/env python3
"""
Forex strategy lab — walk-forward backtests over the Parquet files produced by
forex-data-pipeline.

    python run_backtest.py                                   # all strategies × all pairs, h1
    python run_backtest.py --pairs EURUSD XAUUSD --strategies sma_cross donchian_breakout
    python run_backtest.py --tf d1 --train-years 4 --test-years 1
    python run_backtest.py --spread-mult 1.5                 # stress costs

Outputs → $DATA_DIR/reports/<UTC timestamp>/
    results.csv     one row per strategy × pair (walk-forward + in-sample metrics, verdict)
    summary.json    machine-readable digest for Hermes
    report.html     human report (Arabic)
    equity/*.csv    out-of-sample equity curves
"""
import argparse
import html
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from engine import apply_atr_stop, metrics, run, walk_forward  # noqa: E402
from strategies import INTRADAY_ONLY, STOP_GRID, STRATEGIES  # noqa: E402

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# ---- pass criteria (all must hold on the stitched OUT-OF-SAMPLE result) -------------
CRITERIA = {
    "min_sharpe": 0.5,
    "min_profit_factor": 1.15,
    "min_trades": 100,
    "max_drawdown_pct": -25.0,
    "min_oos_years": 3,
    "min_cost_stress_sharpe": 0.2,   # still positive-ish with 1.5× spreads
}


def load(pair: str, tf: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "parquet" / f"{pair.lower()}_{tf}.parquet")
    df = df.set_index("ts").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close"]].astype(float)


def key(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items())


def evaluate(df, pair, sname, train_years, test_years, spread_mult):
    fn, grid, family = STRATEGIES[sname]
    nets, helds, stress = {}, {}, {}
    for g in grid:
        raw = fn(df, **g)
        for stop in STOP_GRID:
            p = dict(g, stop=stop)
            pos = apply_atr_stop(df, raw, stop)
            k = key(p)
            nets[k], helds[k] = run(df, pos, pair, spread_mult)
            stress[k] = run(df, pos, pair, spread_mult * 1.5)[0]

    min_tr = max(10, int(20 * train_years))   # need enough trades to trust a train window
    wf_net, wf_held, chosen = walk_forward(nets, helds, train_years, test_years, min_tr)

    # in-sample "best on everything" — shows how much optimism walk-forward removes
    is_best = max(nets, key=lambda k: metrics(nets[k], helds[k]).get("sharpe", -9))
    is_m = metrics(nets[is_best], helds[is_best])

    row = {"strategy": sname, "family": family, "pair": pair, "combos_tested": len(nets),
           "is_best_params": is_best, **{f"is_{k}": v for k, v in is_m.items()}}
    if wf_net is None:
        row.update(verdict="NO_DATA", reasons="not enough history for walk-forward")
        return row, None

    wf_m = metrics(wf_net, wf_held)
    # cost stress: same chosen params per window, 1.5× spread
    st_parts = []
    for c in chosen:
        lo = pd.Timestamp(f"{c['test_from']}-01-01", tz="UTC")
        hi = pd.Timestamp(f"{c['test_from'] + test_years}-01-01", tz="UTC") - pd.Timedelta("1ns")
        st_parts.append(stress[c["params"]][lo:hi])
    st_m = metrics(pd.concat(st_parts), wf_held)

    row.update({f"wf_{k}": v for k, v in wf_m.items()})
    row["stress_sharpe"] = st_m.get("sharpe", 0)
    row["params_by_year"] = " | ".join(f"{c['test_from']}:{c['params']}" for c in chosen)
    row["param_changes"] = sum(1 for a, b in zip(chosen, chosen[1:]) if a["params"] != b["params"])

    fails = []
    if not wf_m or wf_m.get("sharpe") is None or pd.isna(wf_m.get("sharpe")):
        fails.append("no walk-forward trades")
    else:
        if wf_m["sharpe"] < CRITERIA["min_sharpe"]: fails.append(f"sharpe {wf_m['sharpe']}")
        if wf_m.get("profit_factor", 0) < CRITERIA["min_profit_factor"]: fails.append(f"PF {wf_m.get('profit_factor', 0)}")
        if wf_m.get("trades", 0) < CRITERIA["min_trades"]: fails.append(f"trades {wf_m.get('trades', 0)}")
        if wf_m.get("max_dd_pct", 0) < CRITERIA["max_drawdown_pct"]: fails.append(f"DD {wf_m.get('max_dd_pct', 0)}%")
        if wf_m.get("years", 0) < CRITERIA["min_oos_years"]: fails.append(f"OOS years {wf_m.get('years', 0)}")
        if row.get("stress_sharpe", 0) < CRITERIA["min_cost_stress_sharpe"]: fails.append(f"stress sharpe {row.get('stress_sharpe', 0)}")
    row["verdict"] = "PASS" if not fails else "FAIL"
    row["reasons"] = "; ".join(fails)
    return row, (1 + wf_net).cumprod()


def html_report(res: pd.DataFrame, meta: dict) -> str:
    cols = ["strategy", "pair", "verdict", "wf_sharpe", "wf_cagr_pct", "wf_max_dd_pct", "wf_profit_factor",
            "wf_trades", "wf_win_rate_pct", "stress_sharpe", "is_sharpe", "reasons"]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for _, r in res.sort_values(["verdict", "wf_sharpe"], ascending=[False, False]).iterrows():
        cls = "pass" if r.get("verdict") == "PASS" else "fail"
        body += f"<tr class='{cls}'>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
    board = res.groupby("strategy")["verdict"].apply(lambda s: f"{(s == 'PASS').sum()}/{len(s)}").to_dict()
    board_html = "".join(f"<li><b>{k}</b>: {v} أزواج اجتازت</li>" for k, v in board.items())
    return f"""<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8">
<title>Forex Strategy Lab</title>
<style>
body{{font-family:system-ui,Segoe UI,Tahoma,sans-serif;margin:24px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:13px;direction:ltr}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left}} th{{background:#f0f0f0}}
tr.pass td{{background:#eaf6ec}} .warn{{background:#fff4e5;border:1px solid #f0c27b;padding:12px;border-radius:6px}}
</style>
<h1>تقرير مختبر الاستراتيجيات — Walk-forward</h1>
<p>الإطار: <b>{meta['tf']}</b> | تدريب {meta['train_years']} سنوات / اختبار {meta['test_years']} سنة |
مضاعف الـ spread: {meta['spread_mult']} | التوليفات المختبرة: {meta['total_combos']} | {meta['generated']}</p>
<div class="warn">النتائج <b>خارج العينة</b> (out-of-sample) فقط هي المعتمدة. كثرة التوليفات المختبرة تعني أن بعض
"النجاح" قد يكون صدفة إحصائية؛ الاستراتيجية الجديرة بالمتابعة تنجح على <b>عدة أزواج</b> وتصمد أمام مضاعفة التكاليف.
لا شيء هنا ضمان لأداء مستقبلي، والخطوة التالية لأي ناجح هي Paper trading لا التداول الحقيقي.</div>
<h2>لوحة الصدارة</h2><ul>{board_html}</ul>
<h2>النتائج</h2><table><tr>{head}</tr>{body}</table>
<p>is_* = أفضل توليفة على كامل البيانات (متفائل). wf_* = walk-forward خارج العينة (الواقعي). الفرق بينهما = مقدار الـ overfitting.</p>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+")
    ap.add_argument("--tf", default=os.getenv("FOREX_TF", "h1"))
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--test-years", type=int, default=1)
    ap.add_argument("--spread-mult", type=float, default=1.0)
    ap.add_argument("--out", default=str(DATA_DIR / "reports"))
    a = ap.parse_args()

    pq = DATA_DIR / "parquet"
    pairs = [p.upper() for p in a.pairs] if a.pairs else sorted(
        f.name.split("_")[0].upper() for f in pq.glob(f"*_{a.tf}.parquet"))
    if not pairs:
        print(f"no parquet files for tf={a.tf} in {pq}")
        return 2

    out = Path(a.out) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out / "equity").mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for pair in pairs:
        try:
            df = load(pair, a.tf)
        except FileNotFoundError:
            print(f"skip {pair}: no parquet")
            continue
        for s in a.strategies:
            if s in INTRADAY_ONLY and a.tf in ("d1", "h4"):
                continue
            t = time.time()
            row, eq = evaluate(df, pair, s, a.train_years, a.test_years, a.spread_mult)
            rows.append(row)
            if eq is not None:
                eq.rename("equity").to_csv(out / "equity" / f"{s}_{pair}.csv")
            print(f"{pair:7} {s:20} {row['verdict']:5} wf_sharpe={row.get('wf_sharpe', '-'):>6} "
                  f"is_sharpe={row.get('is_sharpe', '-'):>6} ({time.time() - t:.1f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    meta = {"tf": a.tf, "train_years": a.train_years, "test_years": a.test_years,
            "spread_mult": a.spread_mult, "total_combos": int(res["combos_tested"].sum()),
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "runtime_sec": round(time.time() - t0, 1), "criteria": CRITERIA}
    passed = res[res["verdict"] == "PASS"]
    summary = {
        **meta,
        "pairs": pairs,
        "passed": passed[["strategy", "pair", "wf_sharpe", "wf_cagr_pct", "wf_max_dd_pct",
                          "wf_profit_factor", "wf_trades", "stress_sharpe"]].to_dict("records"),
        "strategy_board": res.groupby("strategy")["verdict"].apply(lambda s: int((s == "PASS").sum())).to_dict(),
        "avg_is_minus_wf_sharpe": round(float((res.get("is_sharpe", 0) - res.get("wf_sharpe", 0)).mean()), 2)
        if "wf_sharpe" in res else None,
        "report_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    (out / "report.html").write_text(html_report(res, meta), encoding="utf-8")
    (Path(a.out) / "LATEST").write_text(str(out))
    print(f"\n{len(passed)}/{len(res)} PASS → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
