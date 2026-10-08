#!/usr/bin/env python3
"""
Build $DATA_DIR/dashboard/data.json from everything the lab produces:
  parquet/            → data health per pair
  logs/last_run.json  → last pipeline run
  reports/<run>/      → strategy-lab runs (results.csv, summary.json, equity/*.csv)
  research/           → vault.json, registry.jsonl, runs/*/summary.json, promoted/*.json
  imports/registry.jsonl → strategy importer

Then copies index.html next to it. Safe to run any time (read-only on inputs, atomic write).
    python build_dashboard.py [--keep-runs 6]
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
SANDBOX = Path(os.getenv("SANDBOX_DIR", "/sandbox"))   # uploads tested by lab-runner (read-only here)
OUT = DATA_DIR / "dashboard"
HERE = Path(__file__).resolve().parent


def jl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def jload(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):
        return clean(v.item())
    return v


def lin(x, bad, good):
    """0 at `bad`, 1 at `good`, clipped — works for either direction."""
    if x is None:
        return 0.0
    t = (x - bad) / (good - bad)
    return max(0.0, min(1.0, t))


# ---------------------------------------------------------------- pipeline
def pipeline():
    pq = DATA_DIR / "parquet"
    pairs = []
    for f in sorted(pq.glob("*.parquet")):
        parts = f.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            df = pd.read_parquet(f, columns=["ts", "high", "low", "close"])
        except Exception:
            continue
        ts = pd.to_datetime(df["ts"], utc=True).sort_values()
        gaps = ts.diff()
        pairs.append({
            "pair": parts[0].upper(), "tf": parts[1], "rows": int(len(df)),
            "first": ts.iloc[0].isoformat() if len(ts) else None,
            "last": ts.iloc[-1].isoformat() if len(ts) else None,
            "gaps_gt_3d": int((gaps > pd.Timedelta("3D")).sum()),
            "bad_candles": int(((df["high"] < df["low"]) | (df["close"] <= 0)).sum()),
            "size_mb": round(f.stat().st_size / 1e6, 2),
        })
    lr = jload(DATA_DIR / "logs" / "last_run.json", {}) or {}
    return {"pairs": pairs, "last_run": {k: lr.get(k) for k in ("mode", "tf", "started", "finished", "failed")}}


# ---------------------------------------------------------------- strategy lab
SCORE_PARTS = [  # (key, label, column, bad, good, weight)
    ("sharpe", "Sharpe خارج العينة", "wf_sharpe", 0.0, 1.0, 2),
    ("pf", "Profit factor", "wf_profit_factor", 1.0, 1.5, 1),
    ("dd", "أقصى تراجع", "wf_max_dd_pct", -40.0, -10.0, 1),
    ("trades", "عدد الصفقات", "wf_trades", 30, 200, 1),
    ("stress", "الصمود أمام التكاليف ×1.5", "stress_sharpe", 0.0, 0.8, 1),
    ("gap", "فجوة التفاؤل (IS − WF)", "_gap", 1.5, 0.0, 1),
    ("breadth", "الاتساع (أزواج ناجحة)", "_breadth", 0, 3, 2),
]


def score(row):
    parts, tot, wsum = [], 0.0, 0
    for key, label, col, bad, good, w in SCORE_PARTS:
        v = row.get(col)
        s = lin(v, bad, good) if v is not None else 0.0
        parts.append({"key": key, "label": label, "value": clean(v), "score": round(s, 2)})
        tot += s * w
        wsum += w
    return round(100 * tot / wsum), parts


def downsample(eq: pd.Series, n=260):
    if len(eq) > n:
        eq = eq.iloc[:: max(1, len(eq) // n)]
    return [[t.strftime("%Y-%m-%d"), round(float(v), 4)] for t, v in eq.items()]


def lab(keep):
    dirs = [d for root in (DATA_DIR / "reports", SANDBOX / "reports") if root.exists()
            for d in root.glob("*") if (d / "results.csv").exists()]
    dirs.sort(key=lambda d: d.name, reverse=True)
    out = {"runs": [], "latest": None}
    n_full = n_single = 0
    for d in dirs:
        try:
            res = pd.read_csv(d / "results.csv")
        except Exception:
            continue
        single = res["strategy"].nunique() <= 2
        if (single and n_single >= 30) or (not single and n_full >= keep):
            continue
        n_single += single
        n_full += not single
        summ = jload(d / "summary.json", {}) or {}
        res = res.astype(object).where(pd.notna(res), None)
        rows = [{k: clean(v) for k, v in r.items()} for r in res.to_dict("records")]
        passes = Counter(r["strategy"] for r in rows if r.get("verdict") == "PASS")
        for r in rows:
            r["_breadth"] = passes.get(r["strategy"], 0)
            r["_gap"] = (r["is_sharpe"] - r["wf_sharpe"]) if r.get("is_sharpe") is not None and r.get("wf_sharpe") is not None else None
            r["score"], r["score_parts"] = score(r) if r.get("verdict") in ("PASS", "FAIL") else (None, [])
            r["imported"] = str(r["strategy"]).startswith("imp_")
        eq = {}
        for f in (d / "equity").glob("*.csv"):
            try:
                s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
                strat, pair = f.stem.rsplit("_", 1)
                eq[f"{strat}|{pair}"] = downsample(s)
            except Exception:
                continue
        by_strat = defaultdict(list)
        for r in rows:
            by_strat[r["strategy"]].append(r)
        leaderboard = []
        for s, rs in by_strat.items():
            sh = sorted(x["wf_sharpe"] for x in rs if x.get("wf_sharpe") is not None)
            leaderboard.append({
                "strategy": s, "family": rs[0].get("family"), "pairs": len(rs),
                "passed": sum(x.get("verdict") == "PASS" for x in rs),
                "median_wf_sharpe": round(sh[len(sh) // 2], 2) if sh else None,
                "best_score": max((x["score"] or 0) for x in rs),
            })
        leaderboard.sort(key=lambda x: (-x["passed"], -(x["median_wf_sharpe"] or -9)))
        out["runs"].append({
            "id": d.name,
            "meta": {k: summ.get(k) for k in ("tf", "train_years", "test_years", "spread_mult",
                                                "total_combos", "generated", "runtime_sec", "criteria",
                                                "avg_is_minus_wf_sharpe")},
            "rows": rows, "equity": eq, "leaderboard": leaderboard,
        })
    # default view = newest FULL lab run (import tests produce single-strategy runs)
    full = [r for r in out["runs"] if len(r["leaderboard"]) >= 3]
    out["latest"] = (full or out["runs"] or [{"id": None}])[0]["id"]
    return out


# ---------------------------------------------------------------- research
def research():
    R = DATA_DIR / "research"
    vault = jload(R / "vault.json", {}) or {}
    reg = jl(R / "registry.jsonl")
    scans = [r for r in reg if r.get("event") == "SCAN"]
    latest = {}
    for r in scans:
        latest[r["id"]] = r
    vault_rows = [r for r in reg if r.get("event") in ("VAULT_PASS", "VAULT_FAIL")]
    used = {r["id"] for r in reg if r.get("event") == "VAULT"}
    pending = [r for r in latest.values() if r.get("status") == "CANDIDATE" and r["id"] not in used]
    ts = [abs(r["t"]) for r in latest.values() if isinstance(r.get("t"), (int, float))]
    bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10]
    hist = []
    for lo, hi in zip(bins, bins[1:] + [None]):
        n = sum(1 for t in ts if t >= lo and (hi is None or t < hi))
        hist.append({"lo": lo, "hi": hi, "n": n})
    runs = []
    for d in sorted((R / "runs").glob("*"), reverse=True)[:12]:
        s = jload(d / "summary.json")
        if s:
            runs.append({k: s.get(k) for k in ("run_id", "tested_this_run", "bh_significant",
                                                "expected_false_positives_at_5pct_without_correction",
                                                "lifetime_tests")} | {"candidates": len(s.get("candidates", []))})
    fam = defaultdict(lambda: {"tested": 0, "candidates": 0})
    for r in latest.values():
        fam[r.get("family", "?")]["tested"] += 1
        fam[r.get("family", "?")]["candidates"] += r.get("status") == "CANDIDATE"
    promoted = [jload(p) for p in sorted((R / "promoted").glob("*.json"))] if (R / "promoted").exists() else []
    return {
        "vault": {"start": vault.get("vault_start"), "fixed_at": vault.get("fixed_at"),
                  "unlocks": len(vault_rows), "limit": 10},
        "lifetime_tests": len(scans), "unique_hypotheses": len(latest),
        "t_hist": hist, "t_threshold": 3.0, "runs": runs,
        "families": [{"family": k, **v} for k, v in sorted(fam.items())],
        "pending": sorted(({k: clean(r.get(k)) for k in ("id", "family", "pair", "direction", "t", "stability",
                                                         "net_sharpe", "run_id")} for r in pending),
                          key=lambda x: -(x["t"] or 0)),
        "vault_history": [{k: clean(r.get(k)) for k in ("id", "event", "ts", "approved_by", "t", "net_sharpe",
                                                         "net_profit_factor", "net_max_dd_pct", "why")}
                          for r in vault_rows][::-1],
        "promoted": [p for p in promoted if p],
    }


# ---------------------------------------------------------------- imports
def imports():
    reg = jl(DATA_DIR / "imports" / "registry.jsonl") + jl(SANDBOX / "imports" / "registry.jsonl")
    items = {}
    for r in reg:
        it = items.setdefault(r["name"], {"name": r["name"]})
        if r["event"] == "REGISTER":
            it.update({k: r.get(k) for k in ("url", "source", "tier", "lang", "author", "license", "usage", "claim", "ts")})
            it["status"] = "REGISTERED"
        elif r["event"] == "CHECK":
            it["status"] = r["status"]
            it["flags"] = r.get("flags", [])
        elif r["event"] == "TESTED":
            it["status"] = "TESTED"
            it.update({k: r.get(k) for k in ("verdict", "pairs_passed", "pairs_tested", "best_wf_sharpe", "report")})
    board = defaultdict(lambda: {"imported": 0, "blocked": 0, "survivor": 0, "weak": 0, "fail": 0, "pending": 0})
    flags = Counter()
    for it in items.values():
        b = board[(it.get("source", "?"), it.get("tier", 3))]
        b["imported"] += 1
        if it.get("status") == "BLOCKED":
            b["blocked"] += 1
        elif it.get("verdict") == "SURVIVOR":
            b["survivor"] += 1
        elif it.get("verdict") == "WEAK":
            b["weak"] += 1
        elif it.get("verdict") == "FAIL":
            b["fail"] += 1
        else:
            b["pending"] += 1
        for f in it.get("flags") or []:
            flags[(f["flag"], f["severity"], f["why"])] += 1
    return {
        "items": sorted(items.values(), key=lambda x: x.get("ts") or "", reverse=True),
        "board": sorted(({"source": s, "tier": t, **v} for (s, t), v in board.items()),
                        key=lambda x: (x["tier"], -x["survivor"], x["source"])),
        "flags": [{"flag": k, "severity": s, "why": w, "count": n} for (k, s, w), n in flags.most_common()],
        "lifetime_tested": sum(1 for r in reg if r["event"] == "TESTED"),
    }


SUB_FIELDS = ("id", "filename", "kind", "lang", "route", "name", "source", "url", "claim", "created", "state",
              "timeline", "summary_ar", "ambiguities", "red_flags", "static_flags", "describe", "error", "confidence",
              "reason_unsupported", "import_name", "verdict", "pairs_passed", "pairs_tested", "results",
              "best_wf_sharpe", "report", "model")


def submissions(limit=40):
    inbox = SANDBOX / "inbox"
    if not inbox.exists():
        return []
    out = []
    for d in sorted(inbox.iterdir(), reverse=True)[:limit]:
        st = jload(d / "status.json")
        if st:
            out.append({k: st.get(k) for k in SUB_FIELDS if k in st})
    return out


# ---------------------------------------------------------------- ml-lab:v1
def ml():
    M = DATA_DIR / "ml"
    reg = jl(M / "registry.jsonl")
    trains = [r for r in reg if r.get("event") == "TRAIN"]
    vaulted = {r["id"] for r in reg if r.get("event") == "VAULT"}
    verdicts = [r for r in reg if r.get("event") in ("VAULT_PASS", "VAULT_FAIL")]
    keep = ("id", "pair", "ts", "status", "fails", "why", "net_trades", "net_sharpe", "net_profit_factor",
            "net_max_dd_pct", "net_win_rate", "dsr", "null_sharpe", "edge_over_null", "stability", "stress_sharpe",
            "n_trials", "by_year_pct", "top_features", "label_dist", "feature_version", "quantile", "meta")
    rows = []
    for r in trains[::-1][:60]:
        x = {k: clean(r.get(k)) if not isinstance(r.get(k), dict) else r.get(k) for k in keep}
        x["vaulted"] = r["id"] in vaulted
        if r.get("status") == "CANDIDATE" or len(rows) < 8:
            x["equity"] = jload(Path(r.get("run_dir", "")) / "equity.json", []) or []
        rows.append(x)
    gates = defaultdict(int)
    for r in trains:
        for f in filter(None, (r.get("fails") or "").split(",")):
            gates[f.split("=")[0]] += 1
    certs = jload(M / "leakage_cert.json", {}) or {}
    cert = sorted(certs.values(), key=lambda c: c.get("ts", ""))[-1] if certs else None
    paper = jload(M / "paper" / "status.json", {}) or {}
    return {
        "lifetime": len(trains), "candidates": sum(r.get("status") == "CANDIDATE" for r in trains),
        "pending": [x for x in rows if x["status"] == "CANDIDATE" and not x["vaulted"]],
        "rows": rows, "gates": dict(sorted(gates.items(), key=lambda kv: -kv[1])),
        "vault": [{k: clean(r.get(k)) for k in ("id", "event", "ts", "approved_by", "t", "net_sharpe",
                                                 "net_profit_factor", "net_max_dd_pct", "net_trades", "why")}
                  for r in verdicts][::-1],
        "cert": cert, "paper": paper.get("models", []), "paper_ts": paper.get("ts"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-runs", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample": os.getenv("DASHBOARD_SAMPLE") == "1",
        "pipeline": pipeline(), "lab": lab(a.keep_runs), "research": research(), "imports": imports(),
        "submissions": submissions(), "ml": ml(),
    }
    tmp = OUT / "data.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str))
    tmp.replace(OUT / "data.json")
    page = (HERE / "index.html").read_text(encoding="utf-8")
    if not page.lstrip().lower().startswith("<!doctype"):
        page = ('<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                '</head><body>' + page + '</body></html>')
    tmp_html = OUT / "index.html.tmp"
    tmp_html.write_text(page, encoding="utf-8")
    tmp_html.replace(OUT / "index.html")
    kb = (OUT / "data.json").stat().st_size / 1024
    print(f"dashboard built → {OUT} ({kb:.0f} KB, {len(data['lab']['runs'])} lab runs)")


if __name__ == "__main__":
    main()
