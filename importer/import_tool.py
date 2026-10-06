#!/usr/bin/env python3
"""
Strategy importer — bring third-party strategies (Pine Script, MQL4/5, papers, forum posts)
into the forex-backtest lab with provenance, red-flag screening and a source leaderboard.

Commands
  register  --name rsi2_connors --url URL --source quantpedia --lang pine --license MPL-2.0
            --author X [--claim "70% win rate"] [--original path/to/original.pine]
  check     --name rsi2_connors            static red flags on the original + look-ahead test on the Python port
  test      --name rsi2_connors [--pairs EURUSD GBPUSD] [--tf h1]   runs the lab on this strategy only
  list                                      all imports and their status
  board                                     leaderboard: which sources produce survivors

Layout ($DATA_DIR/imports/)
  registry.jsonl                 append-only provenance + events
  originals/<name>.<ext>         untouched original code / text
  strategies/<name>.py           Python port exposing STRATEGY = (fn, grid, family)
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
IMP = DATA_DIR / "imports"
REG = IMP / "registry.jsonl"
HERE = Path(__file__).resolve().parent
for cand in (HERE.parent / "backtest", Path("/app/backtest")):
    if cand.exists():
        BACKTEST = cand
        sys.path.insert(0, str(cand))
        break
else:
    sys.exit("forex-backtest engine not found (/app/backtest)")

SOURCES = {  # tier 1 = highest evidence quality
    "quantpedia": 1, "ssrn": 1, "arxiv": 1, "quantocracy": 1,
    "quantconnect": 2, "tradingview": 2, "mql5": 2, "github": 2,
    "forexfactory": 3, "babypips": 3, "quantifiedstrategies": 3, "other": 3,
}
MAX_GRID = 8
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")

RED_FLAGS = {
    # pattern, severity, explanation (Arabic, shown to Khalid)
    "martingale": (r"martingal|double\s*(the\s*)?lot|lot\w*\s*\*=?\s*[2-9]|lot_?mult|multiplier\s*=",
                   "BLOCK", "مضاعفة الحجم بعد الخسارة (Martingale) — تربح طويلًا ثم تمسح الحساب"),
    "grid": (r"\bgrid\b|grid_?step|distance_?between_?orders",
             "BLOCK", "شبكة أوامر (Grid) — خسارة غير محدودة في السوق ذي الاتجاه"),
    "averaging": (r"averag(e|ing)\s*down|recovery\s*(mode|zone)|cost\s*averag",
                  "BLOCK", "تعزيز المراكز الخاسرة (Averaging/Recovery)"),
    "hedge_lock": (r"\bhedg(e|ing)\b|lock(ing)?\s*position",
                   "WARN", "قفل/تحوّط المراكز — يخفي الخسارة ولا يلغيها"),
    "pine_lookahead": (r"lookahead\s*=\s*barmerge\.lookahead_on|lookahead_on",
                       "BLOCK", "Pine: lookahead_on — يقرأ بيانات مستقبلية (repainting)"),
    "pine_security": (r"request\.security\s*\(|\bsecurity\s*\(",
                      "WARN", "Pine: request.security — تحقق من عدم إعادة الرسم وأن الإطار الأعلى مُزاح"),
    "pine_tick": (r"calc_on_every_tick\s*=\s*true",
                  "WARN", "Pine: calc_on_every_tick — النتائج التاريخية لا تطابق الحي"),
    "pyramiding": (r"pyramiding\s*=\s*([2-9]|\d{2,})",
                   "WARN", "Pyramiding — إضافة صفقات متعددة بنفس الاتجاه، يضخم المخاطرة"),
    "mql_current_bar": (r"(Close|High|Low|Open)\s*\[\s*0\s*\]|iClose\([^)]*,\s*0\s*\)",
                        "WARN", "MQL: الشمعة [0] لم تُغلق بعد — قد تعتمد الإشارة على سعر يتغير"),
    "no_stop": (None, "WARN", "لا يوجد Stop Loss واضح في الكود الأصلي"),
    "paid_claims": (r"guarantee|risk[- ]?free|100\s*%|never\s*lose|holy\s*grail|مضمون",
                    "WARN", "وعود تسويقية (مضمون/بلا مخاطرة) — علامة تحذير قوية"),
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registry():
    return [json.loads(l) for l in REG.read_text().splitlines() if l.strip()] if REG.exists() else []


def append(row):
    IMP.mkdir(parents=True, exist_ok=True)
    with REG.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def get(name):
    rows = [r for r in registry() if r["name"] == name]
    if not rows or rows[0]["event"] != "REGISTER":
        sys.exit(f"unknown import '{name}' — register it first")
    return rows


def load_port(name):
    path = IMP / "strategies" / f"{name}.py"
    if not path.exists():
        sys.exit(f"missing Python port: {path}")
    spec = importlib.util.spec_from_file_location(f"imp_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn, grid, family = mod.STRATEGY
    return fn, grid, family


# ---------------------------------------------------------------- commands
def cmd_register(a):
    if not NAME_RE.match(a.name):
        sys.exit("name must be snake_case, 3–41 chars, start with a letter")
    if any(r["name"] == a.name for r in registry()):
        sys.exit(f"'{a.name}' already registered — imports are immutable; use a new name for a variant")
    src = a.source.lower()
    if src not in SOURCES:
        sys.exit(f"source must be one of {list(SOURCES)}")
    orig = None
    if a.original:
        p = Path(a.original)
        dest = IMP / "originals" / f"{a.name}{p.suffix or '.txt'}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        orig = str(dest)
    license_ = a.license or "UNKNOWN"
    append({"event": "REGISTER", "name": a.name, "ts": now(), "url": a.url, "source": src,
            "tier": SOURCES[src], "lang": a.lang, "author": a.author, "license": license_,
            "usage": "personal-testing-only" if license_.upper() in ("UNKNOWN", "PROPRIETARY", "")
                     else f"check license {license_} before any commercial use",
            "claim": a.claim, "original": orig})
    (IMP / "strategies").mkdir(parents=True, exist_ok=True)
    print(f"registered {a.name} (tier {SOURCES[src]}). Next: write {IMP / 'strategies' / (a.name + '.py')} then `check`.")


def static_flags(text):
    found = []
    low = text.lower()
    for key, (pat, sev, why) in RED_FLAGS.items():
        if key == "no_stop":
            if not re.search(r"stop\s*_?loss|\bsl\b|stoploss|strategy\.exit|stop\s*=", low):
                found.append((key, sev, why))
        elif re.search(pat, low, re.I):
            found.append((key, sev, why))
    return found


def causality(fn, grid):
    import numpy as np
    import pandas as pd
    pq = sorted((DATA_DIR / "parquet").glob("*_h1.parquet"))
    if not pq:
        return False, "no parquet data to test against"
    df = pd.read_parquet(pq[0]).set_index("ts").sort_index()[["open", "high", "low", "close"]].astype(float)
    df = df.iloc[-20000:]
    for g in grid:
        a = fn(df, **g)
        if not isinstance(a, pd.Series) or len(a) != len(df):
            return False, f"{g}: must return a Series aligned to df"
        bad = set(np.unique(a.dropna())) - {-1.0, 0.0, 1.0}
        if bad:
            return False, f"{g}: values outside -1/0/1: {sorted(bad)[:5]}"
        b = fn(df.iloc[:-500], **g)
        if not (a.iloc[:-500].fillna(0).to_numpy() == b.fillna(0).to_numpy()).all():
            return False, f"{g}: LOOK-AHEAD — past signals change when future bars are removed"
    return True, f"{len(grid)} configs causal"


def cmd_check(a):
    rows = get(a.name)
    reg = rows[0]
    flags = []
    if reg.get("original") and Path(reg["original"]).exists():
        flags = static_flags(Path(reg["original"]).read_text(errors="ignore"))
    if reg.get("claim"):
        flags += [f for f in static_flags(reg["claim"]) if f[0] == "paid_claims"]
    port_src = (IMP / "strategies" / f"{a.name}.py").read_text()
    if re.search(r"shift\(\s*-|center\s*=\s*True", port_src):
        flags.append(("port_lookahead", "BLOCK", "الترجمة تستخدم shift(-n) أو center=True"))
    fn, grid, family = load_port(a.name)
    if len(grid) > MAX_GRID:
        flags.append(("grid_size", "BLOCK", f"شبكة المعاملات {len(grid)} > {MAX_GRID} — خطر overfitting"))
    ok, msg = causality(fn, grid)
    if not ok:
        flags.append(("causality", "BLOCK", msg))
    flags = list(dict.fromkeys(flags))          # de-duplicate (same flag from code and claim)
    blocked = [f for f in flags if f[1] == "BLOCK"]
    status = "BLOCKED" if blocked else "READY"
    append({"event": "CHECK", "name": a.name, "ts": now(), "status": status,
            "flags": [{"flag": k, "severity": s, "why": w} for k, s, w in flags],
            "grid_size": len(grid), "family": family, "causality": msg})
    print(f"{a.name}: {status}  ({msg})")
    for k, s, w in flags:
        print(f"  [{s}] {k}: {w}")
    return 0 if status == "READY" else 1


def cmd_test(a):
    rows = get(a.name)
    checks = [r for r in rows if r["event"] == "CHECK"]
    if not checks or checks[-1]["status"] != "READY":
        sys.exit(f"refused: '{a.name}' has not passed `check`")
    if any(r["event"] == "TESTED" for r in rows) and not a.again:
        sys.exit(f"refused: '{a.name}' already tested — re-testing the same import inflates false discoveries "
                 "(use --again only after Khalid approves, it is logged)")
    import strategies  # forex-backtest registry (same dict object run_backtest uses)
    import run_backtest
    key = f"imp_{a.name}"
    strategies.STRATEGIES[key] = load_port(a.name)
    run_backtest.STRATEGIES[key] = strategies.STRATEGIES[key]
    argv = ["run_backtest.py", "--strategies", key, "--tf", a.tf]
    if a.pairs:
        argv += ["--pairs", *a.pairs]
    sys.argv = argv
    rc = run_backtest.main()
    if rc != 0:
        return rc
    import pandas as pd
    out = Path((DATA_DIR / "reports" / "LATEST").read_text().strip())
    res = pd.read_csv(out / "results.csv")
    passed = res.loc[res["verdict"] == "PASS", "pair"].tolist()
    lifetime = sum(1 for r in registry() if r["event"] == "TESTED") + 1
    verdict = "SURVIVOR" if len(passed) >= 3 else ("WEAK" if passed else "FAIL")
    append({"event": "TESTED", "name": a.name, "ts": now(), "tf": a.tf, "report": str(out),
            "pairs_tested": int(len(res)), "pairs_passed": passed, "verdict": verdict,
            "best_wf_sharpe": float(res["wf_sharpe"].max()) if "wf_sharpe" in res else None,
            "retest": bool(a.again), "lifetime_imports_tested": lifetime})
    print(f"\n{a.name}: {verdict} — passed on {len(passed)}/{len(res)} pairs {passed} "
          f"(imports tested so far: {lifetime})")
    if verdict == "WEAK":
        print("note: PASS on 1–2 pairs only is most likely chance — not a candidate")
    return 0


def cmd_list(_):
    latest = {}
    for r in registry():
        latest.setdefault(r["name"], {}).update({k: v for k, v in r.items() if k != "event"} | {"last": r["event"]})
    for n, r in latest.items():
        print(f"{n:28} tier{r.get('tier')} {r.get('source'):18} {r.get('last'):9} "
              f"{r.get('status', '')} {r.get('verdict', '')}")


def cmd_board(_):
    reg = registry()
    srcs = {r["name"]: (r["source"], r["tier"]) for r in reg if r["event"] == "REGISTER"}
    stats = {}
    for r in reg:
        s = stats.setdefault(srcs.get(r["name"], ("?", 3)), {"imported": 0, "blocked": 0, "tested": 0,
                                                             "survivor": 0, "weak": 0})
        if r["event"] == "REGISTER": s["imported"] += 1
        if r["event"] == "CHECK" and r["status"] == "BLOCKED": s["blocked"] += 1
        if r["event"] == "TESTED":
            s["tested"] += 1
            s["survivor"] += r["verdict"] == "SURVIVOR"
            s["weak"] += r["verdict"] == "WEAK"
    print(f"{'source':22}{'tier':>5}{'imported':>10}{'blocked':>9}{'tested':>8}{'survivor':>10}{'weak':>6}")
    for (src, tier), s in sorted(stats.items(), key=lambda x: (x[0][1], -x[1]["survivor"])):
        print(f"{src:22}{tier:>5}{s['imported']:>10}{s['blocked']:>9}{s['tested']:>8}{s['survivor']:>10}{s['weak']:>6}")
    total = sum(s["tested"] for s in stats.values())
    print(f"\nimports tested lifetime: {total} — at 5% chance level ≈{total * 0.05:.1f} would 'pass' by luck alone")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--name", required=True); r.add_argument("--url", required=True)
    r.add_argument("--source", required=True); r.add_argument("--lang", required=True,
                   choices=["pine", "mql4", "mql5", "python", "paper", "text"])
    r.add_argument("--author", default="unknown"); r.add_argument("--license")
    r.add_argument("--claim"); r.add_argument("--original")
    for c in ("check", "test"):
        p = sub.add_parser(c)
        p.add_argument("--name", required=True)
        if c == "test":
            p.add_argument("--pairs", nargs="+"); p.add_argument("--tf", default="h1")
            p.add_argument("--again", action="store_true")
    sub.add_parser("list"); sub.add_parser("board")
    a = ap.parse_args()
    return {"register": cmd_register, "check": cmd_check, "test": cmd_test,
            "list": cmd_list, "board": cmd_board}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
