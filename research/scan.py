#!/usr/bin/env python3
"""
Pattern scanner — tests many hypotheses on the RESEARCH window only (the vault is never read),
controls false discoveries, logs everything to the registry, and lists CANDIDATES.

    python scan.py                                  # all families, all pairs, budget 800
    python scan.py --families hour_of_day session_follow --budget 200
    python scan.py --retest                         # re-score hypotheses already in the registry

A hypothesis becomes a CANDIDATE only if ALL hold (research window):
    1. Benjamini-Hochberg significant at q=0.05 across this run
    2. |t| >= T_MIN (3.0)                         — Harvey-Liu-Zhu style hurdle for data-mined factors
    3. stability >= 0.70                          — right sign in at least 70% of years
    4. after realistic costs: net Sharpe >= 0.3 and profit factor >= 1.05
    5. at least 150 exposed days
"""
import argparse
import html
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from hypotheses import BUILTIN, FAMILIES, generate  # noqa: E402

RULES = {"bh_q": 0.05, "t_min": 3.0, "stability_min": 0.70, "net_sharpe_min": 0.3,
         "net_pf_min": 1.05, "min_days": 150}


def causal(h, R, s_full, cut=500) -> bool:
    """Custom hypotheses must not change past exposure when future bars are removed."""
    end = R[h.pair].index[-cut]
    R_cut = {p: df[df.index < end] for p, df in R.items()}
    s_cut = h.fn(R_cut).reindex(R_cut[h.pair].index).fillna(0.0)
    s_ref = s_full.reindex(R_cut[h.pair].index).fillna(0.0)
    return bool((s_cut.to_numpy() == s_ref.to_numpy()).all())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="h1")
    ap.add_argument("--pairs", nargs="+")
    ap.add_argument("--families", nargs="+", choices=list(FAMILIES))
    ap.add_argument("--budget", type=int, default=800, help="max hypotheses tested this run")
    ap.add_argument("--retest", action="store_true")
    a = ap.parse_args()

    D = C.load_all(a.tf, [p.upper() for p in a.pairs] if a.pairs else None)
    if not D:
        print("no parquet data — run forex-data-pipeline first")
        return 2
    vs = C.vault_start(D)
    R = {p: df[df.index < vs] for p, df in D.items()}          # research window ONLY
    window_end = str(vs.date())
    pairs = sorted(R)

    seen = {(r["id"], r.get("window_end")) for r in C.registry() if r.get("event") == "SCAN"}
    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out = C.RES_DIR / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)

    rows, t0, skipped = [], time.time(), 0
    for h in generate(R, pairs, a.families):
        if len(rows) >= a.budget:
            break
        if not a.retest and (h.id, window_end) in seen:
            skipped += 1
            continue
        try:
            s = h.fn(R)
            if h.family not in BUILTIN and not causal(h, R, s):
                print(f"REJECTED (look-ahead) {h.id}")
                C.append([{"event": "SCAN", "run_id": run_id, "ts": C.now(), "window_end": window_end,
                           "id": h.id, "family": h.family, "pair": h.pair, "status": "REJECTED",
                           "why": "look-ahead detected"}])
                continue
            st, _ = C.evaluate(s, R[h.pair], h.pair)
        except Exception as e:  # a broken custom hypothesis must not kill the run
            print(f"error {h.id}: {e}")
            continue
        if st is None:
            continue
        rows.append({"id": h.id, "family": h.family, "pair": h.pair,
                     "params": json.dumps(h.params, ensure_ascii=False), "desc": h.desc, **st})

    if not rows:
        print(f"nothing new to test (skipped {skipped} already in registry; use --retest)")
        return 0

    df = pd.DataFrame(rows)
    df["bh_pass"] = C.benjamini_hochberg(df["p"].tolist(), RULES["bh_q"])

    def judge(r):
        why = []
        if not r.bh_pass: why.append("BH")
        if abs(r.t) < RULES["t_min"]: why.append(f"t={r.t}")
        if r.stability < RULES["stability_min"]: why.append(f"stab={r.stability}")
        if r.net_sharpe < RULES["net_sharpe_min"]: why.append(f"netSR={r.net_sharpe}")
        if r.net_profit_factor < RULES["net_pf_min"]: why.append(f"PF={r.net_profit_factor}")
        if r.days < RULES["min_days"]: why.append(f"days={r.days}")
        return ("CANDIDATE", "") if not why else ("REJECTED", ",".join(why))

    df[["status", "why"]] = df.apply(lambda r: pd.Series(judge(r)), axis=1)
    df = df.sort_values(["status", "t"], ascending=[True, False])
    df.to_csv(out / "scan.csv", index=False)

    C.append({"event": "SCAN", "run_id": run_id, "ts": C.now(), "window_end": window_end,
              "id": r.id, "family": r.family, "pair": r.pair, "params": r.params,
              "direction": int(r.direction), "t": r.t, "p": r.p, "stability": r.stability,
              "net_sharpe": r.net_sharpe, "status": r.status, "why": r.why}
             for r in df.itertuples())

    lifetime = sum(1 for r in C.registry() if r.get("event") == "SCAN")
    cands = df[df["status"] == "CANDIDATE"]
    summary = {
        "run_id": run_id, "tf": a.tf, "pairs": pairs, "research_window_end": window_end,
        "tested_this_run": len(df), "skipped_already_tested": skipped,
        "lifetime_tests": lifetime, "rules": RULES,
        "expected_false_positives_at_5pct_without_correction": round(len(df) * 0.05, 1),
        "bh_significant": int(df["bh_pass"].sum()),
        "candidates": cands[["id", "desc", "direction", "t", "stability", "net_sharpe",
                             "net_profit_factor", "net_max_dd_pct", "net_trades"]].to_dict("records"),
        "runtime_sec": round(time.time() - t0, 1), "report_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    (out / "report.html").write_text(render(df, summary), encoding="utf-8")
    (C.RES_DIR / "LATEST").write_text(str(out))
    print(f"tested={len(df)} bh={summary['bh_significant']} candidates={len(cands)} "
          f"(lifetime tests {lifetime}) → {out}")
    return 0


def render(df, s):
    cols = ["status", "id", "desc", "direction", "t", "p", "stability", "net_sharpe",
            "net_profit_factor", "net_max_dd_pct", "net_trades", "why"]
    top = df.head(60)
    trs = "".join("<tr class='%s'>%s</tr>" % ("ok" if r["status"] == "CANDIDATE" else "",
                  "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols)) for _, r in top.iterrows())
    return f"""<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><title>Pattern Scan</title>
<style>body{{font-family:system-ui,Tahoma,sans-serif;margin:24px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:12px;direction:ltr}}td,th{{border:1px solid #ddd;padding:5px}}
th{{background:#eee}}tr.ok td{{background:#eaf6ec}}.w{{background:#fff4e5;border:1px solid #f0c27b;padding:12px;border-radius:6px}}</style>
<h1>مسح الأنماط — {s['run_id']}</h1>
<p>فرضيات هذه الدورة: <b>{s['tested_this_run']}</b> | إجمالي ما اختُبر تاريخيًا: <b>{s['lifetime_tests']}</b> |
نهاية نافذة البحث: {s['research_window_end']} (ما بعدها خزنة مقفلة)</p>
<div class="w">بدون تصحيح كان سيظهر نحو <b>{s['expected_false_positives_at_5pct_without_correction']}</b> نمطًا "ناجحًا" بالصدفة.
بعد التصحيح (BH): {s['bh_significant']}. المرشّحة بعد كل الشروط والتكاليف: <b>{len(s['candidates'])}</b>.
المرشّح ليس اكتشافًا مؤكدًا — يبقى معلّقًا حتى اختبار الخزنة بموافقة خالد.</div>
<h2>أعلى 60 نتيجة</h2><table><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr>{trs}</table></html>"""


if __name__ == "__main__":
    sys.exit(main())
