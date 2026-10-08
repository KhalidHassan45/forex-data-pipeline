#!/usr/bin/env python3
"""Add the «التعلّم الآلي» tab to lab-dashboard. Idempotent: running it twice changes nothing.

    python patch_dashboard.py /app/dashboard        # or the dashboard/ folder in the repo
"""
import sys
from pathlib import Path

MARK = "ml-lab:v1"

BUILDER_FN = r'''
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

'''

VIEW_JS = r'''
/* ---------------------------------------------------------------- ml-lab:v1 */
const ML_STATUS = { CANDIDATE: ["warn", "مرشّح"], REJECTED: ["fail", "مرفوض"], PAPER_ACTIVE: ["pass", "Paper نشط"],
  PAPER_PAUSED: ["block", "Paper موقوف"], AWAITING_KHALID_APPROVAL_FOR_PAPER_TRADING: ["wait", "بانتظار موافقتك"] };
Object.assign(VERDICT, ML_STATUS);
const GATE_AR = { trades: "صفقات قليلة", netSR: "Sharpe بعد التكاليف منخفض", PF: "Profit factor منخفض",
  DSR: "لا يصمد أمام كثرة التجارب (DSR)", stability: "غير ثابت عبر السنوات", edge_over_null: "لا يتفوق على الضوضاء",
  stressSR: "ينهار بتكاليف ×1.5" };
function viewML() {
  const X = D.ml;
  if (!X) return [h("div", { class: "empty" }, "لم يُشغَّل ml-lab بعد. ابدأ بـ ml.sh build ثم ml.sh leakcheck.")];
  S.mlSel = S.mlSel || (X.rows[0] && X.rows[0].id);
  const vp = X.vault.filter(v => v.event === "VAULT_PASS").length;
  const active = X.paper.filter(p => p.status === "PAPER_ACTIVE").length;
  const tiles = h("div", { class: "tiles" },
    tile("نماذج مدرّبة (إجمالي)", N(int(X.lifetime)), "كل تدريب يرفع عتبة DSR للتالي"),
    tile("مرشّحة بانتظار الخزنة", N(String(X.pending.length)), X.pending.length ? "تحتاج موافقتك" : "لا شيء معلّق"),
    tile("اجتازت الخزنة", N(`${vp}/${X.vault.length}`), "من النماذج التي فُتحت لها الخزنة"),
    tile("Paper ظلّي نشط", N(String(active)), "بلا وسيط ولا حساب"),
    tile("شهادة التسرّب", X.cert ? pill(X.cert.status === "PASS" ? "PASS" : "WEAK") : "—",
      X.cert ? h("span", {}, "نسخة ", N(X.cert.version), X.cert.leaky_columns.length ? ` · ${X.cert.leaky_columns.length} خاصية مستبعدة` : "") : "شغّل ml.sh leakcheck"));
  const gates = card("لماذا تُرفض النماذج؟", "عدد مرات فشل كل بوابة. الرفض هو النتيجة المتوقعة لمعظم الأفكار.",
    table([{ label: "البوابة", get: g => GATE_AR[g[0]] || g[0] }, { label: "مرات الفشل", n: 1, get: g => int(g[1]) }], Object.entries(X.gates)));
  const pend = card("مرشّحة بانتظار موافقتك", "للموافقة اكتب لـ Hermes: «أوافق على اختبار الخزنة لـ [المعرّف]». الخزنة مشتركة مع البحث، وكل فتح يُحسب على الاثنين.",
    table([{ label: "المعرّف", get: r => h("span", { class: "mono", style: "font-size:12px" }, r.id) },
      { label: "Sharpe", n: 1, get: r => num(r.net_sharpe) }, { label: "DSR", n: 1, get: r => num(r.dsr, 3) },
      { label: "فوق الضوضاء", n: 1, get: r => num(r.edge_over_null) }, { label: "صفقات", n: 1, get: r => int(r.net_trades) }], X.pending));
  const sel = X.rows.find(r => r.id === S.mlSel);
  const list = card("كل التدريبات", "اضغط صفًا لعرض التفاصيل. النتائج خارج العينة (walk-forward) وبعد التكاليف.",
    table([{ label: "الزوج", get: r => N(r.pair) }, { label: "الحكم", get: r => pill(r.status) },
      { label: "صفقات", n: 1, get: r => int(r.net_trades) }, { label: "Sharpe", n: 1, get: r => num(r.net_sharpe) },
      { label: "PF", n: 1, get: r => num(r.net_profit_factor) }, { label: "DSR", n: 1, get: r => num(r.dsr, 3) },
      { label: "الضوضاء", n: 1, get: r => num(r.null_sharpe) }, { label: "الثبات", n: 1, get: r => num((r.stability ?? 0) * 100, 0) + "%" },
      { label: "التاريخ", get: r => N(fmtDate(r.ts)) }], X.rows, { onRow: r => { S.mlSel = r.id; render(); } }));
  const detail = sel ? card(sel.pair + " — " + (sel.status === "CANDIDATE" ? "مرشّح" : "مرفوض"), sel.why,
    h("div", { class: "mono", style: "font-size:11px;color:var(--muted);margin-bottom:8px" }, sel.id),
    sel.equity && sel.equity.length ? equityChart(sel.equity) : h("div", { class: "empty" }, "لا منحنى محفوظ لهذا التدريب."),
    h("div", { class: "grid g2" },
      table([{ label: "السنة", get: y => N(y[0]) }, { label: "العائد", n: 1, get: y => pct(y[1]) }], Object.entries(sel.by_year_pct || {})),
      table([{ label: "أهم الخصائص", get: f => h("span", { class: "mono", style: "font-size:12px" }, f[0]) }, { label: "الأهمية", n: 1, get: f => int(f[1]) }], Object.entries(sel.top_features || {}).slice(0, 10))),
    sel.fails ? h("p", { class: "sub" }, "أسباب الرفض: ", sel.fails.split(",").map(f => GATE_AR[f.split("=")[0]] || f).join(" · ")) : null) : null;
  const vault = card("سجل خزنة ML", null, table([{ label: "النموذج", get: r => h("span", { class: "mono", style: "font-size:12px" }, r.id) },
    { label: "النتيجة", get: r => pill(r.event) }, { label: "Sharpe", n: 1, get: r => num(r.net_sharpe) }, { label: "PF", n: 1, get: r => num(r.net_profit_factor) },
    { label: "أقصى تراجع", n: 1, get: r => pct(r.net_max_dd_pct) }, { label: "صفقات", n: 1, get: r => int(r.net_trades) }, { label: "الموافق", get: r => r.approved_by }], X.vault));
  const paper = card("Paper trading الظلّي", "يُقيَّم كل يوم على شموع لم يرها النموذج قط. لا وسيط ولا أوامر. التوقف تلقائي عند كسر شروط الخطة.",
    table([{ label: "النموذج", get: p => h("span", { class: "mono", style: "font-size:12px" }, p.id) }, { label: "الحالة", get: p => pill(p.status) },
      { label: "صفقات", n: 1, get: p => int(p.ledger?.net_trades) }, { label: "PF", n: 1, get: p => num(p.ledger?.net_profit_factor) },
      { label: "تراجع", n: 1, get: p => pct(p.ledger?.net_max_dd_pct) }, { label: "انحراف PSI", n: 1, get: p => num(p.drift?.mean_psi, 3) },
      { label: "تنبيه", get: p => p.alert ? h("b", {}, p.alert) : p.retrain_suggested ? h("span", { style: "white-space:nowrap", title: Object.keys(p.drift?.drifted || {}).join("، ") }, "انحراف ↻") : "—" }], X.paper));
  return [tiles, h("div", { class: "grid g2" }, pend, gates), list, detail, h("div", { class: "grid g2" }, vault, paper)];
}
'''


def patch_builder(p: Path):
    s = p.read_text(encoding="utf-8")
    if MARK in s:
        return "builder: already patched"
    s = s.replace("\ndef main():", BUILDER_FN + "\ndef main():", 1)
    s = s.replace('"submissions": submissions(),', '"submissions": submissions(), "ml": ml(),', 1)
    assert '"ml": ml()' in s, "could not find the data dict in build_dashboard.py"
    if "from pathlib import Path" not in s:
        s = "from pathlib import Path\n" + s
    p.write_text(s, encoding="utf-8")
    return "builder: patched"


def patch_page(p: Path):
    s = p.read_text(encoding="utf-8")
    if MARK in s:
        return "page: already patched"
    tab = '<button role="tab" id="t-research" aria-selected="false" data-tab="research">البحث والخزنة</button>'
    assert tab in s, "research tab not found in index.html"
    s = s.replace(tab, tab + '\n      <button role="tab" id="t-ml" aria-selected="false" data-tab="ml">التعلّم الآلي</button>', 1)
    s = s.replace("\nfunction render() {", VIEW_JS + "\nfunction render() {", 1)
    s = s.replace("imports: viewImports, data: viewData };", "imports: viewImports, data: viewData, ml: viewML };", 1)
    # promoted ML models show «نموذج» instead of follow/fade
    s = s.replace('p.direction === "follow" ? "اتباع" : "عكس"', 'p.kind === "ml" ? "نموذج ML" : p.direction === "follow" ? "اتباع" : "عكس"')
    assert "ml: viewML" in s, "views map not found in index.html"
    p.write_text(s, encoding="utf-8")
    return "page: patched"


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/dashboard")
    print(patch_builder(d / "build_dashboard.py"))
    print(patch_page(d / "index.html"))
