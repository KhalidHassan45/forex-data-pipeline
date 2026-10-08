"""
lab-api — receives strategy files from the dashboard, analyses them, and queues them for the
sandboxed runner. It NEVER executes uploaded or generated code.

Run: uvicorn api:app --host 0.0.0.0 --port 8000      (behind nginx /api/, Basic Auth there)
Env: SANDBOX_DIR (/sandbox), ANTHROPIC_API_KEY, LAB_LLM_MODEL, LAB_DAILY_LIMIT (20)
"""
import json
import os
import re
import threading
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import store
from extract import MAX_BYTES, ExtractError, extract, kind_of
from spec import SpecError, describe_ar, validate

HERE = Path(__file__).resolve().parent
DAILY_LIMIT = int(os.getenv("LAB_DAILY_LIMIT", "20"))
SOURCES = ["quantpedia", "ssrn", "arxiv", "quantocracy", "quantconnect", "tradingview", "mql5", "github",
           "forexfactory", "babypips", "quantifiedstrategies", "other"]
_llm_slots = threading.BoundedSemaphore(2)

app = FastAPI(title="Strategy Lab intake", docs_url=None, redoc_url=None, openapi_url=None)


def static_flags(text):
    """Reuse the importer's red-flag patterns on the original text (no execution)."""
    import sys
    for cand in (HERE.parent / "importer", Path("/app/importer")):
        if cand.exists():
            sys.path.insert(0, str(cand))
    os.environ.setdefault("DATA_DIR", str(store.SANDBOX))
    try:
        from import_tool import static_flags as sf
        return [{"flag": k, "severity": s, "why": w} for k, s, w in dict.fromkeys(sf(text))]
    except SystemExit:
        return []


def public(st):
    """Fields the dashboard may see (no file paths, no raw text)."""
    keep = ("id", "filename", "kind", "lang", "route", "name", "source", "url", "claim", "license", "created",
            "state", "timeline", "summary_ar", "ambiguities", "red_flags", "static_flags", "describe", "error",
            "confidence", "reason_unsupported", "import_name", "verdict", "pairs_passed", "pairs_tested",
            "results", "best_wf_sharpe", "report", "model", "pages", "truncated")
    return {k: st.get(k) for k in keep if k in st}


def analyse(sid):
    st = store.load(sid)
    try:
        data = (store.sdir(sid) / st["stored_as"]).read_bytes()
        ex = extract(st["filename"], data)
        st.update(kind=ex["kind"], lang=ex["lang"], route=ex["route"], pages=ex.get("pages"), truncated=ex.get("truncated"))
        if ex.get("text"):
            st["static_flags"] = static_flags(ex["text"])
            (store.sdir(sid) / "extracted.txt").write_text(ex["text"])
        store.advance(st, "analyzing")
        pre = [f for f in (st.get("static_flags") or []) if f.get("severity") == "BLOCK"]
        if pre:  # stop before spending an LLM call on a martingale / grid / look-ahead file
            store.advance(st, "blocked", error="علامات حمراء محظورة في الملف الأصلي: " + "، ".join(f["why"] for f in pre))
            return
        if ex["route"] == "template":
            spec = ex["spec"]
            meta = ex.get("meta", {})
            for k in ("url", "claim", "license"):
                st[k] = st.get(k) or meta.get(k)
            if meta.get("source") in SOURCES and st.get("source") == "other":
                st["source"] = meta["source"]
            st.update(summary_ar=spec.get("description") or "استراتيجية من قالب Excel", confidence="high", ambiguities=[], red_flags=[])
        elif ex["route"] == "python_port":
            st.update(summary_ar="ملف Python بصيغة المختبر (STRATEGY). سيُشغَّل في الحاوية المعزولة فقط.",
                      confidence="high", ambiguities=["الكود يُنفَّذ كما هو داخل الصندوق المعزول؛ راجع منطقه قبل الاعتماد على نتيجته."], red_flags=[])
            (store.sdir(sid) / "port.py").write_text(ex["text"])
            store.advance(st, "translated", name=st["name"])
            store.enqueue(st)
            return
        else:
            from translate import TranslateError, translate
            with _llm_slots:
                try:
                    out = translate(ex["text"], ex["lang"], st["filename"])
                except TranslateError as e:
                    store.advance(st, "failed", error=str(e))
                    return
            st.update(summary_ar=out.get("summary_ar"), ambiguities=out.get("ambiguities", []), red_flags=out.get("red_flags", []),
                      confidence=out.get("confidence"), model=out.get("model"), claim=st.get("claim") or out.get("claimed_performance"))
            if not out.get("supported", True):
                store.advance(st, "unsupported", reason_unsupported=out.get("reason_unsupported"))
                return
            spec = out["spec"]
        blocking = [f for f in (st.get("red_flags") or []) if f.get("flag") in ("martingale", "grid", "averaging")]
        blocking += [f for f in (st.get("static_flags") or []) if f.get("severity") == "BLOCK"]
        if blocking:
            store.advance(st, "blocked", error="علامات حمراء محظورة: " + "، ".join(f.get("why") or f.get("flag") for f in blocking))
            return
        validate(spec)
        st["name"] = spec["name"]
        st["describe"] = describe_ar(spec)
        (store.sdir(sid) / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1))
        store.advance(st, "translated")
        store.enqueue(st)
    except (ExtractError, SpecError) as e:
        store.advance(st, "rejected", error=str(e))
    except Exception as e:  # never leave a submission hanging
        store.advance(st, "failed", error=f"خطأ غير متوقع أثناء التحليل: {type(e).__name__}: {e}")


@app.get("/api/health")
def health():
    return {"ok": True, "llm": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.get("/api/sources")
def sources():
    return {"sources": SOURCES}


@app.get("/api/template.xlsx")
def template():
    path = store.SANDBOX / "strategy_template.xlsx"
    if not path.exists():
        from make_template import build
        build(path)
    return FileResponse(path, filename="strategy_template.xlsx",
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/api/submissions")
def list_submissions():
    return {"items": [public(s) for s in store.all_status()], "llm": bool(os.getenv("ANTHROPIC_API_KEY")),
            "daily_limit": DAILY_LIMIT, "used_today": store.count_today()}


@app.get("/api/submissions/{sid}")
def get_submission(sid: str):
    try:
        st = store.load(sid)
    except ValueError:
        st = None
    if not st:
        raise HTTPException(404, "غير موجود")
    return public(st)


@app.post("/api/submissions")
async def create_submission(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    source: str = Form("other"),
    url: str = Form(""),
    claim: str = Form(""),
    license: str = Form(""),
    x_lab_request: str = Header(None),
):
    # custom header forces a CORS preflight → blocks cross-site form posts riding on Basic-Auth credentials
    if x_lab_request != "1":
        raise HTTPException(403, "طلب مرفوض")
    if store.count_today() >= DAILY_LIMIT:
        raise HTTPException(429, f"تم بلوغ الحد اليومي ({DAILY_LIMIT} ملفًا). حاول غدًا.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "حجم الملف أكبر من 10MB")
    fname = os.path.basename(file.filename or "upload")[:120]
    ext = Path(fname).suffix.lower()
    try:
        kind_of(fname)
    except ExtractError as e:
        raise HTTPException(415, str(e))
    if source not in SOURCES:
        source = "other"
    if url and not re.match(r"^https?://", url):
        url = ""
    sid = store.new_id()
    st = {"id": sid, "filename": fname, "stored_as": "original" + ext, "source": source, "url": url[:500],
          "claim": claim[:300], "license": license[:80], "created": store.now(), "name": None}
    d = store.sdir(sid)
    d.mkdir(parents=True, exist_ok=True)
    (d / st["stored_as"]).write_bytes(data)
    store.advance(st, "received")
    background.add_task(analyse, sid)
    return JSONResponse(public(st), status_code=202)
