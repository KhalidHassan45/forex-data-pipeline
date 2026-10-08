"""Submission storage shared by lab-api (writer of new submissions) and lab-runner (tester)."""
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

SANDBOX = Path(os.getenv("SANDBOX_DIR", "/sandbox"))
INBOX = SANDBOX / "inbox"
QUEUE = SANDBOX / "queue"

STAGES = ["received", "analyzing", "translated", "queued", "checking", "testing", "done"]
TERMINAL = {"done", "failed", "blocked", "unsupported", "rejected"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id():
    return time.strftime("%Y%m%d%H%M%S", time.gmtime()) + secrets.token_hex(3)


def sdir(sid):
    if not sid.isalnum() or len(sid) > 32:
        raise ValueError("bad id")
    return INBOX / sid


def load(sid):
    try:
        return json.loads((sdir(sid) / "status.json").read_text())
    except FileNotFoundError:
        return None


def save(st):
    d = sdir(st["id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "status.json.tmp"
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str))
    tmp.replace(d / "status.json")


def advance(st, state, **fields):
    st["state"] = state
    st.setdefault("timeline", []).append({"state": state, "ts": now()})
    st.update(fields)
    save(st)
    return st


def enqueue(st):
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / f"{st['id']}.job").write_text(st["id"])
    return advance(st, "queued")


def all_status(limit=60):
    if not INBOX.exists():
        return []
    out = []
    for d in sorted(INBOX.iterdir(), reverse=True)[:limit]:
        st = load(d.name)
        if st:
            out.append(st)
    return out


def count_today():
    day = time.strftime("%Y%m%d", time.gmtime())
    return sum(1 for d in INBOX.glob(f"{day}*")) if INBOX.exists() else 0
