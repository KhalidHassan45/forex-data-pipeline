"""Append-only audit trail (JSONL, monthly files) + Telegram alerts that can never break trading."""
import json
from datetime import datetime, timezone

import requests


class Audit:
    def __init__(self, cfg):
        self.dir = cfg.state_dir / cfg.mode / "audit"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

    def log(self, event, **kw):
        now = datetime.now(timezone.utc)
        rec = {"ts": now.isoformat(timespec="seconds"), "mode": self.cfg.mode, "event": event, **kw}
        with (self.dir / f"{now:%Y%m}.jsonl").open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    def alert(self, text):
        self.log("ALERT", text=text)
        if not (self.cfg.telegram_token and self.cfg.telegram_chat):
            return
        try:
            requests.post(f"https://api.telegram.org/bot{self.cfg.telegram_token}/sendMessage",
                          data={"chat_id": self.cfg.telegram_chat, "text": f"[{self.cfg.mode.upper()}] {text}"}, timeout=10)
        except Exception:
            pass
