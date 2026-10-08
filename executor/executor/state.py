"""Durable state in SQLite (one DB per mode/account). Every order gets a client id BEFORE it is sent, so a crash
between "send" and "record" can always be resolved against the broker."""
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades(
  client_id TEXT PRIMARY KEY, model_id TEXT, pair TEXT, side INTEGER, units REAL, entry REAL, sl REAL, tp REAL,
  signal_ts TEXT, horizon INTEGER, created_ts TEXT, open_ts TEXT, trade_id TEXT, status TEXT,
  close_ts TEXT, close_px REAL, realized_pl REAL, risk_amt REAL, size_factor REAL, reason TEXT);
CREATE INDEX IF NOT EXISTS trades_status ON trades(status);
CREATE TABLE IF NOT EXISTS processed(model_id TEXT, bar_ts TEXT, PRIMARY KEY(model_id, bar_ts));
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS approvals(model_id TEXT, stage TEXT, approved_by TEXT, ts TEXT, evidence TEXT,
  PRIMARY KEY(model_id, stage));
"""


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    # kv
    def get(self, key, default=None):
        r = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def set(self, key, value):
        self.db.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, json.dumps(value, default=str)))

    # bars
    def processed(self, model_id, bar_ts):
        return self.db.execute("SELECT 1 FROM processed WHERE model_id=? AND bar_ts=?", (model_id, str(bar_ts))).fetchone() is not None

    def mark(self, model_id, bar_ts):
        self.db.execute("INSERT OR IGNORE INTO processed VALUES(?,?)", (model_id, str(bar_ts)))

    # trades
    def insert(self, **t):
        cols = ",".join(t)
        self.db.execute(f"INSERT INTO trades({cols}) VALUES({','.join('?' * len(t))})", [str(v) if k.endswith("_ts") and v is not None else v for k, v in t.items()])

    def update(self, client_id, **f):
        sets = ",".join(f"{k}=?" for k in f)
        self.db.execute(f"UPDATE trades SET {sets} WHERE client_id=?", [*(str(v) if k.endswith('_ts') and v is not None else v for k, v in f.items()), client_id])

    def trades(self, *status, model_id=None):
        q, a = "SELECT * FROM trades", []
        w = []
        if status:
            w.append(f"status IN ({','.join('?' * len(status))})"); a += status
        if model_id:
            w.append("model_id=?"); a.append(model_id)
        if w:
            q += " WHERE " + " AND ".join(w)
        return [dict(r) for r in self.db.execute(q + " ORDER BY created_ts", a)]

    def orders_since(self, ts):
        return self.db.execute("SELECT COUNT(*) c FROM trades WHERE created_ts>=? AND status!='SKIPPED'", (str(ts),)).fetchone()["c"]

    # approvals
    def approve(self, model_id, stage, by, ts, evidence):
        self.db.execute("INSERT OR REPLACE INTO approvals VALUES(?,?,?,?,?)", (model_id, stage, by, str(ts), json.dumps(evidence, default=str)))

    def approval(self, model_id, stage):
        r = self.db.execute("SELECT * FROM approvals WHERE model_id=? AND stage=?", (model_id, stage)).fetchone()
        return dict(r) if r else None
