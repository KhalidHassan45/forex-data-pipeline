"""Shared: data loading, the locked vault, the hypothesis registry, statistics."""
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent / "backtest", Path("/app/backtest")):
    if cand.exists():
        sys.path.insert(0, str(cand))
from engine import SLIPPAGE_PIPS, SPREAD_PIPS, DEFAULT_SPREAD, metrics, pip_size  # noqa: E402

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RES_DIR = DATA_DIR / "research"
VAULT_FILE = RES_DIR / "vault.json"
REGISTRY = RES_DIR / "registry.jsonl"
VAULT_YEARS = 2


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_all(tf="h1", pairs=None):
    pq = DATA_DIR / "parquet"
    files = sorted(pq.glob(f"*_{tf}.parquet"))
    D = {}
    for f in files:
        p = f.name.split("_")[0].upper()
        if pairs and p not in pairs:
            continue
        df = pd.read_parquet(f).set_index("ts").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        D[p] = df[["open", "high", "low", "close"]].astype(float)
    return D


# ---------------------------------------------------------------- vault
def vault_start(D) -> pd.Timestamp:
    """Fixed on first use and NEVER moved: data from this date on is out of bounds for research."""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    if VAULT_FILE.exists():
        return pd.Timestamp(json.loads(VAULT_FILE.read_text())["vault_start"])
    last = max(df.index[-1] for df in D.values())
    vs = pd.Timestamp(year=last.year - VAULT_YEARS, month=last.month, day=1, tz="UTC")
    VAULT_FILE.write_text(json.dumps({"vault_start": vs.isoformat(), "fixed_at": now(),
                                      "note": "do not edit — moving this date invalidates every result"}, indent=2))
    return vs


# ---------------------------------------------------------------- registry
def registry():
    if not REGISTRY.exists():
        return []
    return [json.loads(l) for l in REGISTRY.read_text().splitlines() if l.strip()]


def append(rows):
    RES_DIR.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------- statistics
def p_two_sided(t):
    return math.erfc(abs(t) / math.sqrt(2))


def benjamini_hochberg(pvals, q=0.05):
    p = np.asarray(pvals)
    n = len(p)
    if n == 0:
        return np.array([], bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out


def evaluate(s: pd.Series, df: pd.DataFrame, pair: str, direction=None):
    """
    s: undirected exposure held per bar. Returns stats dict and the directed net returns.
    t-stat is computed on DAILY sums of exposed returns (reduces intraday autocorrelation).
    """
    ret = df["close"].pct_change().fillna(0.0)
    s = s.reindex(df.index).fillna(0.0)
    g = s * ret
    exposed = g[s != 0]
    if exposed.empty:
        return None, None
    daily = exposed.groupby(exposed.index.normalize()).sum()
    n = len(daily)
    sd = daily.std()
    t = float(daily.mean() / sd * math.sqrt(n)) if sd > 0 and n > 1 else 0.0
    d = direction if direction is not None else (1.0 if daily.mean() >= 0 else -1.0)
    held = d * s
    half = (SPREAD_PIPS.get(pair, DEFAULT_SPREAD) / 2 + SLIPPAGE_PIPS) * pip_size(pair)
    turnover = held.diff().abs().fillna(held.abs())
    net = held * ret - turnover * half / df["close"].shift(1).fillna(df["close"])
    m = metrics(net, held)
    yearly = (d * daily).groupby(daily.index.year).mean()
    stats = {
        "t": round(d * t, 2),          # signed in the chosen direction (positive = edge)
        "p": p_two_sided(t),
        "direction": int(d),
        "days": n,
        "stability": round(float((yearly > 0).mean()), 2) if len(yearly) else 0.0,
        "years": int(len(yearly)),
        "gross_bps_per_day": round(float(d * daily.mean()) * 1e4, 2),
        **{f"net_{k}": v for k, v in m.items()},
    }
    return stats, net
