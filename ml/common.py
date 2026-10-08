"""ml-lab shared pieces: paths, data, the SHARED locked vault, costs, registry, metrics, DSR."""
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
ML_DIR = DATA_DIR / "ml"
FEAT_DIR, LABEL_DIR, RUNS_DIR, MODEL_DIR = (ML_DIR / d for d in ("features", "labels", "runs", "models"))
REGISTRY = ML_DIR / "registry.jsonl"
CERT_FILE = ML_DIR / "leakage_cert.json"
RES_DIR = DATA_DIR / "research"              # vault + research registry are SHARED with forex-research
VAULT_FILE = RES_DIR / "vault.json"
RES_REGISTRY = RES_DIR / "registry.jsonl"
VAULT_YEARS = 2
THREADS = int(os.getenv("ML_THREADS", "1"))  # the server serves clients: keep CPU modest

# ---------------------------------------------------------------- costs (engine.py if present)
for cand in (HERE.parent / "backtest", Path("/app/backtest")):
    if (cand / "engine.py").exists():
        sys.path.insert(0, str(cand))
try:
    from engine import SLIPPAGE_PIPS, SPREAD_PIPS, DEFAULT_SPREAD, pip_size  # noqa: E402
except Exception:  # standalone fallback (same spirit as forex-backtest)
    SPREAD_PIPS = {"EURUSD": 0.8, "GBPUSD": 1.2, "USDJPY": 0.9, "USDCHF": 1.3, "AUDUSD": 1.0, "USDCAD": 1.4,
                   "NZDUSD": 1.5, "EURGBP": 1.2, "EURJPY": 1.5, "GBPJPY": 2.2, "XAUUSD": 3.0}
    DEFAULT_SPREAD, SLIPPAGE_PIPS = 2.0, 0.3

    def pip_size(pair):
        pair = pair.upper()
        return 0.1 if pair.startswith("XAU") else 0.01 if pair.endswith("JPY") else 0.0001


def roundtrip_cost_px(pair):
    """Full round-trip cost in price units: spread + slippage on entry and exit."""
    return (SPREAD_PIPS.get(pair.upper(), DEFAULT_SPREAD) + 2 * SLIPPAGE_PIPS) * pip_size(pair)


def utc(x) -> pd.DatetimeIndex:
    """Any datetime-like column → tz-aware UTC DatetimeIndex (parquet round-trips can drop the tz)."""
    return pd.DatetimeIndex(pd.to_datetime(getattr(x, "values", x), utc=True))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cfg_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


# ---------------------------------------------------------------- data
def load_all(tf="h1", pairs=None):
    D = {}
    for f in sorted((DATA_DIR / "parquet").glob(f"*_{tf}.parquet")):
        p = f.name.split("_")[0].upper()
        if pairs and p not in pairs:
            continue
        df = pd.read_parquet(f)
        if "ts" in df.columns:
            df = df.set_index("ts")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        D[p] = df[["open", "high", "low", "close"]].astype(float)
    return D


# ---------------------------------------------------------------- the vault (same file as forex-research)
def vault_start(D) -> pd.Timestamp:
    """Fixed once, never moved. Identical rule to forex-research so both labs share ONE untouched window."""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    if VAULT_FILE.exists():
        return pd.Timestamp(json.loads(VAULT_FILE.read_text())["vault_start"])
    last = max(df.index[-1] for df in D.values())
    vs = pd.Timestamp(year=last.year - VAULT_YEARS, month=last.month, day=1, tz="UTC")
    VAULT_FILE.write_text(json.dumps({"vault_start": vs.isoformat(), "fixed_at": now(),
                                      "note": "do not edit — moving this date invalidates every result"}, indent=2))
    return vs


# ---------------------------------------------------------------- registries
def _jl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def registry():
    return _jl(REGISTRY)


def append(rows, path=REGISTRY):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def research_registry():
    return _jl(RES_REGISTRY)


def vault_unlocks_total():
    """Unlocks by BOTH labs — the vault loses value with every look, whoever looks."""
    return sum(1 for r in research_registry() if r.get("event") == "VAULT")


# ---------------------------------------------------------------- trade metrics
def trade_metrics(trades: pd.DataFrame, start=None, end=None):
    """trades: columns [entry_ts, exit_ts, side, ret] with ret = net fractional return at 1x notional."""
    out = {"net_trades": int(len(trades))}
    if trades.empty:
        return out | {"net_sharpe": 0.0, "net_profit_factor": 0.0, "net_max_dd_pct": 0.0, "net_win_rate": 0.0,
                      "net_avg_bps": 0.0, "net_cagr_pct": 0.0, "t": 0.0, "days": 0}
    r = trades["ret"].astype(float)
    start = start or trades["entry_ts"].min().normalize()
    end = end or trades["exit_ts"].max().normalize()
    days = pd.date_range(start.normalize(), end.normalize(), freq="B", tz="UTC")
    daily = r.groupby(pd.DatetimeIndex(trades["exit_ts"]).normalize()).sum().reindex(days, fill_value=0.0)
    sd = daily.std()
    sharpe = float(daily.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0
    eq = (1 + daily).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    yrs = max(len(days) / 252, 1e-9)
    act = daily[daily != 0]
    t = float(act.mean() / act.std() * math.sqrt(len(act))) if len(act) > 1 and act.std() > 0 else 0.0
    return out | {
        "net_sharpe": round(sharpe, 3),
        "net_profit_factor": round(float(gains / losses), 3) if losses > 0 else 99.0,
        "net_max_dd_pct": round(dd, 2),
        "net_win_rate": round(float((r > 0).mean()), 3),
        "net_avg_bps": round(float(r.mean() * 1e4), 2),
        "net_cagr_pct": round(float((eq.iloc[-1] ** (1 / yrs) - 1) * 100), 2),
        "t": round(t, 2), "days": int(len(days)),
        "_daily": daily,
    }


# ---------------------------------------------------------------- Deflated Sharpe Ratio
def deflated_sharpe(daily: pd.Series, n_trials: int, sr_var_trials: float | None):
    """Bailey & López de Prado (2014). Probability the true Sharpe > the best Sharpe expected by luck
    after n_trials attempts. Works on non-annualised daily Sharpe."""
    x = daily.astype(float).values
    T = len(x)
    if T < 30 or x.std() == 0:
        return 0.0
    sr = x.mean() / x.std()
    skew = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    kurt = float(((x - x.mean()) ** 4).mean() / x.std() ** 4)
    N = max(int(n_trials), 1)
    V = sr_var_trials if sr_var_trials and sr_var_trials > 0 else (0.5 / math.sqrt(252)) ** 2
    nd, g = NormalDist(), 0.5772156649
    sr0 = 0.0 if N == 1 else math.sqrt(V) * ((1 - g) * nd.inv_cdf(1 - 1 / N) + g * nd.inv_cdf(1 - 1 / (N * math.e)))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return round(nd.cdf((sr - sr0) * math.sqrt(T - 1) / denom), 4)


def clean(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}
