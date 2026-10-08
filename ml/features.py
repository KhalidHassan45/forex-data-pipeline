"""Causal feature store. Every feature at bar t uses ONLY bars <= t (verified by leakage.py, not trusted).

Families (v1): returns & momentum, volatility regime, oscillators, trend distance, market structure
(range position, breakouts, swing distance), candle anatomy, time & sessions, month-end, cross-asset
(USD strength, gold), higher timeframes (rolling windows in h1 units — no resampling, so nothing peeks),
scheduled news (optional calendar.csv), and Hermes custom features (optional custom_features.py).

Stored as Parquet: /data/ml/features/<PAIR>_<tf>_<version>.parquet — the version is a hash of this file and
custom_features.py, so any code change produces a new version that must be re-certified for leakage.
"""
import hashlib
import importlib.util

import numpy as np
import pandas as pd

import common as C

CUSTOM_FILE = C.ML_DIR / "custom_features.py"
CALENDAR = C.ML_DIR / "calendar.csv"


def version() -> str:
    h = hashlib.sha1(open(__file__, "rb").read())
    if CUSTOM_FILE.exists():
        h.update(CUSTOM_FILE.read_bytes())
    return "f" + h.hexdigest()[:8]


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df, n):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _usd_sign(p):
    return 1.0 if p.startswith("USD") else -1.0 if p.endswith("USD") and not p.startswith("XAU") else 0.0


def core(df: pd.DataFrame, D: dict, pair: str) -> pd.DataFrame:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    lr = np.log(c).diff()
    a24 = _atr(df, 24)
    F = {}
    # returns & momentum (scaled by volatility so pairs/eras are comparable)
    for k in (1, 3, 6, 12, 24, 72, 120, 480):
        F[f"ret_{k}"] = (c / c.shift(k) - 1) / (a24 / c)
    # volatility regime
    v24, v120, v480 = (lr.rolling(n, min_periods=n).std() for n in (24, 120, 480))
    F["vol_24"], F["vol_120"] = v24 * 1e4, v120 * 1e4
    F["vol_ratio_24_120"], F["vol_ratio_120_480"] = v24 / v120, v120 / v480
    F["atr_pct"] = a24 / c * 1e4
    F["vol_rank_480"] = v24.rolling(480, min_periods=240).rank(pct=True)
    # oscillators
    F["rsi_2"], F["rsi_14"] = _rsi(c, 2), _rsi(c, 14)
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    F["bb_z_20"] = (c - m20) / s20
    # trend distance (ATR units) and slope — 120 h1 ≈ 1 week, 480 ≈ 1 month (the higher timeframes)
    for n in (20, 50, 120, 480):
        sma = c.rolling(n, min_periods=n).mean()
        F[f"dist_sma_{n}"] = (c - sma) / a24
        F[f"slope_sma_{n}"] = (sma - sma.shift(12)) / a24
    # market structure
    for n in (24, 120, 480):
        hh, ll = h.rolling(n, min_periods=n).max(), l.rolling(n, min_periods=n).min()
        F[f"range_pos_{n}"] = (c - ll) / (hh - ll).replace(0, np.nan)
        F[f"dist_high_{n}"] = (hh - c) / a24
        F[f"dist_low_{n}"] = (c - ll) / a24
        prev_hh, prev_ll = hh.shift(1), ll.shift(1)
        F[f"break_up_{n}"] = (c > prev_hh).astype(float)
        F[f"break_dn_{n}"] = (c < prev_ll).astype(float)
    F["bars_since_high_120"] = h.rolling(120, min_periods=120).apply(lambda x: 119 - np.argmax(x), raw=True)
    F["bars_since_low_120"] = l.rolling(120, min_periods=120).apply(lambda x: 119 - np.argmin(x), raw=True)
    # candle anatomy
    rng = (h - l).replace(0, np.nan)
    F["body"] = (c - o) / rng
    F["upper_wick"] = (h - np.maximum(o, c)) / rng
    F["lower_wick"] = (np.minimum(o, c) - l) / rng
    F["range_atr"] = (h - l) / a24
    # time & sessions (UTC)
    hr = df.index.hour
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    F["dow"] = df.index.dayofweek.astype(float)
    F["sess_asia"] = ((hr >= 0) & (hr < 7)).astype(float)
    F["sess_london"] = ((hr >= 7) & (hr < 12)).astype(float)
    F["sess_overlap"] = ((hr >= 12) & (hr < 16)).astype(float)
    F["sess_ny"] = ((hr >= 16) & (hr < 21)).astype(float)
    bme = df.index + pd.offsets.BusinessDay(2)
    F["month_end"] = (bme.month != df.index.month).astype(float)
    # cross-asset: USD strength from the other USD pairs, and gold
    legs = []
    for p, d2 in D.items():
        sgn = _usd_sign(p)
        if sgn and p != pair:
            cc = d2["close"].reindex(df.index)
            legs.append(sgn * np.log(cc / cc.shift(1)))
    if legs:
        usd = pd.concat(legs, axis=1).mean(axis=1, skipna=True)
        for k in (6, 24, 120):
            F[f"usd_str_{k}"] = usd.rolling(k, min_periods=k // 2).sum() * 1e4
    if "XAUUSD" in D and pair != "XAUUSD":
        g = D["XAUUSD"]["close"].reindex(df.index)
        F["xau_ret_24"] = (g / g.shift(24) - 1) * 1e4
    return pd.DataFrame(F, index=df.index)


def news(df: pd.DataFrame, pair: str):
    """Optional: /data/ml/calendar.csv with columns ts (UTC, SCHEDULED time), currency, impact.
    Scheduled times are known in advance, so 'hours to the next event' is not look-ahead."""
    if not CALENDAR.exists():
        return None
    cal = pd.read_csv(CALENDAR, parse_dates=["ts"])
    cal = cal[cal["impact"].str.lower().eq("high") & cal["currency"].str.upper().isin([pair[:3], pair[3:6]])]
    if cal.empty:
        return None
    ev = pd.DatetimeIndex(cal["ts"]).tz_localize("UTC") if cal["ts"].dt.tz is None else pd.DatetimeIndex(cal["ts"])
    ev = ev.sort_values().asi8
    t = df.index.asi8
    i = np.searchsorted(ev, t, side="left")
    nxt = np.where(i < len(ev), (ev[np.minimum(i, len(ev) - 1)] - t) / 3.6e12, np.nan)
    j = np.searchsorted(ev, t, side="right") - 1
    prv = np.where(j >= 0, (t - ev[np.maximum(j, 0)]) / 3.6e12, np.nan)
    return pd.DataFrame({"news_h_to_next": np.clip(nxt, 0, 72), "news_h_since_last": np.clip(prv, 0, 72)},
                        index=df.index)


def custom(df, D, pair):
    if not CUSTOM_FILE.exists():
        return None
    spec = importlib.util.spec_from_file_location("custom_features", CUSTOM_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.custom(df, D, pair)
    return out.add_prefix("cx_") if out is not None else None


def compute(df, D, pair) -> pd.DataFrame:
    parts = [core(df, D, pair), news(df, pair), custom(df, D, pair)]
    F = pd.concat([p for p in parts if p is not None], axis=1)
    return F.replace([np.inf, -np.inf], np.nan).astype("float32")


def path(pair, tf):
    return C.FEAT_DIR / f"{pair}_{tf}_{version()}.parquet"


def build(D, tf, force=False):
    C.FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for p, df in D.items():
        f = path(p, tf)
        if f.exists() and not force and pd.read_parquet(f, columns=["ret_1"]).index.max() >= df.index[-1]:
            out[p] = f; continue
        compute(df, D, p).to_parquet(f)
        out[p] = f
    return out


def load(pair, tf):
    return pd.read_parquet(path(pair, tf))
