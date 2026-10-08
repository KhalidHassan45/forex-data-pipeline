"""
Hypothesis families. Each generator yields H objects whose fn(D) returns an UNDIRECTED
exposure series s in {-1, 0, 1} on D[pair].index, meaning "exposure HELD during bar t".

Causality contract: any price information used to set s[t] must come from bars <= t-1
(hence the .shift(1) everywhere). Calendar information (hour, weekday, month end) is
known in advance and may be used for bar t directly.

The scanner learns the DIRECTION (+1 follow / -1 fade) from the research window only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class H:
    id: str
    family: str
    pair: str
    params: dict
    desc: str
    fn: object = field(repr=False)


def _ret(df):
    return df["close"].pct_change()


# ---------------------------------------------------------------- calendar
def hour_of_day(D, pairs):
    for p in pairs:
        for h in range(24):
            yield H(f"hour|{p}|{h}", "hour_of_day", p, {"hour_utc": h},
                    f"{p}: اتجاه متكرر خلال الساعة {h:02d}:00 UTC",
                    lambda D, p=p, h=h: pd.Series((D[p].index.hour == h).astype(float), D[p].index))


def day_of_week(D, pairs):
    names = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"]
    for p in pairs:
        for d in range(5):
            yield H(f"dow|{p}|{d}", "day_of_week", p, {"weekday": d},
                    f"{p}: اتجاه متكرر يوم {names[d]}",
                    lambda D, p=p, d=d: pd.Series((D[p].index.dayofweek == d).astype(float), D[p].index))


def month_end(D, pairs):
    for p in pairs:
        def fn(D, p=p):
            idx = D[p].index
            days = pd.Series(idx.normalize(), index=idx)
            uniq = pd.Series(pd.unique(days))
            last2 = set(uniq.groupby(uniq.dt.tz_localize(None).dt.to_period("M")).tail(2))
            return pd.Series(days.isin(last2).astype(float).to_numpy(), idx)
        yield H(f"monthend|{p}", "month_end", p, {"days": 2},
                f"{p}: حركة آخر يومي تداول في الشهر (تدفقات إعادة التوازن)", fn)


# ---------------------------------------------------------------- price behaviour
def big_bar(D, pairs):
    for p in pairs:
        for k in (2.0, 3.0):
            for n in (1, 4, 24):
                def fn(D, p=p, k=k, n=n):
                    r = _ret(D[p])
                    z = r / r.rolling(500, min_periods=200).std()
                    trig = np.sign(r).where(z.abs() > k, 0.0).fillna(0.0)
                    return trig.shift(1).rolling(n, min_periods=1).sum().clip(-1, 1).fillna(0.0)
                yield H(f"bigbar|{p}|{k}|{n}", "big_bar", p, {"z": k, "hold_bars": n},
                        f"{p}: بعد شمعة كبيرة (>{k}σ) — استمرار أم ارتداد خلال {n} شمعة", fn)


def vol_regime(D, pairs):
    for p in pairs:
        for regime in ("high", "low"):
            def fn(D, p=p, regime=regime):
                c = D[p]["close"]
                r = c.pct_change()
                mom = np.sign(c / c.shift(20) - 1)
                vol = r.rolling(24).std()
                rank = vol.rolling(24 * 60, min_periods=24 * 20).rank(pct=True)
                cond = rank > 0.8 if regime == "high" else rank < 0.2
                return (mom.where(cond, 0.0)).shift(1).fillna(0.0)
            yield H(f"volreg|{p}|{regime}", "vol_regime", p, {"regime": regime, "lookback": 20},
                    f"{p}: الزخم (20 شمعة) في فترات التذبذب {'المرتفع' if regime == 'high' else 'المنخفض'}", fn)


def session_follow(D, pairs):
    for p in pairs:
        for name, o, e in (("london", 7, 16), ("newyork", 13, 20)):
            def fn(D, p=p, o=o, e=e):
                idx = D[p].index
                r = _ret(D[p])
                first = r.where(idx.hour == o)
                f = first.groupby(idx.normalize()).ffill()
                mask = (idx.hour > o) & (idx.hour < e)
                return np.sign(f).where(mask, 0.0).fillna(0.0)
            yield H(f"session|{p}|{name}", "session_follow", p, {"session": name, "open_utc": o, "end_utc": e},
                    f"{p}: هل تحدد أول ساعة من جلسة {name} اتجاه بقية الجلسة؟", fn)


def lead_lag(D, pairs):
    for a in pairs:
        for b in pairs:
            if a == b:
                continue
            def fn(D, a=a, b=b):
                ra = _ret(D[a]).reindex(D[b].index)
                return np.sign(ra).shift(1).fillna(0.0)
            yield H(f"leadlag|{a}|{b}", "lead_lag", b, {"leader": a},
                    f"{b}: هل تتبع حركة {a} في الساعة السابقة؟", fn)


FAMILIES = {
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "month_end": month_end,
    "big_bar": big_bar,
    "vol_regime": vol_regime,
    "session_follow": session_follow,
    "lead_lag": lead_lag,
}

BUILTIN = set(FAMILIES)

# Hermes-authored families live in custom_hypotheses.py (see references/custom-hypotheses.md)
# it is kept on the persistent volume ($DATA_DIR/research/) so it survives redeploys.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.getenv("DATA_DIR", "data"), "research"))
try:
    from custom_hypotheses import CUSTOM_FAMILIES  # type: ignore
    FAMILIES.update(CUSTOM_FAMILIES)
except ImportError:
    pass


def generate(D, pairs, families=None):
    for name, gen in FAMILIES.items():
        if families and name not in families:
            continue
        yield from gen(D, pairs)


def by_id(D, pairs, hid):
    for h in generate(D, pairs, [f for f in FAMILIES]):
        if h.id == hid:
            return h
    raise KeyError(hid)
