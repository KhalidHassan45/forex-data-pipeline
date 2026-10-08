"""Position sizing and portfolio limits. Pure functions — easy to test, no broker calls."""
import math
from dataclasses import dataclass


def conv(ccy, acct, prices):
    """Rate to convert 1 unit of `ccy` into the account currency, from live mids. None if unknown."""
    if ccy == acct:
        return 1.0
    if f"{ccy}{acct}" in prices:
        return prices[f"{ccy}{acct}"].mid
    if f"{acct}{ccy}" in prices:
        return 1.0 / prices[f"{acct}{ccy}"].mid
    # cross through USD
    if acct != "USD":
        a, b = conv(ccy, "USD", prices), conv("USD", acct, prices)
        return a * b if a and b else None
    return None


@dataclass
class Position:
    pair: str
    units: float      # signed
    entry: float
    sl: float


def notional_by_ccy(positions, prices, acct):
    """Net exposure per currency in account currency: long EURUSD = +EUR, −USD."""
    ex = {}
    for p in positions:
        base, quote = p.pair[:3], p.pair[3:]
        px = prices[p.pair].mid if p.pair in prices else p.entry
        rb, rq = conv(base, acct, prices), conv(quote, acct, prices)
        if rb is None or rq is None:
            raise ValueError(f"no conversion rate for {p.pair} into {acct}")
        ex[base] = ex.get(base, 0.0) + p.units * rb
        ex[quote] = ex.get(quote, 0.0) - p.units * px * rq
    return ex


def gross(positions, prices, acct):
    return sum(abs(p.units) * conv(p.pair[:3], acct, prices) for p in positions)


def stop_risk(p: Position, prices, acct):
    return abs(p.entry - p.sl) * abs(p.units) * conv(p.pair[3:], acct, prices)


def size_units(equity, risk_pct, size_factor, entry, sl, pair, prices, acct, units_precision=0, min_units=1):
    """Units so that hitting the stop loses risk_pct × size_factor of equity."""
    q2a = conv(pair[3:], acct, prices)
    dist = abs(entry - sl)
    if not q2a or dist <= 0:
        return 0.0
    u = equity * risk_pct / 100 * size_factor / (dist * q2a)
    f = 10 ** units_precision
    u = math.floor(u * f) / f
    return u if u >= min_units else 0.0


def check(new: Position, open_pos, equity, prices, acct, L, orders_last_hour):
    """Every reason this trade must NOT be sent. Empty list = allowed."""
    why = []
    allp = open_pos + [new]
    if len(open_pos) + 1 > L.max_positions:
        why.append(f"max_positions {L.max_positions}")
    if orders_last_hour + 1 > L.max_orders_per_hour:
        why.append(f"max_orders_per_hour {L.max_orders_per_hour}")
    opp = [p for p in open_pos if p.pair == new.pair and (p.units > 0) != (new.units > 0)]
    if opp:
        why.append("opposite position open on the same pair (netting risk)")
    risk = sum(stop_risk(p, prices, acct) for p in allp) / equity * 100
    if risk > L.max_open_risk_pct + 1e-9:
        why.append(f"open risk {risk:.2f}% > {L.max_open_risk_pct}%")
    lev = gross(allp, prices, acct) / equity
    if lev > L.max_gross_leverage_x:
        why.append(f"gross leverage {lev:.2f}x > {L.max_gross_leverage_x}x")
    for ccy, v in notional_by_ccy(allp, prices, acct).items():
        if ccy != acct and abs(v) / equity > L.max_currency_exposure_x:
            why.append(f"{ccy} exposure {abs(v) / equity:.2f}x > {L.max_currency_exposure_x}x")
    return why


SCALABLE = ("exposure", "gross leverage", "open risk")


def fit(new: Position, open_pos, equity, prices, acct, L, orders_last_hour, units_precision=0, min_units=1,
        steps=(1.0, 0.75, 0.5, 0.35, 0.25)):
    """Largest fraction of the target size that passes every limit. Limits that size cannot fix
    (position count, order rate, opposite position) are never scaled around. Returns (position|None, reasons, factor)."""
    why = check(new, open_pos, equity, prices, acct, L, orders_last_hour)
    if not why:
        return new, [], 1.0
    if any(not any(k in w for k in SCALABLE) for w in why):
        return None, why, 0.0
    f10 = 10 ** units_precision
    for k in steps[1:]:
        u = math.floor(abs(new.units) * k * f10) / f10
        if u < min_units:
            break
        cand = Position(new.pair, math.copysign(u, new.units), new.entry, new.sl)
        if not check(cand, open_pos, equity, prices, acct, L, orders_last_hour):
            return cand, why, k
    return None, why, 0.0
