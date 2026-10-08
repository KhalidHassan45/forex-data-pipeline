"""Simulated broker: replays historical H1 candles, fills at bid/ask, checks stops conservatively
(a bar touching both stop and target counts as the stop — same rule as ml-lab labels)."""
import itertools

import pandas as pd

from .base import Account, Broker, BrokerTrade, OrderResult, Price


class SimBroker(Broker):
    name = "sim"

    def __init__(self, data: dict, spreads_px: dict, start_cash=100_000.0, currency="USD"):
        self.data = data                       # pair → DataFrame (UTC index, ohlc)
        self.spr = spreads_px                  # pair → spread in price units
        self.cash = start_cash
        self.ccy = currency
        self.now = None                        # current time = open of the bar being "lived"
        self.trades: dict[str, BrokerTrade] = {}
        self._ids = itertools.count(1)
        self.reject_next = None                # tests: force a rejection reason

    # time control
    def set_time(self, ts):
        ts = pd.Timestamp(ts)
        if self.now is not None:
            self._process(self.now, ts)
        self.now = ts

    def _bars(self, pair, t0, t1):
        d = self.data[pair]
        return d[(d.index >= t0) & (d.index < t1)]

    def _process(self, t0, t1):
        for t in list(self.trades.values()):
            if t.state != "OPEN":
                continue
            h = self.spr[t.pair] / 2                     # candles are mid: longs exit on bid, shorts on ask
            for ts, b in self._bars(t.pair, max(t0, t.open_ts), t1).iterrows():
                long = t.units > 0
                hit_sl = (b.low - h <= t.sl) if long else (b.high + h >= t.sl)
                hit_tp = (b.high - h >= t.tp) if long else (b.low + h <= t.tp)
                if hit_sl:
                    px = min(t.sl, b.open - h) if long else max(t.sl, b.open + h)
                    self._close(t, px); break
                if hit_tp:
                    self._close(t, t.tp); break

    def _close(self, t, px):
        t.state, t.close_px = "CLOSED", px
        t.realized_pl = (px - t.entry) * t.units * self._q2a(t.pair, px)
        self.cash += t.realized_pl

    def _q2a(self, pair, px):
        q = pair[3:]
        if q == self.ccy:
            return 1.0
        if f"{self.ccy}{q}" in self.data:
            return 1.0 / self._mid(f"{self.ccy}{q}")
        if f"{q}{self.ccy}" in self.data:
            return self._mid(f"{q}{self.ccy}")
        return 1.0 / px if pair.startswith(self.ccy) else 1.0

    def _mid(self, pair):
        d = self.data[pair]
        row = d[d.index < self.now].iloc[-1]
        return float(row.close)

    # interface
    def account(self):
        unreal = 0.0
        for t in self.trades.values():
            if t.state == "OPEN":
                p = self.prices([t.pair])[t.pair]
                px = p.bid if t.units > 0 else p.ask
                unreal += (px - t.entry) * t.units * self._q2a(t.pair, px)
        return Account(nav=self.cash + unreal, balance=self.cash, currency=self.ccy)

    def prices(self, pairs):
        out = {}
        for p in pairs:
            if p not in self.data:
                continue
            d = self.data[p]
            nxt = d[d.index >= self.now]
            px = float(nxt.iloc[0].open) if len(nxt) else float(d.iloc[-1].close)
            out[p] = Price(p, px - self.spr[p] / 2, px + self.spr[p] / 2, self.now, True)
        return out

    def candles(self, pair, count):
        d = self.data[pair]
        return d[d.index + pd.Timedelta(hours=1) <= self.now].tail(count)

    def market_order(self, pair, units, sl, tp, client_id):
        if self.reject_next:
            r, self.reject_next = self.reject_next, None
            return OrderResult(False, reason=r)
        p = self.prices([pair])[pair]
        px = p.ask if units > 0 else p.bid
        if (units > 0 and not (sl < px < tp)) or (units < 0 and not (tp < px < sl)):
            return OrderResult(False, reason="STOP_LOSS_ON_FILL_PRICE_INVALID")
        tid = str(next(self._ids))
        self.trades[tid] = BrokerTrade(tid, pair, float(units), px, self.now, client_id, sl, tp)
        return OrderResult(True, tid, px, float(units))

    def open_trades(self):
        return [t for t in self.trades.values() if t.state == "OPEN"]

    def trade(self, trade_id):
        if trade_id.startswith("@"):                 # OANDA-style lookup by client id
            return next((t for t in self.trades.values() if t.client_id == trade_id[1:]), None)
        return self.trades.get(trade_id)

    def close_trade(self, trade_id):
        t = self.trades.get(trade_id)
        if not t or t.state != "OPEN":
            return OrderResult(False, trade_id, reason="TRADE_DOESNT_EXIST")
        p = self.prices([t.pair])[t.pair]
        px = p.bid if t.units > 0 else p.ask
        self._close(t, px)
        return OrderResult(True, trade_id, px)
