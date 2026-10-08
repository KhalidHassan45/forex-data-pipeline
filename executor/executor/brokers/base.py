"""Broker interface. Every adapter speaks our pair names (EURUSD) and UTC timestamps."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Price:
    pair: str
    bid: float
    ask: float
    ts: pd.Timestamp
    tradeable: bool = True

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def spread(self):
        return self.ask - self.bid


@dataclass
class Account:
    nav: float
    balance: float
    currency: str
    margin_used: float = 0.0


@dataclass
class BrokerTrade:
    trade_id: str
    pair: str
    units: float          # signed: + long, − short
    entry: float
    open_ts: pd.Timestamp
    client_id: str = ""
    sl: float | None = None
    tp: float | None = None
    state: str = "OPEN"   # OPEN | CLOSED
    realized_pl: float = 0.0
    close_px: float | None = None


@dataclass
class OrderResult:
    ok: bool
    trade_id: str = ""
    fill_px: float = 0.0
    units: float = 0.0
    reason: str = ""
    raw: dict | None = None


class Broker(ABC):
    name = "base"

    @abstractmethod
    def account(self) -> Account: ...

    @abstractmethod
    def prices(self, pairs) -> dict: ...

    @abstractmethod
    def candles(self, pair, count) -> pd.DataFrame:
        """COMPLETE H1 candles only, index = bar open time (UTC), columns open/high/low/close (mid)."""

    @abstractmethod
    def market_order(self, pair, units, sl, tp, client_id) -> OrderResult: ...

    @abstractmethod
    def open_trades(self) -> list: ...

    @abstractmethod
    def trade(self, trade_id) -> BrokerTrade | None: ...

    @abstractmethod
    def close_trade(self, trade_id) -> OrderResult: ...

    def precision(self, pair) -> int:
        return 3 if pair.endswith("JPY") or pair.startswith("XAU") else 5
