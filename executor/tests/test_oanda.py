"""OANDA adapter against recorded v20 response shapes (no network)."""
import json

import pytest

from executor.brokers.oanda import OandaBroker


class R:
    def __init__(self, code, j):
        self.status_code, self._j, self.text = code, j, json.dumps(j)

    def json(self):
        return self._j

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self):
        self.headers, self.sent = {}, []

    def get(self, url, params=None, timeout=None):
        if url.endswith("/summary"):
            return R(200, {"account": {"NAV": "100012.5", "balance": "100000", "currency": "USD", "marginUsed": "0"}})
        if url.endswith("/instruments"):
            return R(200, {"instruments": [{"name": "USD_JPY", "displayPrecision": 3, "tradeUnitsPrecision": 0, "minimumTradeSize": "1"}]})
        if url.endswith("/pricing"):
            return R(200, {"prices": [{"instrument": "USD_JPY", "time": "2026-03-03T10:00:01.123456789Z", "tradeable": True,
                                       "bids": [{"price": "150.010"}], "asks": [{"price": "150.022"}]}]})
        if "/candles" in url:
            return R(200, {"candles": [
                {"complete": True, "time": "2026-03-03T08:00:00.000000000Z", "mid": {"o": "150.0", "h": "150.2", "l": "149.9", "c": "150.1"}},
                {"complete": False, "time": "2026-03-03T09:00:00.000000000Z", "mid": {"o": "150.1", "h": "150.3", "l": "150.0", "c": "150.2"}}]})
        if url.endswith("/openTrades"):
            return R(200, {"trades": [{"id": "77", "instrument": "USD_JPY", "currentUnits": "-1000", "price": "150.0",
                                       "openTime": "2026-03-03T09:00:05Z", "clientExtensions": {"id": "mlx-abc"},
                                       "stopLossOrder": {"price": "150.5"}, "takeProfitOrder": {"price": "149.25"}}]})
        return R(404, {"errorMessage": "no"})

    def request(self, method, url, json=None, timeout=None):
        self.sent.append((method, url, json))
        if method == "POST":
            return R(201, {"orderFillTransaction": {"price": "150.022", "tradeOpened": {"tradeID": "88", "units": "1000"}}})
        return R(200, {"orderFillTransaction": {"price": "150.1"}})


@pytest.fixture
def b():
    o = OandaBroker("001-001-1-001", "tok", "practice")
    o.s = FakeSession()
    return o


def test_reads(b):
    assert b.account().nav == 100012.5
    p = b.prices(["USDJPY"])["USDJPY"]
    assert p.bid == 150.010 and p.ask == 150.022 and p.ts.tzinfo is not None
    c = b.candles("USDJPY", 10)
    assert len(c) == 1 and str(c.index.tz) == "UTC"          # incomplete candle dropped
    t = b.open_trades()[0]
    assert t.pair == "USDJPY" and t.units == -1000 and t.client_id == "mlx-abc" and t.sl == 150.5


def test_order_body(b):
    r = b.market_order("USDJPY", 1000.4, 149.8123, 150.4567, "mlx-xyz")
    assert r.ok and r.trade_id == "88" and r.fill_px == 150.022
    _, url, body = b.s.sent[-1]
    o = body["order"]
    assert url.endswith("/accounts/001-001-1-001/orders")
    assert o["instrument"] == "USD_JPY" and o["units"] == "1000" and o["type"] == "MARKET" and o["timeInForce"] == "FOK"
    assert o["stopLossOnFill"]["price"] == "149.812" and o["takeProfitOnFill"]["price"] == "150.457"
    assert o["tradeClientExtensions"]["id"] == "mlx-xyz"


def test_live_host():
    assert OandaBroker("a", "t", "live").base.startswith("https://api-fxtrade.oanda.com")
    with pytest.raises(ValueError):
        OandaBroker("a", "t", "prod")
