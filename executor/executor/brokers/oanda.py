"""OANDA v20 REST adapter. practice → api-fxpractice.oanda.com, live → api-fxtrade.oanda.com.

Safety: GETs retry with back-off; order POSTs are NEVER retried blindly — an order whose outcome is unknown is
resolved on the next cycle by looking for its client id among open trades (see engine.reconcile)."""
import time

import pandas as pd
import requests

from .base import Account, Broker, BrokerTrade, OrderResult, Price

HOSTS = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}


def inst(pair):
    return f"{pair[:3]}_{pair[3:]}"


def pair_of(instrument):
    return instrument.replace("_", "")


class OandaBroker(Broker):
    name = "oanda"

    def __init__(self, account_id, token, env="practice", timeout=10):
        if env not in HOSTS:
            raise ValueError(env)
        self.base = f"{HOSTS[env]}/v3"
        self.acc = account_id
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                               "Accept-Datetime-Format": "RFC3339"})
        self.timeout = timeout
        self._inst = {}

    # ------------------------------------------------------------ http
    def _get(self, path, params=None, tries=3):
        err = None
        for k in range(tries):
            try:
                r = self.s.get(self.base + path, params=params, timeout=self.timeout)
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json()
                err = f"HTTP {r.status_code}"
            except requests.HTTPError:
                raise
            except requests.RequestException as e:
                err = str(e)
            time.sleep(1.5 * (k + 1))
        raise ConnectionError(f"OANDA GET {path} failed: {err}")

    def _send(self, method, path, body=None):
        r = self.s.request(method, self.base + path, json=body, timeout=self.timeout)
        try:
            j = r.json()
        except ValueError:
            j = {"errorMessage": r.text[:300]}
        return r.status_code, j

    # ------------------------------------------------------------ meta
    def _instrument(self, pair):
        if pair not in self._inst:
            j = self._get(f"/accounts/{self.acc}/instruments", {"instruments": inst(pair)})
            self._inst[pair] = j["instruments"][0]
        return self._inst[pair]

    def precision(self, pair):
        return int(self._instrument(pair)["displayPrecision"])

    def units_precision(self, pair):
        return int(self._instrument(pair).get("tradeUnitsPrecision", 0))

    def min_units(self, pair):
        return float(self._instrument(pair).get("minimumTradeSize", 1))

    # ------------------------------------------------------------ reads
    def account(self):
        a = self._get(f"/accounts/{self.acc}/summary")["account"]
        return Account(nav=float(a["NAV"]), balance=float(a["balance"]), currency=a["currency"],
                       margin_used=float(a.get("marginUsed", 0)))

    def prices(self, pairs):
        j = self._get(f"/accounts/{self.acc}/pricing", {"instruments": ",".join(inst(p) for p in pairs)})
        out = {}
        for p in j.get("prices", []):
            if not p.get("bids") or not p.get("asks"):
                continue
            out[pair_of(p["instrument"])] = Price(pair_of(p["instrument"]), float(p["bids"][0]["price"]),
                                                  float(p["asks"][0]["price"]), pd.Timestamp(p["time"]),
                                                  bool(p.get("tradeable", True)))
        return out

    def candles(self, pair, count):
        j = self._get(f"/instruments/{inst(pair)}/candles", {"granularity": "H1", "count": min(int(count), 5000),
                                                             "price": "M"})
        rows = [(pd.Timestamp(c["time"]), float(c["mid"]["o"]), float(c["mid"]["h"]), float(c["mid"]["l"]),
                 float(c["mid"]["c"])) for c in j.get("candles", []) if c.get("complete")]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).set_index("ts")
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        return df

    def _trade_obj(self, t):
        return BrokerTrade(trade_id=str(t["id"]), pair=pair_of(t["instrument"]),
                           units=float(t.get("currentUnits", t.get("initialUnits", 0))), entry=float(t["price"]),
                           open_ts=pd.Timestamp(t["openTime"]), client_id=(t.get("clientExtensions") or {}).get("id", ""),
                           sl=float(t["stopLossOrder"]["price"]) if t.get("stopLossOrder") else None,
                           tp=float(t["takeProfitOrder"]["price"]) if t.get("takeProfitOrder") else None,
                           state=t.get("state", "OPEN"), realized_pl=float(t.get("realizedPL", 0) or 0),
                           close_px=float(t["averageClosePrice"]) if t.get("averageClosePrice") else None)

    def open_trades(self):
        return [self._trade_obj(t) for t in self._get(f"/accounts/{self.acc}/openTrades").get("trades", [])]

    def trade(self, trade_id):
        try:
            return self._trade_obj(self._get(f"/accounts/{self.acc}/trades/{trade_id}")["trade"])
        except requests.HTTPError:
            return None

    # ------------------------------------------------------------ writes
    def market_order(self, pair, units, sl, tp, client_id):
        prec, up = self.precision(pair), self.units_precision(pair)
        body = {"order": {
            "type": "MARKET", "instrument": inst(pair), "units": f"{round(units, up):.{up}f}",
            "timeInForce": "FOK", "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{sl:.{prec}f}", "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": f"{tp:.{prec}f}", "timeInForce": "GTC"},
            "tradeClientExtensions": {"id": client_id, "tag": "ml-lab"},
        }}
        code, j = self._send("POST", f"/accounts/{self.acc}/orders", body)
        fill = j.get("orderFillTransaction")
        if code == 201 and fill and fill.get("tradeOpened"):
            to = fill["tradeOpened"]
            return OrderResult(True, str(to["tradeID"]), float(fill["price"]), float(to["units"]), raw=j)
        reason = (j.get("orderCancelTransaction") or {}).get("reason") or j.get("errorMessage") or f"HTTP {code}"
        return OrderResult(False, reason=reason, raw=j)

    def close_trade(self, trade_id):
        code, j = self._send("PUT", f"/accounts/{self.acc}/trades/{trade_id}/close", {"units": "ALL"})
        fill = j.get("orderFillTransaction")
        if code == 200 and fill:
            return OrderResult(True, trade_id, float(fill.get("price", 0)), raw=j)
        return OrderResult(False, trade_id, reason=j.get("errorMessage") or f"HTTP {code}", raw=j)
