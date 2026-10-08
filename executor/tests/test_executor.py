"""Unit tests: sizing, limits, kill switch, daily halt, idempotency, time exits, reconciliation, gating, live guards."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from executor import risk as R
from executor.brokers.base import Price
from executor.brokers.sim import SimBroker
from executor.config import LIVE_ACK_PHRASE, Config, RiskLimits
from executor.engine import Engine, market_open

T0 = pd.Timestamp("2026-03-02 00:00", tz="UTC")   # a Monday


def candles(p0, n=400, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(T0 - pd.Timedelta(hours=200), periods=n, freq="h", tz="UTC")
    c = p0 * np.exp(np.cumsum(rng.normal(drift, 0.0008, n)))
    o = np.r_[p0, c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0004, "low": np.minimum(o, c) * 0.9996, "close": c}, index=idx)


class FakeBridge:
    def __init__(self, cards, side=1, size=1.0, atr=0.0020, horizon=6):
        self._cards, self.side, self.size, self.atr, self.h = cards, side, size, atr, horizon
        self.calls = 0

    def cards(self):
        return self._cards

    def expected_spread_px(self, pair):
        return 0.0001 if not pair.endswith("JPY") else 0.01

    def paper_evidence(self, card):
        return card.get("_ev", {"days": 0, "trades": 0, "pf": 0})

    def feature_version(self):
        return "fTEST"

    def signal(self, card, D):
        self.calls += 1
        bar = D[card["pair"]].index[-1]
        return {"side": self.side, "size": self.size, "bar_ts": bar, "atr": self.atr if not card["pair"].endswith("JPY") else 0.2,
                "p_long": .6, "p_short": .1, "pt": 1.5, "sl": 1.0, "horizon": self.h}


def card(mid="ml|EURUSD|h1|x", pair="EURUSD", status="PAPER_ACTIVE", **kw):
    return {"id": mid, "pair": pair, "status": status, "feature_version": "fTEST",
            "label_cfg": {"horizon": 6, "pt": 1.5, "sl": 1.0}, "vault": {"net_max_dd_pct": -5}, **kw}


@pytest.fixture
def env(tmp_path):
    def make(cards, mode="sim", risk=None, **bk):
        cfg = Config(mode=mode, state_dir=tmp_path, universe=["EURUSD", "USDJPY", "GBPUSD", "EURGBP"])
        if risk:
            cfg.risk = RiskLimits(**risk)
        data = {"EURUSD": candles(1.10, seed=1), "USDJPY": candles(150.0, seed=2), "GBPUSD": candles(1.30, seed=3),
                "EURGBP": candles(0.85, seed=4)}
        b = SimBroker(data, {"EURUSD": 0.0001, "USDJPY": 0.01, "GBPUSD": 0.00012, "EURGBP": 0.00012})
        clock = {"t": T0}
        b.set_time(T0)
        br = FakeBridge(cards, **bk)
        e = Engine(cfg, b, br, now_fn=lambda: clock["t"])

        def step(h=1):
            for _ in range(h):
                clock["t"] += pd.Timedelta(hours=1); b.set_time(clock["t"])
            return e.run_once()
        e.step, e.clock = step, clock
        return e
    return make


# ---------------------------------------------------------------- sizing & conversion
def P(**kv):
    return {k: Price(k, v - 1e-5, v + 1e-5, T0) for k, v in kv.items()}


def test_size_usd_quote():
    u = R.size_units(100_000, 0.5, 1.0, 1.1000, 1.0980, "EURUSD", P(EURUSD=1.1), "USD")
    assert u == pytest.approx(250_000, rel=1e-3)            # $500 risk / 0.0020


def test_size_jpy_quote_converts():
    u = R.size_units(100_000, 0.5, 1.0, 150.00, 149.80, "USDJPY", P(USDJPY=150.0), "USD")
    loss = 0.20 * u / 150.0
    assert loss == pytest.approx(500, rel=1e-2)


def test_size_cross_and_meta_factor():
    px = P(EURGBP=0.85, GBPUSD=1.30)
    u = R.size_units(100_000, 0.5, 0.5, 0.8500, 0.8480, "EURGBP", px, "USD")
    assert 0.0020 * u * 1.30 == pytest.approx(250, rel=1e-2)


def test_currency_exposure_netting():
    px = P(EURUSD=1.1, USDJPY=150.0)
    ex = R.notional_by_ccy([R.Position("EURUSD", 10_000, 1.1, 1.09), R.Position("USDJPY", 11_000, 150, 149)], px, "USD")
    assert ex["EUR"] == pytest.approx(11_000, rel=1e-3)
    assert ex["USD"] == pytest.approx(0, abs=50)          # short USD via EURUSD, long USD via USDJPY


def test_limits_block():
    L = RiskLimits(max_open_risk_pct=1.0)
    px = P(EURUSD=1.1)
    open_ = [R.Position("EURUSD", 30_000, 1.1, 1.098)]       # 0.6% risk on 100k... ($60)?
    new = R.Position("EURUSD", -600_000, 1.1, 1.102)
    why = R.check(new, open_, 100_000, px, "USD", L, 0)
    assert any("opposite" in w for w in why) and any("open risk" in w for w in why)


def test_hard_caps_in_code():
    c = Config(mode="sim", risk=RiskLimits(risk_per_trade_pct=3.0))
    with pytest.raises(SystemExit):
        c.validate()


def test_live_requires_ack_and_allowlist():
    with pytest.raises(SystemExit):
        Config(mode="live", oanda_account="A", oanda_token="t").validate()
    with pytest.raises(SystemExit):
        Config(mode="live", oanda_account="A", oanda_token="t", live_ack=LIVE_ACK_PHRASE, live_accounts=["B"]).validate()
    Config(mode="live", oanda_account="A", oanda_token="t", live_ack=LIVE_ACK_PHRASE, live_accounts=["A"]).validate()


def test_market_hours():
    assert not market_open(pd.Timestamp("2026-03-07 12:00", tz="UTC"))     # Saturday
    assert not market_open(pd.Timestamp("2026-03-06 20:30", tz="UTC"))     # Friday evening
    assert not market_open(pd.Timestamp("2026-03-03 22:00", tz="UTC"))     # rollover
    assert market_open(pd.Timestamp("2026-03-03 10:00", tz="UTC"))


# ---------------------------------------------------------------- engine
def test_one_order_per_bar_and_risk_respected(env):
    e = env([card()])
    s = e.step()
    s = e.run_once(); e.run_once()                            # same bar again → no duplicate
    tr = e.st.trades()
    assert len(tr) == 1 and tr[0]["status"] == "OPEN"
    assert 0 < tr[0]["risk_amt"] / 100_000 * 100 <= 0.5 + 1e-9      # never above target (scaled to fit exposure caps)


def test_full_risk_when_limits_allow(env):
    e = env([card()], atr=0.01)
    e.step()
    assert e.st.trades()[0]["risk_amt"] / 100_000 * 100 == pytest.approx(0.5, rel=0.02)


def test_scaled_down_to_exposure_cap(env):
    e = env([card()], atr=0.002)                                # tight stop → full risk would need 2.7× EUR exposure
    e.step()
    t = e.st.trades()[0]
    assert t["status"] == "OPEN" and abs(t["units"]) * 1.1 / 100_000 <= 1.5


def test_time_exit_after_horizon(env):
    e = env([card()], atr=0.05)                               # stops far away → only time can close it
    e.step()
    e.step(7)
    t = e.st.trades()[0]
    assert t["status"] == "CLOSED"


def test_model_not_paper_active_never_trades(env):
    e = env([card(status="VAULT_FAILED"), card(mid="m2", status="PAPER_PAUSED")])
    e.step(3)
    assert e.st.trades() == []


def test_practice_needs_recorded_approval(env, tmp_path):
    e = env([card()], mode="sim")
    e.cfg.mode = "practice"                                   # stage 'practice' without approval
    e.step()
    assert e.st.trades() == []
    ok, info = e.approve("ml|EURUSD|h1|x", "practice", "Khalid")
    assert not ok and "evidence" in info                      # 0 days of paper < 90


def test_practice_approval_with_evidence(env):
    e = env([card(_ev={"days": 120, "trades": 55, "pf": 1.3})])
    e.cfg.mode = "practice"
    ok, _ = e.approve("ml|EURUSD|h1|x", "practice", "Khalid")
    assert ok
    from executor.state import State
    e.st = State(e.cfg.state_dir / "practice" / "state.db")
    e.step()
    assert len(e.st.trades("OPEN")) == 1


def test_drawdown_kill_flattens_and_locks(env):
    e = env([card()], risk={"max_drawdown_pct": 0.3})
    e.step()
    assert len(e.st.trades("OPEN")) == 1
    e.b.cash -= 1_000                                          # simulate a 1% loss
    s = e.step()
    assert s.get("kill") and not e.st.trades("OPEN")
    e.step(2)
    assert not e.st.trades("OPEN")                             # locked: no new entries


def test_daily_loss_halts_entries(env):
    e = env([card(), card(mid="ml|GBPUSD|h1|y", pair="GBPUSD")], risk={"daily_loss_limit_pct": 0.5}, atr=0.05)
    e.st.set("day", str(T0.date())); e.st.set("day_start", 100_000)
    e.b.cash -= 600
    e.step()
    assert e.st.trades() == [] and e.st.get("halt_day")


def test_rejection_and_unknown_outcome_reconcile(env):
    e = env([card()])
    e.b.reject_next = "MARKET_HALTED"
    e.step()
    assert e.st.trades()[0]["status"] == "REJECTED"

    def boom(*a, **k):
        raise ConnectionError("timeout after send")
    real = e.b.market_order
    e.b.market_order = lambda *a, **k: (real(*a, **k), boom())  # order IS placed, then the call fails
    e.step()
    e.b.market_order = real
    unk = [t for t in e.st.trades() if t["status"] == "UNKNOWN"]
    assert len(unk) == 1
    e.reconcile()
    assert [t for t in e.st.trades() if t["client_id"] == unk[0]["client_id"]][0]["status"] in ("OPEN", "CLOSED")


def test_opposite_direction_blocked(env):
    e = env([card(), card(mid="ml|EURUSD|h1|z")], atr=0.01)
    e.step()
    assert len(e.st.trades("OPEN")) == 2                       # two models, same direction: allowed within limits
    e.ml.side = -1
    e._consider(card(mid="ml|EURUSD|h1|w"), {"EURUSD": e.b.candles("EURUSD", 300)}, e.b.prices(["EURUSD"]),
                e.b.account().nav, "USD", True, e.clock["t"])
    assert len(e.st.trades("OPEN")) == 2


def test_spread_filter(env):
    e = env([card()])
    e.b.spr["EURUSD"] = 0.0010                                 # 10× the validated spread
    e.step()
    assert e.st.trades() == []


def test_stale_signal_bar_skipped(env):
    e = env([card()])
    e.clock["t"] += pd.Timedelta(minutes=45); e.b.now = e.clock["t"]
    e.run_once()
    assert e.st.trades() == []
