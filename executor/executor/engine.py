"""One execution cycle. Safe to run every few minutes: entries are evaluated once per closed H1 bar per model.

Order of a cycle:  account → kill switch / drawdown → reconcile with broker → time exits → daily-loss halt
                   → eligible models → signal → price & spread checks → sizing → portfolio limits → order."""
import hashlib
import json
from datetime import timedelta

import pandas as pd

from . import risk as R
from .audit import Audit
from .state import State

STAGE_FOR_MODE = {"sim": "paper", "practice": "practice", "live": "live"}


def client_id(model_id, bar_ts):
    return "mlx-" + hashlib.sha1(f"{model_id}|{bar_ts}".encode()).hexdigest()[:16]


def market_open(ts: pd.Timestamp):
    """No NEW entries: Fri ≥ 20:00 UTC → Sun 22:00 UTC (thin weekend liquidity) and 21:50–22:15 UTC daily rollover."""
    d, m = ts.dayofweek, ts.hour * 60 + ts.minute
    if d == 5 or (d == 4 and m >= 20 * 60) or (d == 6 and m < 22 * 60):
        return False
    return not (21 * 60 + 50 <= m < 22 * 60 + 15)


class Engine:
    def __init__(self, cfg, broker, bridge, now_fn=None):
        self.cfg, self.b, self.ml = cfg, broker, bridge
        self.st = State(cfg.state_dir / cfg.mode / "state.db")
        self.audit = Audit(cfg)
        self.now = now_fn or (lambda: pd.Timestamp.now(tz="UTC"))
        self.L = cfg.risk

    # ------------------------------------------------------------ eligibility (the ml-lab ladder, enforced)
    def eligible(self, card):
        if card["status"] == "PAPER_PAUSED":
            return False, "paused by ml-lab kill switch"
        if card["status"] != "PAPER_ACTIVE":
            return False, f"ml-lab status {card['status']}"
        stage = STAGE_FOR_MODE[self.cfg.mode]
        if stage == "paper":
            return True, ""
        if not self.st.approval(card["id"], stage):
            return False, f"no {stage} approval recorded"
        return True, ""

    # ------------------------------------------------------------ helpers
    def _positions(self):
        return [R.Position(t.pair, t.units, t.entry, t.sl if t.sl is not None else t.entry) for t in self.b.open_trades()]

    def flatten(self, why):
        n = 0
        for t in self.st.trades("OPEN"):
            r = self.b.close_trade(t["trade_id"])
            self.audit.log("CLOSE", client_id=t["client_id"], reason=why, ok=r.ok, px=r.fill_px, err=r.reason)
            n += r.ok
        self.reconcile()
        return n

    def reconcile(self):
        live = {t.trade_id: t for t in self.b.open_trades()}
        by_cid = {t.client_id: t for t in live.values() if t.client_id}
        for t in self.st.trades("OPEN"):
            if t["trade_id"] in live:
                continue
            bt = self.b.trade(t["trade_id"])
            self.st.update(t["client_id"], status="CLOSED", close_ts=self.now(),
                           close_px=bt.close_px if bt else None, realized_pl=bt.realized_pl if bt else None)
            self.audit.log("CLOSED", client_id=t["client_id"], pair=t["pair"], pl=bt.realized_pl if bt else None)
        for t in self.st.trades("PENDING", "UNKNOWN"):
            bt = by_cid.get(t["client_id"]) or self.b.trade("@" + t["client_id"])
            if bt:
                st = "OPEN" if bt.state == "OPEN" else "CLOSED"
                self.st.update(t["client_id"], status=st, trade_id=bt.trade_id, entry=bt.entry, units=bt.units,
                               open_ts=bt.open_ts, realized_pl=bt.realized_pl if st == "CLOSED" else None)
                self.audit.log("RECONCILED", client_id=t["client_id"], status=st)
            elif self.now() - pd.Timestamp(t["created_ts"]) > timedelta(minutes=10):
                self.st.update(t["client_id"], status="REJECTED", reason="not found at broker after 10 min")
        mine = {t["trade_id"] for t in self.st.trades("OPEN")}
        foreign = [t for t in live.values() if t.trade_id not in mine]
        if foreign and not self.st.get("foreign_warned"):
            self.audit.alert(f"{len(foreign)} open trade(s) on this account were not placed by fx-executor — "
                             "they still count toward exposure limits")
            self.st.set("foreign_warned", True)

    def time_exits(self):
        for t in self.st.trades("OPEN"):
            c = self.b.candles(t["pair"], int(t["horizon"]) + 10)
            done = int((c.index > pd.Timestamp(t["signal_ts"])).sum())
            if done >= int(t["horizon"]):
                r = self.b.close_trade(t["trade_id"])
                self.audit.log("TIME_EXIT", client_id=t["client_id"], bars=done, ok=r.ok, px=r.fill_px, err=r.reason)
        self.reconcile()

    # ------------------------------------------------------------ the cycle
    def run_once(self):
        now = self.now()
        acct = self.b.account()
        eq = acct.nav
        hwm = max(self.st.get("hwm", eq), eq); self.st.set("hwm", hwm)
        day = str(now.date())
        if self.st.get("day") != day:
            self.st.set("day", day); self.st.set("day_start", eq); self.st.set("halt_day", None)
        status = {"ts": now, "mode": self.cfg.mode, "equity": eq, "currency": acct.currency, "hwm": hwm,
                  "dd_pct": round((eq / hwm - 1) * 100, 3), "day_pnl_pct": round((eq / self.st.get("day_start", eq) - 1) * 100, 3)}

        if self.st.get("kill"):
            self.flatten("kill switch active")
            return self._finish(status | {"kill": self.st.get("kill")})
        if status["dd_pct"] <= -self.L.max_drawdown_pct:
            k = {"ts": now, "why": f"drawdown {status['dd_pct']}% from high-water mark {hwm:.2f}"}
            self.st.set("kill", k)
            n = self.flatten("max drawdown kill switch")
            self.audit.alert(f"KILL SWITCH: {k['why']} — {n} trade(s) closed. Trading locked until a named reset.")
            return self._finish(status | {"kill": k})

        self.reconcile()
        self.time_exits()
        if status["day_pnl_pct"] <= -self.L.daily_loss_limit_pct and self.st.get("halt_day") != day:
            self.st.set("halt_day", day)
            self.audit.alert(f"daily loss {status['day_pnl_pct']}% ≥ limit: no new entries until tomorrow (UTC)")
        halted = self.st.get("halt_day") == day
        can_enter = (not halted) and market_open(now) and eq >= self.L.min_equity

        cards = self.ml.cards()
        elig = [(c, *self.eligible(c)) for c in cards]
        active = [c for c, ok, _ in elig if ok]
        status["models"] = [{"id": c["id"], "eligible": ok, "why": w} for c, ok, w in elig]
        if active:
            D = {p: self.b.candles(p, self.cfg.candles) for p in self.cfg.universe}
            D = {p: d for p, d in D.items() if len(d)}
            prices = self.b.prices(sorted(D))
            for card in active:
                self._consider(card, D, prices, eq, acct.currency, can_enter, now)
        status.update(can_enter=can_enter, halted=halted, open=[{k: t[k] for k in ("model_id", "pair", "side", "units", "entry", "sl", "tp", "open_ts")}
                                                              for t in self.st.trades("OPEN")])
        return self._finish(status)

    def _consider(self, card, D, prices, eq, acct, can_enter, now):
        mid, pair = card["id"], card["pair"]
        if pair not in D:
            self.audit.log("SKIP", model=mid, why="no candles"); return
        bar = D[pair].index[-1]
        if self.st.processed(mid, bar):
            return
        sig = self.ml.signal(card, D)
        self.st.mark(mid, bar)
        base = {"model": mid, "pair": pair, "bar": bar, **{k: sig.get(k) for k in ("side", "size", "p_long", "p_short")}}
        if sig.get("skip") or sig["side"] == 0 or sig["size"] <= 0:
            self.audit.log("NO_SIGNAL", **base, why=sig.get("skip", "")); return
        def skip(why):
            self.audit.log("SKIP", **base, why=why)
        if not can_enter:
            return skip("entries blocked (halt / market hours / equity)")
        if self.st.trades("OPEN", "PENDING", "UNKNOWN", model_id=mid):
            return skip("model already has a position")
        p = prices.get(pair)
        if not p or not p.tradeable:
            return skip("no tradeable price")
        if (now - p.ts).total_seconds() > self.L.max_price_age_sec:
            return skip(f"stale price ({p.ts})")
        exp = self.ml.expected_spread_px(pair)
        if p.spread > self.L.max_spread_mult * exp:
            return skip(f"spread {p.spread:.6f} > {self.L.max_spread_mult}× validated {exp:.6f}")
        if now - (bar + pd.Timedelta(hours=1)) > timedelta(minutes=30):
            return skip("signal bar is older than 30 minutes (late cycle)")
        side = sig["side"]
        entry = p.ask if side > 0 else p.bid
        sl = entry - side * sig["sl"] * sig["atr"]
        tp = entry + side * sig["pt"] * sig["atr"]
        up = getattr(self.b, "units_precision", lambda _: 0)(pair)
        mu = getattr(self.b, "min_units", lambda _: 1)(pair)
        units = R.size_units(eq, self.L.risk_per_trade_pct, sig["size"], entry, sl, pair, prices, acct, up, mu)
        if units <= 0:
            return skip("size rounds to zero")
        new = R.Position(pair, side * units, entry, sl)
        new, why, k = R.fit(new, self._positions(), eq, prices, acct, self.L,
                            self.st.orders_since(now - timedelta(hours=1)), up, mu)
        if new is None:
            return skip("; ".join(why))
        if k < 1:
            self.audit.log("SCALED", **base, factor=k, why="; ".join(why))
        units = abs(new.units)
        cid = client_id(mid, bar)
        risk_amt = R.stop_risk(new, prices, acct)
        self.st.insert(client_id=cid, model_id=mid, pair=pair, side=side, units=side * units, entry=entry, sl=sl, tp=tp,
                       signal_ts=bar, horizon=sig["horizon"], created_ts=now, status="PENDING",
                       risk_amt=risk_amt, size_factor=sig["size"])
        try:
            r = self.b.market_order(pair, side * units, sl, tp, cid)
        except Exception as e:                                   # outcome unknown → resolved by reconcile()
            self.st.update(cid, status="UNKNOWN", reason=str(e)[:200])
            self.audit.alert(f"order outcome unknown for {pair} ({e}); will reconcile")
            return
        if r.ok:
            self.st.update(cid, status="OPEN", trade_id=r.trade_id, entry=r.fill_px, units=r.units, open_ts=now)
            self.audit.log("OPEN", **base, client_id=cid, units=r.units, fill=r.fill_px, sl=sl, tp=tp,
                           risk_pct=round(risk_amt / eq * 100, 3))
            self.audit.alert(f"OPEN {pair} {'BUY' if side > 0 else 'SELL'} {abs(r.units):g} @ {r.fill_px} "
                             f"SL {sl:.5f} TP {tp:.5f} risk {risk_amt / eq * 100:.2f}% · {mid}")
        else:
            self.st.update(cid, status="REJECTED", reason=r.reason)
            self.audit.log("REJECTED", **base, client_id=cid, why=r.reason)

    def _finish(self, status):
        f = self.cfg.state_dir / self.cfg.mode / "status.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str))
        return status

    # ------------------------------------------------------------ approvals
    def approve(self, model_id, stage, by):
        cards = {c["id"]: c for c in self.ml.cards()}
        c = cards.get(model_id)
        if not c:
            return False, "no such model card"
        if c["status"] != "PAPER_ACTIVE":
            return False, f"ml-lab status is {c['status']} (must be PAPER_ACTIVE)"
        G = self.cfg.gates
        if stage == "practice":
            ev = self.ml.paper_evidence(c)
            need = ev["days"] >= G.paper_min_days and ev["trades"] >= G.paper_min_trades and ev["pf"] >= G.paper_min_pf
        elif stage == "live":
            prac = State(self.cfg.state_dir / "practice" / "state.db")
            if not prac.approval(model_id, "practice"):
                return False, "never approved for practice"
            closed = [t for t in prac.trades("CLOSED", model_id=model_id) if t["realized_pl"] is not None]
            first = min((pd.Timestamp(t["created_ts"]) for t in prac.trades(model_id=model_id)), default=self.now())
            g = sum(t["realized_pl"] for t in closed if t["realized_pl"] > 0)
            l = -sum(t["realized_pl"] for t in closed if t["realized_pl"] < 0)
            ev = {"days": (self.now() - first).days, "trades": len(closed), "pf": round(g / l, 3) if l > 0 else (99.0 if g > 0 else 0.0)}
            need = ev["days"] >= G.practice_min_days and ev["trades"] >= G.practice_min_trades and ev["pf"] >= G.practice_min_pf
        else:
            return False, "stage must be practice or live"
        if not need:
            return False, f"evidence not sufficient for {stage}: {ev} vs gates {G}"
        target = State(self.cfg.state_dir / stage / "state.db")
        target.approve(model_id, stage, by, self.now(), ev)
        self.audit.log("APPROVED", model=model_id, stage=stage, by=by, evidence=ev)
        return True, ev
