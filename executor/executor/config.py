"""Configuration: environment variables (secrets) + /state/config.json (limits). Secrets never go in the JSON."""
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

LIVE_ACK_PHRASE = "REAL-MONEY-APPROVED-BY-KHALID"
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "XAUUSD"]


@dataclass
class RiskLimits:
    risk_per_trade_pct: float = 0.5          # equity at risk to the stop, before the meta-model size factor
    max_open_risk_pct: float = 2.0           # sum of all open stop-risks
    max_positions: int = 6
    max_currency_exposure_x: float = 1.5     # |net notional per currency| / equity
    max_gross_leverage_x: float = 5.0        # sum |notional| / equity
    daily_loss_limit_pct: float = 2.0        # from start-of-UTC-day equity → no new entries today
    max_drawdown_pct: float = 8.0            # from high-water mark → KILL: flatten + lock until a named reset
    max_spread_mult: float = 2.0             # live spread vs the cost table the model was validated with
    max_orders_per_hour: int = 10
    max_price_age_sec: int = 120
    min_equity: float = 1000.0
    news_blackout_min: int = 15              # needs calendar.csv in ml-lab


@dataclass
class Gates:
    """Evidence a model needs before each stage. Mirrors the ml-lab plan: vault → 3 months paper → practice → live."""
    paper_min_days: int = 90
    paper_min_trades: int = 40
    paper_min_pf: float = 1.0
    practice_min_days: int = 60
    practice_min_trades: int = 30
    practice_min_pf: float = 1.0


@dataclass
class Config:
    mode: str = "practice"                   # sim | practice | live
    state_dir: Path = Path("/state")
    ml_data_dir: Path = Path("/mldata")      # ml-lab's /data, mounted READ-ONLY
    ml_code_dir: Path = Path("/app/ml")
    universe: list = field(default_factory=lambda: list(UNIVERSE))
    candles: int = 1200
    risk: RiskLimits = field(default_factory=RiskLimits)
    gates: Gates = field(default_factory=Gates)
    oanda_account: str = ""
    oanda_token: str = ""
    live_ack: str = ""
    live_accounts: list = field(default_factory=list)
    telegram_token: str = ""
    telegram_chat: str = ""

    @classmethod
    def load(cls):
        c = cls(
            mode=os.getenv("FX_MODE", "practice"),
            state_dir=Path(os.getenv("FX_STATE_DIR", "/state")),
            ml_data_dir=Path(os.getenv("ML_DATA_DIR", "/mldata")),
            ml_code_dir=Path(os.getenv("ML_CODE_DIR", "/app/ml")),
            oanda_account=os.getenv("OANDA_ACCOUNT_ID", ""),
            oanda_token=os.getenv("OANDA_TOKEN", ""),
            live_ack=os.getenv("FX_LIVE_ACK", ""),
            live_accounts=[a for a in os.getenv("FX_LIVE_ACCOUNTS", "").split(",") if a],
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
        f = c.state_dir / "config.json"
        if f.exists():
            j = json.loads(f.read_text())
            c.risk = RiskLimits(**{**asdict(c.risk), **j.get("risk", {})})
            c.gates = Gates(**{**asdict(c.gates), **j.get("gates", {})})
            c.universe = j.get("universe", c.universe)
        c.validate()
        return c

    def validate(self):
        if self.mode not in ("sim", "practice", "live"):
            raise SystemExit(f"FX_MODE must be sim|practice|live, got {self.mode}")
        r = self.risk
        hard = {"risk_per_trade_pct": 1.0, "max_open_risk_pct": 5.0, "max_gross_leverage_x": 10.0,
                "daily_loss_limit_pct": 5.0, "max_drawdown_pct": 20.0}
        for k, cap in hard.items():
            if getattr(r, k) <= 0 or getattr(r, k) > cap:
                raise SystemExit(f"risk.{k}={getattr(r, k)} outside the hard cap (0, {cap}] written in code")
        if self.mode == "live":
            if self.live_ack != LIVE_ACK_PHRASE:
                raise SystemExit("live mode refused: FX_LIVE_ACK is not set to the approval phrase")
            if self.oanda_account not in self.live_accounts:
                raise SystemExit("live mode refused: account is not in FX_LIVE_ACCOUNTS allow-list")
        if self.mode in ("practice", "live") and not (self.oanda_account and self.oanda_token):
            raise SystemExit("OANDA_ACCOUNT_ID and OANDA_TOKEN are required")

    def public(self):
        """Safe to log / show: no secrets."""
        return {"mode": self.mode, "universe": self.universe, "risk": asdict(self.risk), "gates": asdict(self.gates),
                "account": (self.oanda_account[:4] + "…") if self.oanda_account else ""}
