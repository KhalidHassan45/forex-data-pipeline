"""Bridge to ml-lab: reads model cards and paper evidence (read-only) and turns the latest closed bar into a signal
with EXACTLY the feature code and frozen thresholds the model was validated with."""
import json
import os
import sys

import numpy as np
import pandas as pd


class MLBridge:
    def __init__(self, cfg):
        self.cfg = cfg
        os.environ["DATA_DIR"] = str(cfg.ml_data_dir)          # ml-lab modules read DATA_DIR at import time
        sys.path.insert(0, str(cfg.ml_code_dir))
        import common, features, labels, vault, joblib  # noqa: E401
        self.C, self.F, self.LB, self.V, self.joblib = common, features, labels, vault, joblib
        self._art = {}

    # ------------------------------------------------------------ models & evidence
    def cards(self):
        out = []
        for f in sorted((self.cfg.ml_data_dir / "ml" / "models").glob("*/card.json")):
            c = json.loads(f.read_text())
            c["_dir"] = f.parent
            out.append(c)
        return out

    def paper_evidence(self, card):
        d = self.cfg.ml_data_dir / "ml" / "paper" / self.V.safe(card["id"]) / "ledger.parquet"
        since = pd.Timestamp(card.get("paper_since") or pd.Timestamp.utcnow())
        days = (pd.Timestamp.now(tz="UTC") - since).days
        if not d.exists():
            return {"days": days, "trades": 0, "pf": 0.0}
        t = pd.read_parquet(d)
        g, l = t.loc[t.ret > 0, "ret"].sum(), -t.loc[t.ret < 0, "ret"].sum()
        return {"days": int(days), "trades": int(len(t)), "pf": round(float(g / l), 3) if l > 0 else (99.0 if g > 0 else 0.0)}

    def expected_spread_px(self, pair):
        return self.C.SPREAD_PIPS.get(pair, self.C.DEFAULT_SPREAD) * self.C.pip_size(pair)

    def feature_version(self):
        return self.F.version()

    # ------------------------------------------------------------ signal
    def signal(self, card, D):
        if card["feature_version"] != self.F.version():
            return {"side": 0, "skip": "feature code differs from the validated version"}
        pair = card["pair"]
        df = D[pair]
        X = self.F.compute(df, D, pair)
        last = X.iloc[[-1]]
        if last.isna().mean(axis=1).iloc[0] > 0.2:
            return {"side": 0, "skip": "not enough history for features"}
        if card["id"] not in self._art:
            self._art[card["id"]] = self.joblib.load(card["_dir"] / "model.joblib")
        art = self._art[card["id"]]
        P, s, size = self.V.predict(art, last)
        lc = card["label_cfg"]
        atr = float(self.LB.atr(df, int(lc.get("atr_n", self.LB.DEFAULT["atr_n"]))).iloc[-1])
        return {"side": int(s.iloc[0]), "size": float(size.iloc[0]), "bar_ts": last.index[0], "atr": atr,
                "p_long": round(float(P["p_long"].iloc[0]), 4), "p_short": round(float(P["p_short"].iloc[0]), 4),
                "pt": float(lc["pt"]), "sl": float(lc["sl"]), "horizon": int(lc["horizon"])}
