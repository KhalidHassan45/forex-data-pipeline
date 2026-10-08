"""
Strategy Spec — a small, declarative language for trading rules, and its compiler.

Every uploaded strategy (Excel template, PDF, Pine, MQL, text) is turned into a SPEC, never into
free code. The compiler below is the only thing that turns a spec into positions, so a malicious
file cannot run anything: the worst it can do is produce a bad strategy that fails the lab.

Spec (JSON):
{
  "name": "rsi_trend_pullback",                  # snake_case
  "family": "reversion" | "trend" | "session" | "other",
  "description": "...",
  "params": {"band": [10, 20], "rsi_n": [2]},     # ≤ 8 combinations in total
  "indicators": {
      "rsi":    {"type": "rsi", "period": "{rsi_n}"},
      "trend":  {"type": "sma", "period": 200},
      "exit_ma":{"type": "sma", "period": 5}
  },
  "rules": {
      "long_entry":  "rsi < {band} and close > trend",
      "long_exit":   "close > exit_ma",
      "short_entry": "rsi > 100 - {band} and close < trend",
      "short_exit":  "close < exit_ma"
  },
  "stop_atr": 0            # optional: ATR multiple handled by the lab's stop logic (0 = none)
}

Expression syntax (rules):
  numbers, indicator names, open high low close, hour weekday day month,
  name[n] = value n bars ago (n ≥ 1 only), + - * /, < <= > >= == !=, and or not, ( ),
  crosses_above(a, b), crosses_below(a, b), abs(x), {param} placeholders.
Decisions are taken on the close of each bar; the lab executes them on the next bar.
"""
import itertools
import re

import numpy as np
import pandas as pd

MAX_COMBOS = 8
MAX_INDICATORS = 16
MAX_PERIOD = 2000
FAMILIES = {"trend", "reversion", "session", "other"}
SOURCES = {"open", "high", "low", "close"}
BUILTINS = SOURCES | {"hour", "weekday", "day", "month"}
RULE_KEYS = ("long_entry", "long_exit", "short_entry", "short_exit")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")

INDICATOR_TYPES = {
    # type: (required numeric args, optional args with defaults)
    "sma": (["period"], {"source": "close"}),
    "ema": (["period"], {"source": "close"}),
    "rsi": (["period"], {"source": "close"}),
    "atr": (["period"], {}),
    "stdev": (["period"], {"source": "close"}),
    "highest": (["period"], {"source": "high"}),
    "lowest": (["period"], {"source": "low"}),
    "bb_upper": (["period", "k"], {"source": "close"}),
    "bb_lower": (["period", "k"], {"source": "close"}),
    "roc": (["period"], {"source": "close"}),          # % change over `period` bars
    "zscore": (["period"], {"source": "close"}),
    "macd": (["fast", "slow"], {"source": "close"}),   # fast EMA − slow EMA
    "macd_signal": (["fast", "slow", "signal"], {"source": "close"}),
}


class SpecError(ValueError):
    """Raised with a message the dashboard shows to the user (Arabic + the technical detail)."""


# ---------------------------------------------------------------- expression parser (no eval)
TOKEN_RE = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|([A-Za-z_][A-Za-z0-9_]*)|(<=|>=|==|!=|[()\[\],+\-*/<>]))")


def tokenize(src):
    pos, out = 0, []
    src = src.strip()
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m or m.end() == pos:
            raise SpecError(f"رمز غير مفهوم في القاعدة عند: «{src[pos:pos + 12]}»")
        num, name, op = m.groups()
        if num is not None:
            out.append(("num", float(num)))
        elif name is not None:
            low = name.lower()
            out.append(("kw", low) if low in ("and", "or", "not") else ("name", low))
        else:
            out.append(("op", op))
        pos = m.end()
    return out


class Parser:
    """Recursive-descent parser → small AST of tuples. Never calls eval/exec."""

    def __init__(self, src):
        self.t = tokenize(src)
        self.i = 0
        self.src = src

    def peek(self, kind=None, val=None):
        if self.i >= len(self.t):
            return None
        k, v = self.t[self.i]
        if (kind is None or k == kind) and (val is None or v == val):
            return self.t[self.i]
        return None

    def take(self, kind=None, val=None):
        tok = self.peek(kind, val)
        if tok is None:
            got = self.t[self.i][1] if self.i < len(self.t) else "نهاية القاعدة"
            raise SpecError(f"توقعت «{val or kind}» ووجدت «{got}» في: {self.src}")
        self.i += 1
        return tok

    def parse(self):
        node = self.or_()
        if self.i != len(self.t):
            raise SpecError(f"بقايا غير مفهومة في القاعدة: {self.src}")
        return node

    def or_(self):
        n = self.and_()
        while self.peek("kw", "or"):
            self.i += 1
            n = ("or", n, self.and_())
        return n

    def and_(self):
        n = self.not_()
        while self.peek("kw", "and"):
            self.i += 1
            n = ("and", n, self.not_())
        return n

    def not_(self):
        if self.peek("kw", "not"):
            self.i += 1
            return ("not", self.not_())
        return self.cmp()

    def cmp(self):
        a = self.arith()
        for op in ("<=", ">=", "==", "!=", "<", ">"):
            if self.peek("op", op):
                self.i += 1
                return ("cmp", op, a, self.arith())
        return a

    def arith(self):
        n = self.term()
        while self.peek("op", "+") or self.peek("op", "-"):
            op = self.take()[1]
            n = ("bin", op, n, self.term())
        return n

    def term(self):
        n = self.factor()
        while self.peek("op", "*") or self.peek("op", "/"):
            op = self.take()[1]
            n = ("bin", op, n, self.factor())
        return n

    def factor(self):
        if self.peek("op", "-"):
            self.i += 1
            return ("neg", self.factor())
        if self.peek("op", "("):
            self.i += 1
            n = self.or_()
            self.take("op", ")")
            return n
        tok = self.peek("num")
        if tok:
            self.i += 1
            return ("num", tok[1])
        name = self.take("name")[1]
        if self.peek("op", "("):                         # function call
            self.i += 1
            args = [self.or_()]
            while self.peek("op", ","):
                self.i += 1
                args.append(self.or_())
            self.take("op", ")")
            if name in ("crosses_above", "crosses_below") and len(args) == 2:
                return (name, args[0], args[1])
            if name == "abs" and len(args) == 1:
                return ("abs", args[0])
            raise SpecError(f"دالة غير مدعومة: {name}()")
        lag = 0
        if self.peek("op", "["):
            self.i += 1
            n = self.take("num")[1]
            self.take("op", "]")
            if n < 1 or n != int(n) or n > MAX_PERIOD:
                raise SpecError(f"الإزاحة {name}[{n}] يجب أن تكون عددًا صحيحًا ≥ 1 (الماضي فقط)")
            lag = int(n)
        return ("var", name, lag)


def names_in(node, acc=None):
    acc = set() if acc is None else acc
    if isinstance(node, tuple):
        if node[0] == "var":
            acc.add(node[1])
        for x in node[1:]:
            names_in(x, acc)
    return acc


# ---------------------------------------------------------------- indicators
def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df, n):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def compute_indicator(df, spec):
    t = spec["type"]
    src = df[spec.get("source", INDICATOR_TYPES[t][1].get("source", "close"))]
    p = spec.get("period")
    if t == "sma": return src.rolling(p).mean()
    if t == "ema": return src.ewm(span=p, adjust=False, min_periods=p).mean()
    if t == "rsi": return _rsi(src, p)
    if t == "atr": return _atr(df, p)
    if t == "stdev": return src.rolling(p).std(ddof=0)
    if t == "highest": return src.rolling(p).max()
    if t == "lowest": return src.rolling(p).min()
    if t in ("bb_upper", "bb_lower"):
        m, sd = src.rolling(p).mean(), src.rolling(p).std(ddof=0)
        return m + spec["k"] * sd if t == "bb_upper" else m - spec["k"] * sd
    if t == "roc": return (src / src.shift(p) - 1) * 100
    if t == "zscore": return (src - src.rolling(p).mean()) / src.rolling(p).std(ddof=0)
    if t in ("macd", "macd_signal"):
        m = src.ewm(span=spec["fast"], adjust=False).mean() - src.ewm(span=spec["slow"], adjust=False).mean()
        return m if t == "macd" else m.ewm(span=spec["signal"], adjust=False).mean()
    raise SpecError(f"مؤشر غير مدعوم: {t}")


# ---------------------------------------------------------------- evaluation
def evaluate(node, env):
    k = node[0]
    if k == "num": return node[1]
    if k == "var":
        v = env[node[1]]
        return v.shift(node[2]) if node[2] else v
    if k == "neg": return -evaluate(node[1], env)
    if k == "abs": return abs(evaluate(node[1], env))
    if k == "bin":
        a, b = evaluate(node[2], env), evaluate(node[3], env)
        if node[1] == "/":
            b = b.replace(0, np.nan) if isinstance(b, pd.Series) else (np.nan if b == 0 else b)
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[node[1]]
    if k == "cmp":
        a, b = evaluate(node[2], env), evaluate(node[3], env)
        r = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[node[1]]
        return _bool(r, env)
    if k in ("and", "or"):
        a, b = _bool(evaluate(node[1], env), env), _bool(evaluate(node[2], env), env)
        return (a & b) if k == "and" else (a | b)
    if k == "not": return ~_bool(evaluate(node[1], env), env)
    if k in ("crosses_above", "crosses_below"):
        a, b = evaluate(node[1], env), evaluate(node[2], env)
        a = a if isinstance(a, pd.Series) else pd.Series(a, index=env["close"].index)
        b = b if isinstance(b, pd.Series) else pd.Series(b, index=env["close"].index)
        if k == "crosses_above": return _bool((a > b) & (a.shift() <= b.shift()), env)
        return _bool((a < b) & (a.shift() >= b.shift()), env)
    raise SpecError(f"عقدة غير معروفة: {k}")


def _bool(x, env):
    if isinstance(x, pd.Series):
        return x.fillna(False).astype(bool)
    return pd.Series(bool(x), index=env["close"].index)


# ---------------------------------------------------------------- validation + compile
def _fill(value, params):
    """Replace {param} placeholders; numeric strings become numbers."""
    if isinstance(value, str):
        def rep(m):
            if m.group(1) not in params:
                raise SpecError(f"المعامل {{{m.group(1)}}} غير معرّف في params")
            return str(params[m.group(1)])
        s = re.sub(r"\{([a-z_][a-z0-9_]*)\}", rep, value)
        return s
    return value


def _num(v, field):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise SpecError(f"القيمة «{v}» في {field} ليست رقمًا")
    return f


def grid_of(spec):
    params = spec.get("params") or {}
    for k, vals in params.items():
        if not NAME_RE.match(k):
            raise SpecError(f"اسم معامل غير صالح: {k}")
        if not isinstance(vals, list) or not vals:
            raise SpecError(f"قيم المعامل {k} يجب أن تكون قائمة غير فارغة")
    keys = list(params)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[params[k] for k in keys])] or [{}]
    if len(combos) > MAX_COMBOS:
        raise SpecError(f"عدد التوليفات {len(combos)} أكبر من {MAX_COMBOS} — قلّل قيم المعاملات (خطر overfitting)")
    return combos


def resolve(spec, params):
    """Return concrete indicators + parsed rule ASTs for one parameter combination."""
    inds = {}
    for name, ind in (spec.get("indicators") or {}).items():
        if not NAME_RE.match(name) or name in BUILTINS:
            raise SpecError(f"اسم مؤشر غير صالح أو محجوز: {name}")
        t = str(ind.get("type", "")).lower()
        if t not in INDICATOR_TYPES:
            raise SpecError(f"نوع مؤشر غير مدعوم: «{t}». المدعوم: {', '.join(INDICATOR_TYPES)}")
        req, opt = INDICATOR_TYPES[t]
        out = {"type": t}
        for f in req:
            if f not in ind:
                raise SpecError(f"المؤشر {name} ينقصه الحقل {f}")
            v = _num(_fill(ind[f], params), f"{name}.{f}")
            if f != "k":
                if v != int(v) or not (1 <= v <= MAX_PERIOD):
                    raise SpecError(f"{name}.{f} يجب أن يكون عددًا صحيحًا بين 1 و {MAX_PERIOD}")
                v = int(v)
            out[f] = v
        src = str(_fill(ind.get("source", opt.get("source", "close")), params)).lower()
        if "source" in opt:
            if src not in SOURCES:
                raise SpecError(f"مصدر غير صالح للمؤشر {name}: {src}")
            out["source"] = src
        inds[name] = out
    if len(inds) > MAX_INDICATORS:
        raise SpecError(f"عدد المؤشرات أكبر من {MAX_INDICATORS}")
    rules = {}
    for key in RULE_KEYS:
        src = (spec.get("rules") or {}).get(key)
        if src is None or str(src).strip() == "":
            continue
        ast = Parser(_fill(str(src), params)).parse()
        unknown = names_in(ast) - set(inds) - BUILTINS
        if unknown:
            raise SpecError(f"أسماء غير معرّفة في {key}: {', '.join(sorted(unknown))}")
        rules[key] = ast
    if "long_entry" not in rules and "short_entry" not in rules:
        raise SpecError("لا توجد قاعدة دخول (long_entry أو short_entry)")
    return inds, rules


def validate(spec):
    if not isinstance(spec, dict):
        raise SpecError("المواصفة يجب أن تكون كائن JSON")
    name = str(spec.get("name", ""))
    if not NAME_RE.match(name):
        raise SpecError(f"اسم الاستراتيجية «{name}» يجب أن يكون snake_case بالإنجليزية")
    fam = spec.get("family", "other")
    if fam not in FAMILIES:
        raise SpecError(f"العائلة يجب أن تكون واحدة من: {', '.join(sorted(FAMILIES))}")
    stop = spec.get("stop_atr", 0) or 0
    if not (0 <= float(stop) <= 10):
        raise SpecError("stop_atr يجب أن يكون بين 0 و 10")
    grid = grid_of(spec)
    for g in grid:
        resolve(spec, g)
    return grid


def positions(df, inds, rules):
    env = {s: df[s] for s in SOURCES}
    idx = df.index
    env.update(hour=pd.Series(idx.hour, index=idx), weekday=pd.Series(idx.dayofweek, index=idx),
               day=pd.Series(idx.day, index=idx), month=pd.Series(idx.month, index=idx))
    for n, ind in inds.items():
        env[n] = compute_indicator(df, ind)
    sig = {k: evaluate(a, env).to_numpy() for k, a in rules.items()}
    n = len(df)
    le, lx = sig.get("long_entry"), sig.get("long_exit")
    se, sx = sig.get("short_entry"), sig.get("short_exit")
    pos = np.zeros(n)
    p = 0.0
    for i in range(n):
        go_long = le is not None and le[i]
        go_short = se is not None and se[i]
        if p > 0:
            if go_short and not go_long:
                p = -1.0
            elif lx is not None and lx[i]:
                p = 0.0
        elif p < 0:
            if go_long and not go_short:
                p = 1.0
            elif sx is not None and sx[i]:
                p = 0.0
        if p == 0:
            if go_long and not go_short:
                p = 1.0
            elif go_short and not go_long:
                p = -1.0
        pos[i] = p
    return pd.Series(pos, index=idx)


def compile_spec(spec):
    """→ STRATEGY tuple (fn, grid, family) compatible with forex-backtest."""
    grid = validate(spec)
    stop = float(spec.get("stop_atr", 0) or 0)

    def fn(df, **params):
        inds, rules = resolve(spec, params)
        pos = positions(df, inds, rules)
        if stop:
            from engine import apply_atr_stop  # forex-backtest
            pos = apply_atr_stop(df, pos, stop)
        return pos

    fn.__name__ = spec["name"]
    return fn, grid, spec.get("family", "other")


def describe_ar(spec):
    """Human summary for the dashboard."""
    r = spec.get("rules", {})
    lines = []
    labels = {"long_entry": "دخول شراء", "long_exit": "خروج من الشراء", "short_entry": "دخول بيع", "short_exit": "خروج من البيع"}
    for k in RULE_KEYS:
        if r.get(k):
            lines.append({"label": labels[k], "rule": r[k]})
    inds = [{"name": n, "def": ", ".join(f"{k}={v}" for k, v in i.items())} for n, i in (spec.get("indicators") or {}).items()]
    return {"rules": lines, "indicators": inds, "params": spec.get("params") or {}, "stop_atr": spec.get("stop_atr", 0)}
