"""Forecasting + live evaluation for 30s BTC bars.

Model: per-bar log returns are standardized by an EWMA volatility estimate.
A heavily-shrunk ridge regression on recent standardized returns and
order-flow imbalance predicts the next bar; it is iterated 30 steps to get
a 15-minute path. Uncertainty bands come from volatility * sqrt(horizon).

BTC at this horizon is close to a random walk, so the Scorekeeper compares
every forecast against the naive "price stays where it is" baseline.
"""
import math

HORIZON = 30          # 30 bars x 30s = 15 minutes
BAR_SEC = 30
Z = {50: 0.6745, 80: 1.2816, 95: 1.96}


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def solve(A, b):
    """Gaussian elimination for the small ridge normal equations."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[c][c]:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] if M[i][i] else 0.0 for i in range(n)]


class Forecaster:
    EWMA_LAMBDA = 0.97
    RIDGE_LAMBDA = 60.0
    MIN_TRAIN = 40
    TRAIN_WINDOW = 360    # last 3 hours of bars at most
    FEATURES = ["z[t]", "z[t-1]", "mean z[t-2..t-9]", "order-flow imbalance"]

    def __init__(self):
        self.coef = [0.0] * len(self.FEATURES)
        self.n_train = 0
        self.sigma = None

    def _sigmas(self, rets):
        """EWMA sigma known at the end of each bar (no lookahead)."""
        var = sum(r * r for r in rets[:20]) / max(1, len(rets[:20])) or 1e-8
        out = []
        for r in rets:
            out.append(math.sqrt(var))
            var = self.EWMA_LAMBDA * var + (1 - self.EWMA_LAMBDA) * r * r
        return out, math.sqrt(var)

    @staticmethod
    def _feat(z, ofi, t):
        tail = z[max(0, t - 9):max(0, t - 1)]
        return [z[t], z[t - 1] if t >= 1 else 0.0,
                sum(tail) / len(tail) if tail else 0.0, ofi[t]]

    def fit_predict(self, bars, candle_sigma30=None):
        """bars: list of dicts with close, ofi. Returns forecast dict."""
        closes = [b["close"] for b in bars]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        ofi = [b["ofi"] for b in bars[1:]]
        if len(rets) < 10:
            return None
        sig_before, sig_now = self._sigmas(rets)
        if candle_sigma30:
            # blend fast (EWMA) and slow (1-min candles) volatility
            sig_now = math.sqrt(0.6 * sig_now ** 2 + 0.4 * candle_sigma30 ** 2)
        self.sigma = sig_now
        z = [r / s if s > 0 else 0.0 for r, s in zip(rets, sig_before)]
        z = [max(-6.0, min(6.0, v)) for v in z]

        # train: features at t -> z[t+1]
        start = max(1, len(z) - 1 - self.TRAIN_WINDOW)
        X = [self._feat(z, ofi, t) for t in range(start, len(z) - 1)]
        y = [z[t + 1] for t in range(start, len(z) - 1)]
        self.n_train = len(y)
        if self.n_train >= self.MIN_TRAIN:
            k = len(self.FEATURES)
            A = [[sum(x[i] * x[j] for x in X) + (self.RIDGE_LAMBDA if i == j else 0)
                  for j in range(k)] for i in range(k)]
            b = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(k)]
            self.coef = solve(A, b)
        else:
            self.coef = [0.0] * len(self.FEATURES)

        # iterate the path forward
        zz, oo = z[:], ofi[:]
        cum, points = 0.0, []
        base = closes[-1]
        t0 = bars[-1]["t"]
        for h in range(1, HORIZON + 1):
            t = len(zz) - 1
            zhat = sum(c * f for c, f in zip(self.coef, self._feat(zz, oo, t)))
            zhat = max(-0.5, min(0.5, zhat))
            zz.append(zhat)
            oo.append(oo[-1] * 0.5)           # order-flow signal decays
            cum += zhat * sig_now
            sd = sig_now * math.sqrt(h)
            p = {"h": h, "t": t0 + h * BAR_SEC, "mean": base * math.exp(cum),
                 "p_up": phi(cum / sd) if sd > 0 else 0.5}
            for lvl, zq in Z.items():
                p[f"lo{lvl}"] = base * math.exp(cum - zq * sd)
                p[f"hi{lvl}"] = base * math.exp(cum + zq * sd)
            points.append(p)
        return {"issued_at": t0, "base": base, "sigma30": sig_now, "points": points}


class Scorekeeper:
    """Matches past forecasts to realized closes; model vs random walk."""

    def __init__(self):
        self.pending = {}     # issued_at -> forecast
        self.history = {}     # issued_at -> forecast (kept for overlay)
        self.stats = [dict(n=0, ae_m=0.0, ae_rw=0.0, hit=0, dir_n=0, in80=0)
                      for _ in range(HORIZON + 1)]

    def add(self, fc):
        self.pending[fc["issued_at"]] = fc
        self.history[fc["issued_at"]] = fc
        cutoff = fc["issued_at"] - 2 * HORIZON * BAR_SEC
        for k in [k for k in self.history if k < cutoff]:
            del self.history[k]

    def realize(self, t, close, log=None):
        for issued, fc in list(self.pending.items()):
            h = (t - issued) // BAR_SEC
            if h < 1:
                continue
            if h > HORIZON:
                del self.pending[issued]
                continue
            p = fc["points"][h - 1]
            s = self.stats[h]
            s["n"] += 1
            s["ae_m"] += abs(p["mean"] - close)
            s["ae_rw"] += abs(fc["base"] - close)
            s["in80"] += p["lo80"] <= close <= p["hi80"]
            dp, da = p["mean"] - fc["base"], close - fc["base"]
            if dp and da:
                s["dir_n"] += 1
                s["hit"] += (dp > 0) == (da > 0)
            if log:
                log(issued, h, fc["base"], p, close)
            if h == HORIZON:
                del self.pending[issued]

    def summary(self):
        out = []
        for h in range(1, HORIZON + 1):
            s = self.stats[h]
            n = s["n"]
            out.append({
                "h": h, "n": n,
                "mae_model": s["ae_m"] / n if n else None,
                "mae_rw": s["ae_rw"] / n if n else None,
                "hit": s["hit"] / s["dir_n"] if s["dir_n"] else None,
                "cov80": s["in80"] / n if n else None,
            })
        return out
