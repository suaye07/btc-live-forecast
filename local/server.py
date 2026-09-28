"""Live 15-minute BTC-USD forecast, refreshed every 30 seconds.

Run:  python3 server.py   then open http://localhost:8765
No third-party packages needed. Data: Coinbase Exchange public API.
"""
import csv
import json
import math
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from model import BAR_SEC, HORIZON, Forecaster, Scorekeeper

API = "https://api.exchange.coinbase.com/products/BTC-USD"
PORT = int(os.environ.get("PORT", 8765))
BACKFILL_MIN = 90
POLL_SEC = 2
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "predictions_log.csv")


def get(path, params=""):
    req = urllib.request.Request(f"{API}{path}{params}", headers={"User-Agent": "btc-live/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r), r.headers


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.bars = {}            # bar start ts -> {buy, sell, close}
        self.last_id = 0
        self.last_price = None
        self.last_trade_ts = None
        self.forecaster = Forecaster()
        self.score = Scorekeeper()
        self.forecast = None
        self.candle_sigma30 = None
        self.status = "starting"
        self.error = None
        self.closed_upto = None   # start ts of last bar treated as closed

    # ---- data ------------------------------------------------------
    def _ingest(self, trades):
        for tr in trades:
            tid = tr["trade_id"]
            if tid <= self.last_id:
                continue
            ts, px, sz = parse_ts(tr["time"]), float(tr["price"]), float(tr["size"])
            b = self.bars.setdefault(int(ts // BAR_SEC) * BAR_SEC, {"buy": 0.0, "sell": 0.0, "close": px, "last_ts": 0})
            # Coinbase "side" is the maker side: maker sell == taker buy
            b["buy" if tr["side"] == "sell" else "sell"] += sz
            if ts >= b["last_ts"]:
                b["close"], b["last_ts"] = px, ts
            if self.last_trade_ts is None or ts >= self.last_trade_ts:
                self.last_price, self.last_trade_ts = px, ts
        if trades:
            self.last_id = max(self.last_id, max(t["trade_id"] for t in trades))

    def backfill(self):
        self.status = "backfilling trades"
        stop = time.time() - BACKFILL_MIN * 60
        trades, after = [], None
        for _ in range(60):
            page, hdr = get("/trades", "?limit=1000" + (f"&after={after}" if after else ""))
            if not page:
                break
            trades += page
            after = hdr.get("cb-after")
            if parse_ts(page[-1]["time"]) < stop or not after:
                break
            time.sleep(0.2)
        with self.lock:
            self._ingest(sorted(trades, key=lambda t: t["trade_id"]))
        self.refresh_candle_vol()

    def refresh_candle_vol(self):
        try:
            candles, _ = get("/candles", "?granularity=60")
            closes = [c[4] for c in sorted(candles)][-180:]
            r = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
            m = sum(r) / len(r)
            self.candle_sigma30 = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1)) / math.sqrt(2)
        except Exception as e:
            self.error = f"candles: {e}"

    def poll(self):
        """Fetch newest trades, paging back until we overlap what we have."""
        page, hdr = get("/trades", "?limit=1000")
        got = list(page)
        for _ in range(5):
            if not page or page[-1]["trade_id"] <= self.last_id or not hdr.get("cb-after"):
                break
            page, hdr = get("/trades", f"?limit=1000&after={hdr['cb-after']}")
            got += page
        with self.lock:
            self._ingest(sorted(got, key=lambda t: t["trade_id"]))

    # ---- bars & forecasting -----------------------------------------
    def closed_bars(self, upto):
        """Contiguous closed bars ending at bar start `upto`, gaps forward-filled."""
        keys = sorted(k for k in self.bars if k <= upto)
        if not keys:
            return []
        out, prev = [], None
        for t in range(keys[0], upto + 1, BAR_SEC):
            b = self.bars.get(t)
            if b:
                prev = b["close"]
                tot = b["buy"] + b["sell"]
                ofi = (b["buy"] - b["sell"]) / tot if tot else 0.0
            else:
                ofi = 0.0
            out.append({"t": t + BAR_SEC, "close": prev, "ofi": ofi})  # t = bar close time
        return out

    def on_bar_close(self, bar_start):
        bars = self.closed_bars(bar_start)
        if len(bars) < 12:
            return
        t_close = bars[-1]["t"]
        self.score.realize(t_close, bars[-1]["close"], log=self.log_row)
        fc = self.forecaster.fit_predict(bars, self.candle_sigma30)
        if fc:
            fc["coef"] = dict(zip(self.forecaster.FEATURES, self.forecaster.coef))
            fc["n_train"] = self.forecaster.n_train
            self.forecast = fc
            self.score.add(fc)
            self.status = "live"
        cutoff = t_close - 4 * 3600
        for k in [k for k in self.bars if k < cutoff]:
            del self.bars[k]

    def log_row(self, issued, h, base, p, actual):
        new = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["issued_utc", "horizon_s", "base", "pred", "lo80", "hi80", "actual"])
            w.writerow([datetime.fromtimestamp(issued, timezone.utc).isoformat(), h * BAR_SEC,
                        round(base, 2), round(p["mean"], 2), round(p["lo80"], 2),
                        round(p["hi80"], 2), round(actual, 2)])

    def run(self):
        while True:
            try:
                if self.closed_upto is None:
                    self.backfill()
                    # most recent fully closed bar (allow 2s for late trades)
                    self.closed_upto = int((time.time() - 2) // BAR_SEC) * BAR_SEC - BAR_SEC
                    with self.lock:
                        # warm up: score & forecast through history silently
                        bars = self.closed_bars(self.closed_upto)
                        if bars:
                            first = bars[0]["t"] - BAR_SEC
                            for s in range(first + 12 * BAR_SEC, self.closed_upto + 1, BAR_SEC):
                                self.on_bar_close(s)
                self.poll()
                now_closed = int((time.time() - 2) // BAR_SEC) * BAR_SEC - BAR_SEC
                with self.lock:
                    while self.closed_upto < now_closed:
                        self.closed_upto += BAR_SEC
                        self.on_bar_close(self.closed_upto)
                if int(time.time()) % 300 < POLL_SEC:
                    self.refresh_candle_vol()
                self.error = None
            except Exception as e:
                self.error = str(e)
            time.sleep(POLL_SEC)

    def state(self):
        with self.lock:
            upto = self.closed_upto
            bars = self.closed_bars(upto)[-60:] if upto else []
            fc = self.forecast
            past = None
            if fc:
                past = self.score.history.get(fc["issued_at"] - HORIZON * BAR_SEC)
            return {
                "server_time": time.time(),
                "status": self.status, "error": self.error,
                "last_price": self.last_price, "last_trade_ts": self.last_trade_ts,
                "bars": [{"t": b["t"], "close": b["close"]} for b in bars],
                "forecast": fc,
                "past_forecast": past and {"issued_at": past["issued_at"], "base": past["base"],
                                           "points": [{"t": p["t"], "mean": p["mean"]} for p in past["points"]]},
                "score": self.score.summary(),
                "bar_sec": BAR_SEC, "horizon": HORIZON,
            }


ENGINE = Engine()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/state"):
            body, ctype = json.dumps(ENGINE.state()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body, ctype = f.read(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=ENGINE.run, daemon=True).start()
    print(f"BTC live forecast → http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
