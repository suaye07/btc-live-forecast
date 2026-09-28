// Browser port of local/model.py + local/server.py: Coinbase trades -> 30s bars
// -> 15-minute forecast path, scored live against a "no change" baseline.
const BAR_SEC = 30, HORIZON = 30;
const Z = {50: 0.6745, 80: 1.2816, 95: 1.96};
const API = "https://api.exchange.coinbase.com/products/BTC-USD";
const BACKFILL_MIN = 90, POLL_MS = 2000;

function erf(x) {  // Abramowitz & Stegun 7.1.26
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
}
const phi = x => 0.5 * (1 + erf(x / Math.SQRT2));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const parseTs = s => Date.parse(s.replace(/(\.\d{3})\d+/, "$1")) / 1000;

function solve(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c || !M[c][c]) continue;
      const f = M[r][c] / M[c][c];
      M[r] = M[r].map((x, j) => x - f * M[c][j]);
    }
  }
  return M.map((row, i) => row[i] ? row[n] / row[i] : 0);
}

class Forecaster {
  static EWMA_LAMBDA = 0.97; static RIDGE_LAMBDA = 60; static MIN_TRAIN = 40; static TRAIN_WINDOW = 360;
  static FEATURES = ["z[t]", "z[t-1]", "mean z[t-2..t-9]", "order-flow imbalance"];
  coef = [0, 0, 0, 0]; nTrain = 0;

  sigmas(rets) {  // EWMA sigma known at the end of each bar (no lookahead)
    const head = rets.slice(0, 20);
    let v = head.reduce((a, r) => a + r * r, 0) / Math.max(1, head.length) || 1e-8;
    const out = rets.map(r => { const s = Math.sqrt(v); v = Forecaster.EWMA_LAMBDA * v + (1 - Forecaster.EWMA_LAMBDA) * r * r; return s; });
    return [out, Math.sqrt(v)];
  }

  static feat(z, ofi, t) {
    const tail = z.slice(Math.max(0, t - 9), Math.max(0, t - 1));
    return [z[t], t >= 1 ? z[t - 1] : 0, tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : 0, ofi[t]];
  }

  fitPredict(bars, candleSigma30) {
    const closes = bars.map(b => b.close);
    const rets = closes.slice(1).map((c, i) => Math.log(c / closes[i]));
    const ofi = bars.slice(1).map(b => b.ofi);
    if (rets.length < 10) return null;
    let [sigBefore, sig] = this.sigmas(rets);
    if (candleSigma30) sig = Math.sqrt(0.6 * sig ** 2 + 0.4 * candleSigma30 ** 2);
    const z = rets.map((r, i) => Math.max(-6, Math.min(6, sigBefore[i] > 0 ? r / sigBefore[i] : 0)));

    const X = [], y = [];
    for (let t = Math.max(1, z.length - 1 - Forecaster.TRAIN_WINDOW); t < z.length - 1; t++) {
      X.push(Forecaster.feat(z, ofi, t)); y.push(z[t + 1]);
    }
    this.nTrain = y.length;
    const k = Forecaster.FEATURES.length;
    if (this.nTrain >= Forecaster.MIN_TRAIN) {
      const A = [...Array(k)].map((_, i) => [...Array(k)].map((_, j) =>
        X.reduce((a, x) => a + x[i] * x[j], 0) + (i === j ? Forecaster.RIDGE_LAMBDA : 0)));
      const b = [...Array(k)].map((_, i) => X.reduce((a, x, n) => a + x[i] * y[n], 0));
      this.coef = solve(A, b);
    } else this.coef = Array(k).fill(0);

    const zz = [...z], oo = [...ofi], base = closes.at(-1), t0 = bars.at(-1).t, points = [];
    let cum = 0;
    for (let h = 1; h <= HORIZON; h++) {
      const f = Forecaster.feat(zz, oo, zz.length - 1);
      const zhat = Math.max(-0.5, Math.min(0.5, this.coef.reduce((a, c, i) => a + c * f[i], 0)));
      zz.push(zhat); oo.push(oo.at(-1) * 0.5);  // order-flow signal decays
      cum += zhat * sig;
      const sd = sig * Math.sqrt(h);
      const p = {h, t: t0 + h * BAR_SEC, mean: base * Math.exp(cum), p_up: sd > 0 ? phi(cum / sd) : 0.5};
      for (const [lvl, q] of Object.entries(Z)) {
        p["lo" + lvl] = base * Math.exp(cum - q * sd);
        p["hi" + lvl] = base * Math.exp(cum + q * sd);
      }
      points.push(p);
    }
    return {issued_at: t0, base, sigma30: sig, points};
  }
}

class Scorekeeper {
  pending = new Map(); history = new Map();
  stats = [...Array(HORIZON + 1)].map(() => ({n: 0, aeM: 0, aeRw: 0, hit: 0, dirN: 0, in80: 0}));

  add(fc) {
    this.pending.set(fc.issued_at, fc); this.history.set(fc.issued_at, fc);
    for (const k of this.history.keys()) if (k < fc.issued_at - 2 * HORIZON * BAR_SEC) this.history.delete(k);
  }

  realize(t, close) {
    for (const [issued, fc] of this.pending) {
      const h = Math.floor((t - issued) / BAR_SEC);
      if (h < 1) continue;
      if (h > HORIZON) { this.pending.delete(issued); continue; }
      const p = fc.points[h - 1], s = this.stats[h];
      s.n++; s.aeM += Math.abs(p.mean - close); s.aeRw += Math.abs(fc.base - close);
      s.in80 += p.lo80 <= close && close <= p.hi80;
      const dp = p.mean - fc.base, da = close - fc.base;
      if (dp && da) { s.dirN++; s.hit += (dp > 0) === (da > 0); }
      if (h === HORIZON) this.pending.delete(issued);
    }
  }

  summary() {
    return this.stats.slice(1).map((s, i) => ({
      h: i + 1, n: s.n,
      mae_model: s.n ? s.aeM / s.n : null, mae_rw: s.n ? s.aeRw / s.n : null,
      hit: s.dirN ? s.hit / s.dirN : null, cov80: s.n ? s.in80 / s.n : null,
    }));
  }
}

class Engine {
  bars = new Map(); lastId = 0; lastPrice = null; lastTradeTs = null;
  forecaster = new Forecaster(); score = new Scorekeeper();
  forecast = null; candleSigma30 = null; status = "starting"; error = null; closedUpto = null;

  async get(path) {
    const r = await fetch(API + path, {cache: "no-store"});
    if (!r.ok) throw new Error(`Coinbase HTTP ${r.status}`);
    return {data: await r.json(), after: r.headers.get("cb-after")};
  }

  ingest(trades) {
    trades.sort((a, b) => a.trade_id - b.trade_id);
    for (const tr of trades) {
      if (tr.trade_id <= this.lastId) continue;
      const ts = parseTs(tr.time), px = +tr.price, sz = +tr.size, key = Math.floor(ts / BAR_SEC) * BAR_SEC;
      let b = this.bars.get(key);
      if (!b) this.bars.set(key, b = {buy: 0, sell: 0, close: px, lastTs: 0});
      b[tr.side === "sell" ? "buy" : "sell"] += sz;  // Coinbase "side" is the maker side
      if (ts >= b.lastTs) { b.close = px; b.lastTs = ts; }
      if (this.lastTradeTs == null || ts >= this.lastTradeTs) { this.lastPrice = px; this.lastTradeTs = ts; }
    }
    if (trades.length) this.lastId = Math.max(this.lastId, trades.at(-1).trade_id);
  }

  async backfill() {
    const stop = Date.now() / 1000 - BACKFILL_MIN * 60;
    let trades = [], after = null;
    for (let i = 0; i < 60; i++) {
      const {data, after: next} = await this.get("/trades?limit=1000" + (after ? `&after=${after}` : ""));
      if (!data.length) break;
      trades = trades.concat(data);
      const oldest = parseTs(data.at(-1).time);
      this.status = `loading history ${Math.min(100, Math.round((1 - (oldest - stop) / (BACKFILL_MIN * 60)) * 100))}%`;
      after = next;
      if (oldest < stop || !after) break;
      await sleep(200);
    }
    this.ingest(trades);
    await this.refreshCandleVol();
  }

  async refreshCandleVol() {
    try {
      const {data} = await this.get("/candles?granularity=60");
      const closes = data.sort((a, b) => a[0] - b[0]).map(c => c[4]).slice(-180);
      const r = closes.slice(1).map((c, i) => Math.log(c / closes[i]));
      const m = r.reduce((a, b) => a + b, 0) / r.length;
      this.candleSigma30 = Math.sqrt(r.reduce((a, x) => a + (x - m) ** 2, 0) / (r.length - 1)) / Math.SQRT2;
    } catch (e) { this.error = "candles: " + e.message; }
  }

  async poll() {  // newest trades, paging back until we overlap what we have
    let {data, after} = await this.get("/trades?limit=1000");
    let got = [...data];
    for (let i = 0; i < 5 && data.length && data.at(-1).trade_id > this.lastId && after; i++) {
      ({data, after} = await this.get(`/trades?limit=1000&after=${after}`));
      got = got.concat(data);
    }
    this.ingest(got);
  }

  closedBars(upto) {  // contiguous closed bars ending at bar start `upto`, gaps forward-filled
    const keys = [...this.bars.keys()].filter(k => k <= upto);
    if (!keys.length) return [];
    const out = []; let prev = null;
    for (let t = Math.min(...keys); t <= upto; t += BAR_SEC) {
      const b = this.bars.get(t); let ofi = 0;
      if (b) { prev = b.close; const tot = b.buy + b.sell; ofi = tot ? (b.buy - b.sell) / tot : 0; }
      out.push({t: t + BAR_SEC, close: prev, ofi});  // t = bar close time
    }
    return out;
  }

  onBarClose(barStart) {
    const bars = this.closedBars(barStart);
    if (bars.length < 12) return;
    const tClose = bars.at(-1).t;
    this.score.realize(tClose, bars.at(-1).close);
    const fc = this.forecaster.fitPredict(bars, this.candleSigma30);
    if (fc) { this.forecast = fc; this.score.add(fc); this.status = "live"; }
    for (const k of this.bars.keys()) if (k < tClose - 4 * 3600) this.bars.delete(k);
  }

  lastClosedBar() { return Math.floor((Date.now() / 1000 - 2) / BAR_SEC) * BAR_SEC - BAR_SEC; }

  async run() {
    for (;;) {
      try {
        if (this.closedUpto == null) {
          await this.backfill();
          this.closedUpto = this.lastClosedBar();
          const bars = this.closedBars(this.closedUpto);  // replay history to warm up the scorecard
          if (bars.length) for (let s = bars[0].t - BAR_SEC + 12 * BAR_SEC; s <= this.closedUpto; s += BAR_SEC) this.onBarClose(s);
        }
        await this.poll();
        const now = this.lastClosedBar();
        while (this.closedUpto < now) { this.closedUpto += BAR_SEC; this.onBarClose(this.closedUpto); }
        if (Date.now() - (this.candleAt || 0) > 300000) { this.candleAt = Date.now(); await this.refreshCandleVol(); }
        this.error = null;
      } catch (e) { this.error = e.message; }
      await sleep(POLL_MS);
    }
  }

  state() {
    const fc = this.forecast, past = fc && this.score.history.get(fc.issued_at - HORIZON * BAR_SEC);
    return {
      server_time: Date.now() / 1000, status: this.status, error: this.error,
      last_price: this.lastPrice, last_trade_ts: this.lastTradeTs,
      bars: this.closedUpto == null ? [] : this.closedBars(this.closedUpto).slice(-60),
      forecast: fc, past_forecast: past || null, score: this.score.summary(),
    };
  }
}
