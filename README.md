# BTC 15-minute live forecast

Every 30 seconds this page predicts the BTC-USD price at each 30-second step over the next 15 minutes (30 points). It shows 50%, 80% and 95% uncertainty bands and scores itself live against a "price doesn't change" baseline.

It runs entirely in the browser: open the GitHub Pages link, or open `index.html` locally. There's no server and nothing to install. Data comes from the public Coinbase Exchange API.

- **Data:** it loads the last 90 minutes of trades into 30s bars, then polls new trades every 2s.
- **Model:** it takes 30s log returns, scales them by EWMA volatility, and fits a heavily shrunk ridge regression on the last two returns, a longer-run return average and the order-flow imbalance (taker buy vs sell volume). The regression is run forward 30 steps. The bands are sigma × √horizon, where sigma blends fast (EWMA 30s) and slow (1-minute candles) volatility.
- **Scorecard:** every forecast is matched to the realized closes. Skill > 0 means it beats the "no change" baseline, and the 80% coverage should come out close to 80%.

At this horizon BTC is close to a random walk. This is an experiment, not financial advice.

## Files

- `index.html` is the dashboard, and `engine.js` holds the data feed, model and scoring.
- `local/` has the original Python version (`python3 local/server.py`, then open http://localhost:8765). It does the same thing and also appends every scored prediction to `predictions_log.csv`.
