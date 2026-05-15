# SentimentTradingView

Interactive Monte Carlo simulation portfolio dashboard for TradingView-starred stocks, with X sentiment analysis and strategy signals (BUY / SELL / UNSURE).

## Features

- **Monte Carlo simulation** — Geometric Brownian Motion engine trained on live-calibrated drift/volatility, configurable paths (100 / 1k / 10k) and horizons (5d / 30d / 90d / 252d).
- **Starred TradingView watchlist** — mirrors your TradingView starred list (AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, JPM, V, SPY).
- **Live price streaming** — simulated tick updates every 3s with deterministic seed for reproducibility.
- **Strategy signals** — BUY / SELL / UNSURE derived from RSI + MA crossover + MC upside/downside asymmetry.
- **X sentiment panel** — positive / neutral / negative breakdown plus sample posts per ticker.
- **Portfolio KPIs** — value, P&L, expected return, 95% VaR, Sharpe ratio, max drawdown.
- **Starred-stocks summary** — narrative overview of the entire watchlist with aggregate signal.
- **Accessible by default** — WCAG 2.2 AA: keyboard nav, ARIA live regions, screen-reader chart alternatives, 4.5:1 contrast in dark mode.

## Running

No build step. Live at https://pablowilliams.github.io/SentimentTradingView/.

For local development, any static server works:

```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Data pipeline

The dashboard works on plain GitHub Pages with no backend and no third-party CORS proxy. A scheduled GitHub Action (`.github/workflows/refresh-data.yml`) fetches Yahoo Finance v8 for the watchlist every 20 minutes during US trading hours and commits the result to `data/quotes.json`. The dashboard reads that file with a same-origin `fetch` on load and on each tick, so there is nothing in the data path that can fail because of CORS, rate limits, or proxy outages.

The watchlist is defined in two places: `STARRED` at the top of `app.js` for client-side display, and the `TICKERS` env in the refresh-data workflow for the cron fetch. Keep them in sync.

If the same-origin snapshot is missing or empty (for example during local development before the cron has ever run), the dashboard falls back to a CORS proxy chain, then to fully mocked random walks. The connection strip in the bottom toolbar shows which path is active.

## Architecture

- `index.html` — semantic layout, landmarks, headings
- `app.js` — data, Monte Carlo engine, sentiment mocking, signal logic, rendering
- `styles.css` — dark financial theme with AA-contrast tokens

## Data sources

All data in this build is **mocked** for offline/demo use. To go live, wire in:
- Prices: Polygon / Alpaca / Finnhub REST + WebSocket
- TradingView starred watchlist: TradingView API or scrape of authenticated session
- Sentiment: X API v2 (Basic tier) + VADER or fine-tuned FinBERT
- MC calibration: historical returns from your price feed, EWMA vol

See `app.js` for the integration seams (functions prefixed `fetch*` are the swap points).

## License

Private. All rights reserved.
