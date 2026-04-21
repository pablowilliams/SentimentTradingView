"use strict";

/* =========================================================================
   SentimentTradingView — Monte Carlo dashboard
   =========================================================================
   All data is mocked for the demo. Integration seams:
     - fetchPrices()        -> Polygon / Alpaca / Finnhub REST + WS
     - fetchStarredList()   -> TradingView watchlist API
     - fetchSentiment()     -> X API v2 + VADER/FinBERT
   ========================================================================= */

// ========== Starred TradingView watchlist ==========
// mu = annualized drift (decimal), sigma = annualized volatility (decimal)
const STARRED = [
  { ticker: "AAPL",  name: "Apple Inc.",               sector: "Technology",            price: 184.32, mu: 0.14, sigma: 0.26 },
  { ticker: "MSFT",  name: "Microsoft Corp.",          sector: "Technology",            price: 418.75, mu: 0.17, sigma: 0.23 },
  { ticker: "NVDA",  name: "NVIDIA Corp.",             sector: "Semiconductors",        price: 892.14, mu: 0.38, sigma: 0.52 },
  { ticker: "TSLA",  name: "Tesla Inc.",               sector: "Auto / EV",             price: 241.58, mu: 0.09, sigma: 0.58 },
  { ticker: "GOOGL", name: "Alphabet Inc. (Class A)",  sector: "Internet / Ads",        price: 168.90, mu: 0.15, sigma: 0.28 },
  { ticker: "AMZN",  name: "Amazon.com Inc.",          sector: "E-commerce / Cloud",    price: 186.43, mu: 0.16, sigma: 0.31 },
  { ticker: "META",  name: "Meta Platforms Inc.",      sector: "Social / Ads",          price: 506.21, mu: 0.22, sigma: 0.36 },
  { ticker: "JPM",   name: "JPMorgan Chase & Co.",     sector: "Financials",            price: 215.47, mu: 0.08, sigma: 0.22 },
  { ticker: "V",     name: "Visa Inc.",                sector: "Payments",              price: 276.84, mu: 0.11, sigma: 0.20 },
  { ticker: "SPY",   name: "SPDR S&P 500 ETF Trust",   sector: "Broad-market ETF",      price: 528.15, mu: 0.08, sigma: 0.16 },
];

// Pre-compute a synthetic 60-day price history per ticker (for RSI / MAs).
function generateHistory(stock, seed) {
  const rng = makeRng(seed + hashString(stock.ticker));
  const days = 60;
  const dt = 1 / 252;
  const history = new Array(days);
  let S = stock.price / Math.exp((stock.mu - 0.5 * stock.sigma ** 2) * (days * dt));
  for (let i = 0; i < days; i++) {
    const z = gaussian(rng);
    S = S * Math.exp((stock.mu - 0.5 * stock.sigma ** 2) * dt + stock.sigma * Math.sqrt(dt) * z);
    history[i] = S;
  }
  // Anchor end to current price
  const scale = stock.price / history[days - 1];
  for (let i = 0; i < days; i++) history[i] *= scale;
  return history;
}

function hashString(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return h;
}

// ========== Seeded RNG (Mulberry32) ==========
function makeRng(seed) {
  let a = (seed | 0) || 1;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box–Muller standard normal
function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ========== Technical indicators ==========
function sma(arr, period) {
  if (arr.length < period) return null;
  let s = 0;
  for (let i = arr.length - period; i < arr.length; i++) s += arr[i];
  return s / period;
}

function rsi(arr, period = 14) {
  if (arr.length < period + 1) return 50;
  let gains = 0, losses = 0;
  for (let i = arr.length - period; i < arr.length; i++) {
    const d = arr[i] - arr[i - 1];
    if (d > 0) gains += d; else losses -= d;
  }
  const avgG = gains / period;
  const avgL = losses / period;
  if (avgL === 0) return 100;
  const rs = avgG / avgL;
  return 100 - 100 / (1 + rs);
}

// ========== Monte Carlo ==========
function monteCarlo({ S0, mu, sigma, days, nPaths, seed }) {
  const rng = makeRng(seed);
  const dt = 1 / 252;
  const sampleN = Math.min(40, nPaths); // only return 40 sample paths for drawing
  const samplePaths = [];
  const finalPrices = new Float64Array(nPaths);
  // For median + percentile bands, track all paths as column-major arrays
  const pxByDay = new Array(days + 1);
  for (let d = 0; d <= days; d++) pxByDay[d] = new Float64Array(nPaths);
  for (let d = 0; d <= days; d++) pxByDay[d][0] = S0; // placeholder

  const drift = (mu - 0.5 * sigma * sigma) * dt;
  const diff = sigma * Math.sqrt(dt);

  for (let p = 0; p < nPaths; p++) {
    let s = S0;
    pxByDay[0][p] = s;
    const path = p < sampleN ? new Array(days + 1) : null;
    if (path) path[0] = s;
    for (let d = 1; d <= days; d++) {
      const z = gaussian(rng);
      s = s * Math.exp(drift + diff * z);
      pxByDay[d][p] = s;
      if (path) path[d] = s;
    }
    finalPrices[p] = s;
    if (path) samplePaths.push(path);
  }

  // Percentile bands per day
  const percentiles = computeBands(pxByDay, days);
  const summary = summarizeFinals(finalPrices, S0);

  return { samplePaths, percentiles, summary, days, S0, nPaths };
}

function computeBands(pxByDay, days) {
  const p05 = new Array(days + 1);
  const p50 = new Array(days + 1);
  const p95 = new Array(days + 1);
  for (let d = 0; d <= days; d++) {
    const arr = Array.from(pxByDay[d]).sort((a, b) => a - b);
    const n = arr.length;
    p05[d] = arr[Math.floor(0.05 * (n - 1))];
    p50[d] = arr[Math.floor(0.50 * (n - 1))];
    p95[d] = arr[Math.floor(0.95 * (n - 1))];
  }
  return { p05, p50, p95 };
}

function summarizeFinals(finals, S0) {
  const sorted = Array.from(finals).sort((a, b) => a - b);
  const n = sorted.length;
  const q = (x) => sorted[Math.max(0, Math.min(n - 1, Math.floor(x * (n - 1))))];
  const mean = sorted.reduce((a, b) => a + b, 0) / n;
  const median = q(0.5);
  const p05 = q(0.05);
  const p95 = q(0.95);
  return {
    expectedReturn: (mean - S0) / S0,
    medianReturn: (median - S0) / S0,
    ci95Low: (p05 - S0) / S0,
    ci95High: (p95 - S0) / S0,
    meanPrice: mean,
    medianPrice: median,
    probUp: sorted.filter((x) => x > S0).length / n,
  };
}

// ========== Sentiment (mock X API) ==========
const SENTIMENT_POSTS = {
  AAPL: [
    { sent: "pos", author: "@techinvestor", handle: "6h", text: "Apple's services revenue growth is flying under the radar. Long $AAPL through earnings." },
    { sent: "pos", author: "@fundmanager_jk", handle: "1d", text: "Buyback pace is unreal. $AAPL still my largest position." },
    { sent: "neu", author: "@mkt_watch", handle: "2h", text: "$AAPL iPhone China shipments flat YoY. Waiting for Vision Pro data." },
    { sent: "neg", author: "@shortstories", handle: "9h", text: "Multiple compression coming for $AAPL. No growth justifies 29x fwd." },
  ],
  MSFT: [
    { sent: "pos", author: "@ai_thesis", handle: "3h", text: "$MSFT + OpenAI datacenter buildout is the most durable AI trade." },
    { sent: "pos", author: "@cloud_analyst", handle: "8h", text: "Azure growth reaccelerating. Copilot monetization underpriced in $MSFT." },
    { sent: "neu", author: "@longvolpete", handle: "1d", text: "$MSFT expensive vs rest of megacap. Fairly priced at best." },
  ],
  NVDA: [
    { sent: "pos", author: "@chip_bull", handle: "1h", text: "Blackwell ramp is happening faster than consensus. $NVDA to new highs." },
    { sent: "pos", author: "@datacenter_pm", handle: "4h", text: "Hyperscaler capex up across the board. $NVDA is the shovel." },
    { sent: "neg", author: "@contrarian_f", handle: "7h", text: "$NVDA pricing peak. Custom ASICs taking share in 18 months." },
    { sent: "neu", author: "@quantdaily", handle: "12h", text: "$NVDA options vol is insane. Skew flat — market agnostic short-term." },
  ],
  TSLA: [
    { sent: "neg", author: "@ev_skeptic", handle: "2h", text: "$TSLA margins compressing again. FSD thesis keeps getting delayed." },
    { sent: "neu", author: "@auto_news", handle: "5h", text: "$TSLA China registrations mixed. Discount wars continue." },
    { sent: "pos", author: "@musk_follower", handle: "1d", text: "Robotaxi day will reprice $TSLA. Long and strong." },
    { sent: "neg", author: "@valuationvic", handle: "10h", text: "$TSLA still priced as an AI company, delivers like an automaker." },
  ],
  GOOGL: [
    { sent: "pos", author: "@search_analyst", handle: "3h", text: "Gemini ramp stabilizing Search share. $GOOGL cheap vs MSFT." },
    { sent: "neu", author: "@adtech_vc", handle: "6h", text: "$GOOGL YouTube shorts monetization improving. DoJ overhang still there." },
    { sent: "pos", author: "@cloud_watcher", handle: "11h", text: "GCP growing fastest of the three. $GOOGL re-rate incoming." },
  ],
  AMZN: [
    { sent: "pos", author: "@retail_pm", handle: "2h", text: "$AMZN operating margins nowhere near peak. AWS reaccelerating." },
    { sent: "pos", author: "@logistics_nerd", handle: "7h", text: "$AMZN fulfillment regionalization paying off. Cost per unit down 8% YoY." },
    { sent: "neu", author: "@cloud_watcher", handle: "1d", text: "$AMZN AWS gaining share but Azure narrowing gap. Close call." },
  ],
  META: [
    { sent: "pos", author: "@ads_quant", handle: "4h", text: "$META ad platform AI gains still ramping. CPM + click volume both up." },
    { sent: "neu", author: "@metaverse_sceptic", handle: "9h", text: "Reality Labs burn still huge. $META ex-RL is a 20x PE story." },
    { sent: "pos", author: "@fundmanager_jk", handle: "1d", text: "$META buyback + growth combo hard to beat in megacap." },
  ],
  JPM: [
    { sent: "neu", author: "@bank_credit", handle: "3h", text: "$JPM deposit costs stabilizing. NII flat into next quarter." },
    { sent: "pos", author: "@ib_analyst", handle: "7h", text: "Investment banking pipeline strongest since 2021. $JPM to benefit." },
    { sent: "neg", author: "@macro_bear", handle: "1d", text: "Credit losses still bottoming. $JPM reserves look thin." },
  ],
  V: [
    { sent: "pos", author: "@payments_vc", handle: "5h", text: "$V cross-border volume +16% YoY. Travel recovery persists." },
    { sent: "pos", author: "@compounders", handle: "12h", text: "$V is the quintessential toll road. Adding on any weakness." },
    { sent: "neu", author: "@reg_watch", handle: "1d", text: "$V interchange regulatory risk always present but priced in." },
  ],
  SPY: [
    { sent: "pos", author: "@macro_daily", handle: "2h", text: "Breadth finally improving. $SPY supported through 5,200." },
    { sent: "neu", author: "@vol_pm", handle: "6h", text: "$SPY realized vol low, VIX calm. Insurance cheap if hedging." },
    { sent: "neg", author: "@contrarian_f", handle: "10h", text: "$SPY top-5 concentration at records. Mean revert eventually." },
  ],
};

function getSentiment(ticker) {
  const posts = SENTIMENT_POSTS[ticker] || [];
  const pos = posts.filter((p) => p.sent === "pos").length;
  const neu = posts.filter((p) => p.sent === "neu").length;
  const neg = posts.filter((p) => p.sent === "neg").length;
  const total = posts.length || 1;
  // boost with some deterministic jitter so tickers without posts still differ
  const h = hashString(ticker);
  const jitter = ((h % 11) - 5) * 0.01;
  const posPct = Math.max(0, Math.min(1, pos / total + 0.1 + jitter));
  const negPct = Math.max(0, Math.min(1, neg / total + 0.05 - jitter));
  let neuPct = Math.max(0, 1 - posPct - negPct);
  // Re-normalize to ensure sum to 1
  const sum = posPct + neuPct + negPct;
  return {
    positive: posPct / sum,
    neutral: neuPct / sum,
    negative: negPct / sum,
    posts,
    score: posPct / sum - negPct / sum, // net in [-1, +1]
  };
}

// ========== Strategies ==========
const STRATEGIES = [
  {
    id: "ma_cross",
    name: "MA Cross",
    desc: "20/50-day",
    apply: (ctx) => {
      const s20 = sma(ctx.history, 20);
      const s50 = sma(ctx.history, 50);
      if (s20 == null || s50 == null) return { signal: "UNSURE", detail: "Not enough history for 50-day MA." };
      if (s20 > s50 * 1.005) return { signal: "BUY",  detail: `20-day (${s20.toFixed(2)}) above 50-day (${s50.toFixed(2)}) — bullish trend.` };
      if (s20 < s50 * 0.995) return { signal: "SELL", detail: `20-day (${s20.toFixed(2)}) below 50-day (${s50.toFixed(2)}) — bearish trend.` };
      return { signal: "UNSURE", detail: `20-day ≈ 50-day — consolidation.` };
    },
  },
  {
    id: "rsi_mr",
    name: "RSI Mean Reversion",
    desc: "14-period",
    apply: (ctx) => {
      const r = rsi(ctx.history, 14);
      if (r < 30) return { signal: "BUY",  detail: `RSI ${r.toFixed(1)} — oversold, mean-revert up.` };
      if (r > 70) return { signal: "SELL", detail: `RSI ${r.toFixed(1)} — overbought, mean-revert down.` };
      return { signal: "UNSURE", detail: `RSI ${r.toFixed(1)} — mid-range, no edge.` };
    },
  },
  {
    id: "momentum",
    name: "Momentum",
    desc: "20-day return",
    apply: (ctx) => {
      const h = ctx.history;
      if (h.length < 21) return { signal: "UNSURE", detail: "Not enough data." };
      const r20 = (h[h.length - 1] - h[h.length - 21]) / h[h.length - 21];
      if (r20 > 0.05)  return { signal: "BUY",  detail: `+${(r20 * 100).toFixed(1)}% past 20d — ride the trend.` };
      if (r20 < -0.05) return { signal: "SELL", detail: `${(r20 * 100).toFixed(1)}% past 20d — negative momo.` };
      return { signal: "UNSURE", detail: `${(r20 * 100).toFixed(1)}% past 20d — flat.` };
    },
  },
  {
    id: "mc_asymmetry",
    name: "MC Asymmetry",
    desc: "MC up/down skew",
    apply: (ctx) => {
      if (!ctx.mcSummary) return { signal: "UNSURE", detail: "Run a simulation." };
      const { ci95High, ci95Low, expectedReturn, probUp } = ctx.mcSummary;
      const upside = ci95High;
      const downside = -ci95Low;
      const ratio = upside / Math.max(downside, 0.0001);
      if (ratio > 1.3 && expectedReturn > 0.01) return { signal: "BUY", detail: `Up/down ratio ${ratio.toFixed(2)}x, E[R] ${(expectedReturn * 100).toFixed(1)}%, Pr(up) ${(probUp * 100).toFixed(0)}%.` };
      if (ratio < 0.8 && expectedReturn < -0.01) return { signal: "SELL", detail: `Up/down ratio ${ratio.toFixed(2)}x, E[R] ${(expectedReturn * 100).toFixed(1)}%, Pr(up) ${(probUp * 100).toFixed(0)}%.` };
      return { signal: "UNSURE", detail: `Up/down ratio ${ratio.toFixed(2)}x — no clear edge.` };
    },
  },
  {
    id: "x_sentiment",
    name: "X Sentiment",
    desc: "net score",
    apply: (ctx) => {
      const s = ctx.sentiment;
      if (s.score > 0.25)  return { signal: "BUY",  detail: `Net +${(s.score * 100).toFixed(0)} — crowd bullish.` };
      if (s.score < -0.15) return { signal: "SELL", detail: `Net ${(s.score * 100).toFixed(0)} — crowd bearish.` };
      return { signal: "UNSURE", detail: `Net ${(s.score * 100).toFixed(0)} — mixed crowd.` };
    },
  },
];

function signalScore(signal) { return signal === "BUY" ? 1 : signal === "SELL" ? -1 : 0; }

function combineSignals(results) {
  if (!results.length) return { signal: "UNSURE", detail: "Select at least one strategy." };
  const avg = results.reduce((a, r) => a + signalScore(r.signal), 0) / results.length;
  if (avg > 0.35)  return { signal: "BUY",    detail: "Majority bullish." };
  if (avg < -0.35) return { signal: "SELL",   detail: "Majority bearish." };
  return { signal: "UNSURE", detail: "Mixed strategy signals." };
}

// ========== State ==========
const state = {
  selectedTicker: "AAPL",
  selectedStrategies: new Set(["ma_cross", "mc_asymmetry", "x_sentiment"]),
  sims: 1000,
  horizon: 30,
  seed: 42,
  tableSort: { key: "ticker", dir: "asc" },
  stocks: null,     // enriched runtime stocks
  portfolio: null,
  mcResult: null,
  prevPrices: {},
  prevKpi: {},
  prevSignals: {},
  prevAggregateSignal: null,
};

// ========== Enrichment ==========
function buildStocksRuntime() {
  const stocks = STARRED.map((s) => {
    const history = generateHistory(s, state.seed);
    const prevClose = history[history.length - 2];
    const price = history[history.length - 1];
    const change = (price - prevClose) / prevClose;
    const sentiment = getSentiment(s.ticker);
    // Quick MC summary for table (fewer sims, fixed 30d)
    const mc = monteCarlo({ S0: price, mu: s.mu, sigma: s.sigma, days: 30, nPaths: 500, seed: state.seed + hashString(s.ticker) });
    return {
      ...s,
      price,
      prevClose,
      change,
      history,
      sentiment,
      mcSummary: mc.summary,
    };
  });
  return stocks;
}

function computePortfolioKpis(stocks) {
  const w = 1 / stocks.length;
  const pv = stocks.reduce((a, s) => a + s.price * 10, 0); // pretend 10 shares each
  const prevPv = stocks.reduce((a, s) => a + s.prevClose * 10, 0);
  const pnl = pv - prevPv;
  const pnlPct = pnl / prevPv;

  const expReturn = stocks.reduce((a, s) => a + w * s.mcSummary.expectedReturn, 0);
  const var95 = stocks.reduce((a, s) => a + w * s.mcSummary.ci95Low, 0); // weighted downside
  const avgSigma = stocks.reduce((a, s) => a + w * s.sigma, 0);
  const avgMu = stocks.reduce((a, s) => a + w * s.mu, 0);
  const sharpe = (avgMu - 0.045) / Math.max(avgSigma, 0.01); // rf=4.5%

  // weighted sentiment
  const sentScore = stocks.reduce((a, s) => a + w * s.sentiment.score, 0);

  return {
    value: pv,
    pnl,
    pnlPct,
    expectedReturn: expReturn,
    var95,
    sharpe,
    sentimentScore: sentScore,
  };
}

// ========== Rendering ==========
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const prefersReducedMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Count-up animation: interpolates numeric value through a formatter
function animateNumber(el, from, to, formatter, duration = 600) {
  if (!el) return;
  if (from === to || !isFinite(from) || !isFinite(to) || prefersReducedMotion()) {
    el.textContent = formatter(to);
    return;
  }
  const start = performance.now();
  const ease = (t) => 1 - Math.pow(1 - t, 4); // easeOutQuart
  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const v = from + (to - from) * ease(t);
    el.textContent = formatter(v);
    if (t < 1) requestAnimationFrame(frame);
    else {
      el.classList.remove("landing");
      void el.offsetWidth;
      el.classList.add("landing");
    }
  }
  requestAnimationFrame(frame);
}

function renderStrategyOptions() {
  const el = $("#strategy-options");
  el.innerHTML = STRATEGIES.map((s) => `
    <label class="strategy-chip">
      <input type="checkbox" value="${s.id}" ${state.selectedStrategies.has(s.id) ? "checked" : ""} />
      <span>${s.name}</span>
      <span class="desc">${s.desc}</span>
    </label>
  `).join("");
  el.querySelectorAll("input[type=checkbox]").forEach((input) => {
    input.addEventListener("change", (e) => {
      const id = e.target.value;
      if (e.target.checked) state.selectedStrategies.add(id);
      else state.selectedStrategies.delete(id);
      renderAll();
      announce(`Strategy ${id} ${e.target.checked ? "enabled" : "disabled"}.`);
    });
  });
}

function renderTickerSelect() {
  const el = $("#ticker-select");
  el.innerHTML = state.stocks.map((s) =>
    `<option value="${s.ticker}" ${state.selectedTicker === s.ticker ? "selected" : ""}>${s.ticker} — ${s.name}</option>`
  ).join("");
  el.addEventListener("change", (e) => {
    state.selectedTicker = e.target.value;
    renderDetail();
    renderStocksTable();
    announce(`Selected ${state.selectedTicker}.`);
  });
}

function pctFmt(x, digits = 2) {
  if (x == null || !isFinite(x)) return "—";
  const sign = x > 0 ? "+" : "";
  return sign + (x * 100).toFixed(digits) + "%";
}

function priceFmt(x) {
  if (x == null || !isFinite(x)) return "—";
  return "$" + x.toFixed(2);
}

function dollarFmt(x) {
  if (x == null || !isFinite(x)) return "—";
  return "$" + x.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function signalBadge(signal, opts = {}) {
  const cls = signal === "BUY" ? "badge-buy" : signal === "SELL" ? "badge-sell" : "badge-unsure";
  const icon = signal === "BUY" ? "▲" : signal === "SELL" ? "▼" : "◆";
  const label = `Signal: ${signal}`;
  return `<span class="badge ${cls}" aria-label="${label}${opts.context ? ". " + opts.context : ""}"><span class="icon" aria-hidden="true">${icon}</span>${signal}</span>`;
}

function deltaLabel(x) {
  const cls = x > 0 ? "delta-up" : x < 0 ? "delta-down" : "delta-flat";
  const arrow = x > 0 ? "▲" : x < 0 ? "▼" : "—";
  return `<span class="${cls}"><span aria-hidden="true">${arrow}</span> ${pctFmt(x)}</span>`;
}

let kpisBuilt = false;
const kpiDefs = () => {
  const k = state.portfolio;
  return [
    { id: "kpi-value",  label: "Portfolio value", value: k.value,           fmt: (v) => dollarFmt(v), mod: "accent",                              sub: `${state.stocks.length} starred · equal weight` },
    { id: "kpi-pnl",    label: "Today P&L",       value: k.pnl,             fmt: (v) => dollarFmt(v), mod: k.pnl >= 0 ? "positive" : "negative",  valueMod: k.pnl >= 0 ? "up" : "down", subHtml: deltaLabel(k.pnlPct) },
    { id: "kpi-er",     label: "Expected return", value: k.expectedReturn,  fmt: (v) => pctFmt(v),    mod: k.expectedReturn >= 0 ? "positive" : "negative", valueMod: k.expectedReturn >= 0 ? "up" : "down", sub: `${state.horizon}-day · MC` },
    { id: "kpi-var",    label: "95% VaR",         value: k.var95,           fmt: (v) => pctFmt(v),    mod: "negative", valueMod: "down",           sub: "5th percentile" },
    { id: "kpi-sharpe", label: "Sharpe",          value: k.sharpe,          fmt: (v) => v.toFixed(2), mod: "accent",                              sub: "Ex-ante · rf 4.5%" },
    { id: "kpi-sent",   label: "Crowd score",     value: k.sentimentScore,  fmt: (v) => pctFmt(v, 0), mod: k.sentimentScore >= 0 ? "positive" : "negative", valueMod: k.sentimentScore >= 0 ? "up" : "down", sub: "X net · pos − neg" },
  ];
};

function renderKPIs() {
  const defs = kpiDefs();
  if (!kpisBuilt) {
    $("#kpi-strip").innerHTML = defs.map((d) => `
      <div class="kpi ${d.mod}" id="${d.id}">
        <p class="kpi-label">${d.label}</p>
        <p class="kpi-value ${d.valueMod || ""}" data-val>${d.fmt(d.value)}</p>
        <p class="kpi-sub" data-sub></p>
      </div>
    `).join("");
    kpisBuilt = true;
    state.prevKpi = {};
  }
  defs.forEach((d) => {
    const wrap = $(`#${d.id}`);
    if (!wrap) return;
    wrap.className = `kpi ${d.mod}`;
    const val = wrap.querySelector("[data-val]");
    val.className = `kpi-value ${d.valueMod || ""}`;
    const sub = wrap.querySelector("[data-sub]");
    const prev = state.prevKpi[d.id] ?? d.value;
    animateNumber(val, prev, d.value, d.fmt, 520);
    state.prevKpi[d.id] = d.value;
    if (d.subHtml) sub.innerHTML = d.subHtml;
    else sub.textContent = d.sub || "";
  });

  // Right rail
  const buys = state.stocks.filter((x) => getCombinedSignalForStock(x).signal === "BUY").length;
  const sells = state.stocks.filter((x) => getCombinedSignalForStock(x).signal === "SELL").length;
  const avgProbUp = state.stocks.reduce((a, x) => a + x.mcSummary.probUp, 0) / state.stocks.length;
  const k = state.portfolio;

  const railUpdates = [
    { sel: "#rail-winrate-val", value: avgProbUp * 100,      fmt: (v) => v.toFixed(1) + "%", up: avgProbUp >= 0.5 },
    { sel: "#rail-sharpe-val",  value: k.sharpe,             fmt: (v) => v.toFixed(2),       up: null },
    { sel: "#rail-er-val",      value: k.expectedReturn,     fmt: (v) => pctFmt(v),          up: k.expectedReturn >= 0 },
    { sel: "#rail-var-val",     value: k.var95,              fmt: (v) => pctFmt(v),          up: false },
    { sel: "#rail-sent-val",    value: k.sentimentScore,     fmt: (v) => pctFmt(v, 0),       up: k.sentimentScore >= 0 },
  ];
  railUpdates.forEach((r) => {
    const el = $(r.sel);
    if (!el) return;
    if (r.up !== null) el.className = "rail-box-value " + (r.up ? "up" : "down");
    const prev = state.prevKpi[r.sel] ?? r.value;
    animateNumber(el, prev, r.value, r.fmt, 520);
    state.prevKpi[r.sel] = r.value;
  });
  setText("#rail-er-sub", `Portfolio · ${state.horizon}d`);
  setText("#rail-buys-val", `${buys} / ${sells}`);
  setText("#rail-buys-sub", `${state.stocks.length} starred total`);

  // Terminal strip
  setText("#term-horizon", state.horizon + "D");
  setText("#term-paths", state.sims >= 1000 ? (state.sims / 1000) + "K" : String(state.sims));
  setText("#term-seed", String(state.seed));
  setText("#term-watch", String(state.stocks.length));
  setText("#term-live", String(state.stocks.length));
}

function setText(sel, txt) {
  const el = $(sel);
  if (el) el.textContent = txt;
}

function renderStocksTable() {
  const body = $("#stocks-body");

  // Compute per-ticker price deltas and signal changes before re-rendering
  const priceChanges = {};
  const signalChanges = {};
  for (const s of state.stocks) {
    const prev = state.prevPrices[s.ticker];
    if (prev != null && Math.abs(prev - s.price) > 1e-6) {
      priceChanges[s.ticker] = s.price > prev ? "up" : "down";
    }
    state.prevPrices[s.ticker] = s.price;

    const combined = getCombinedSignalForStock(s).signal;
    if (state.prevSignals[s.ticker] && state.prevSignals[s.ticker] !== combined) {
      signalChanges[s.ticker] = true;
    }
    state.prevSignals[s.ticker] = combined;
  }

  let rows = state.stocks.map((s) => {
    const sig = getCombinedSignalForStock(s);
    return { ...s, combinedSignal: sig };
  });

  // Sort
  const { key, dir } = state.tableSort;
  rows.sort((a, b) => {
    let av, bv;
    switch (key) {
      case "ticker":    av = a.ticker; bv = b.ticker; break;
      case "price":     av = a.price; bv = b.price; break;
      case "change":    av = a.change; bv = b.change; break;
      case "expRet":    av = a.mcSummary.expectedReturn; bv = b.mcSummary.expectedReturn; break;
      case "sentiment": av = a.sentiment.score; bv = b.sentiment.score; break;
      default:          av = a.ticker; bv = b.ticker;
    }
    const cmp = typeof av === "string" ? av.localeCompare(bv) : av - bv;
    return dir === "asc" ? cmp : -cmp;
  });

  body.innerHTML = rows.map((s) => {
    const selected = s.ticker === state.selectedTicker;
    const ciLabel = `${pctFmt(s.mcSummary.ci95Low, 1)} / ${pctFmt(s.mcSummary.ci95High, 1)}`;
    const sentDom = s.sentiment.positive > s.sentiment.negative ? "Pos" : s.sentiment.negative > s.sentiment.positive ? "Neg" : "Mix";
    const sentScore = pctFmt(s.sentiment.score, 0);
    const dotCls = s.combinedSignal.signal === "BUY" ? "buy" : s.combinedSignal.signal === "SELL" ? "sell" : "unsure";
    const conviction = convictionFromStock(s);
    return `
      <tr tabindex="0" role="button" aria-pressed="${selected}" aria-selected="${selected}" data-ticker="${s.ticker}" aria-label="${s.ticker}, ${s.name}. Price ${priceFmt(s.price)}, change ${pctFmt(s.change)}. Action ${s.combinedSignal.signal}, conviction ${conviction.score} of 8.">
        <td class="ticker"><span class="status-dot ${dotCls}" aria-hidden="true"></span>${s.ticker}</td>
        <td class="name">${s.name}</td>
        <td class="num">${priceFmt(s.price)}</td>
        <td class="num">${deltaLabel(s.change)}</td>
        <td class="num ${s.mcSummary.expectedReturn >= 0 ? "delta-up" : "delta-down"}">${pctFmt(s.mcSummary.expectedReturn)}</td>
        <td class="num">${ciLabel}</td>
        <td class="num">${sentDom} ${sentScore}</td>
        <td>${convictionBar(conviction)}</td>
        <td>${signalBadge(s.combinedSignal.signal, { context: s.combinedSignal.detail })}</td>
      </tr>
    `;
  }).join("");

  // Update sort indicators
  $$("thead th[aria-sort]").forEach((th) => th.setAttribute("aria-sort", "none"));
  const activeBtn = $(`thead th button[data-sort="${key}"]`);
  if (activeBtn) {
    activeBtn.closest("th").setAttribute("aria-sort", dir === "asc" ? "ascending" : "descending");
  }

  // Tick-flash on price cells for any ticker whose price moved
  if (!prefersReducedMotion()) {
    Object.entries(priceChanges).forEach(([ticker, dir]) => {
      const row = body.querySelector(`tr[data-ticker="${ticker}"]`);
      if (!row) return;
      const priceCell = row.children[2];
      if (!priceCell) return;
      const color = dir === "up" ? "rgba(0,255,136,0.28)" : "rgba(255,51,102,0.28)";
      const text  = dir === "up" ? "#00ff88" : "#ff3366";
      priceCell.animate(
        [
          { backgroundColor: color,          color: text,            boxShadow: `inset 0 0 0 1px ${text}` },
          { backgroundColor: "transparent",  color: "",              boxShadow: "inset 0 0 0 0 transparent" },
        ],
        { duration: 720, easing: "cubic-bezier(0.22,1,0.36,1)", fill: "none" }
      );
    });
    // Pop action-badge when the combined signal flips
    Object.keys(signalChanges).forEach((ticker) => {
      const row = body.querySelector(`tr[data-ticker="${ticker}"]`);
      if (!row) return;
      const badge = row.querySelector(".badge");
      if (!badge) return;
      badge.animate(
        [
          { transform: "scale(0.94)" },
          { transform: "scale(1.06)" },
          { transform: "scale(1)" },
        ],
        { duration: 320, easing: "cubic-bezier(0.25,1,0.5,1)" }
      );
    });
  }
}

function convictionFromStock(s) {
  // Composite score combining MC expected return, prob up, sentiment, magnitude — bucketed into 0..8
  const mcs = s.mcSummary;
  const raw =
    0.5 * Math.tanh(mcs.expectedReturn * 8) +
    0.3 * (mcs.probUp - 0.5) * 2 +
    0.2 * s.sentiment.score;
  const neg = raw < 0;
  const score = Math.min(8, Math.round(Math.abs(raw) * 12));
  return { score, neg };
}

function convictionBar({ score, neg }) {
  const cells = Array.from({ length: 8 }, (_, i) =>
    `<span class="cell ${i < score ? "on" : ""} ${neg ? "neg" : ""}"></span>`
  ).join("");
  const label = `Conviction ${score} of 8, ${neg ? "bearish" : "bullish"}`;
  return `<span class="bar" role="img" aria-label="${label}">${cells}</span>`;
}

function getCombinedSignalForStock(s) {
  const ctx = { history: s.history, mcSummary: s.mcSummary, sentiment: s.sentiment };
  const perStrategy = STRATEGIES
    .filter((st) => state.selectedStrategies.has(st.id))
    .map((st) => ({ id: st.id, name: st.name, ...st.apply(ctx) }));
  const combined = combineSignals(perStrategy);
  return { ...combined, perStrategy };
}

function renderDetail(onComplete) {
  const stock = state.stocks.find((s) => s.ticker === state.selectedTicker);
  if (!stock) { if (onComplete) onComplete(); return; }

  $("#detail-ticker").textContent = stock.ticker;
  $("#detail-ticker-2").textContent = stock.ticker;
  $("#detail-name").textContent = `— ${stock.name}`;
  $("#detail-subtitle").textContent = `Sector: ${stock.sector} · Live price ${priceFmt(stock.price)} (${pctFmt(stock.change)}) · Drift μ=${(stock.mu * 100).toFixed(1)}%, vol σ=${(stock.sigma * 100).toFixed(1)}%.`;

  const combined = getCombinedSignalForStock(stock);
  $("#detail-signal").innerHTML = signalBadge(combined.signal, { context: combined.detail });

  // MC chart
  const mc = state.mcResult && state.mcResult.ticker === stock.ticker
    ? state.mcResult
    : runMCForStock(stock);
  state.mcResult = { ...mc, ticker: stock.ticker };
  renderChart(mc, stock, onComplete);

  // Sentiment
  renderSentiment(stock);

  // Per-strategy grid
  renderStrategyBreakdown(combined.perStrategy, stock);
}

function runMCForStock(stock) {
  const seed = state.seed + hashString(stock.ticker);
  return monteCarlo({ S0: stock.price, mu: stock.mu, sigma: stock.sigma, days: state.horizon, nPaths: state.sims, seed });
}

const SVG_NS = "http://www.w3.org/2000/svg";

function renderChart(mc, stock, onComplete) {
  const svgW = 720, svgH = 300;
  const padL = 56, padR = 12, padT = 12, padB = 34;
  const innerW = svgW - padL - padR, innerH = svgH - padT - padB;
  const days = mc.days;

  let yMin = Infinity, yMax = -Infinity;
  for (let d = 0; d <= days; d++) {
    if (mc.percentiles.p05[d] < yMin) yMin = mc.percentiles.p05[d];
    if (mc.percentiles.p95[d] > yMax) yMax = mc.percentiles.p95[d];
  }
  for (const path of mc.samplePaths) for (const p of path) { if (p < yMin) yMin = p; if (p > yMax) yMax = p; }
  const pad = (yMax - yMin) * 0.05 || 1;
  yMin -= pad; yMax += pad;

  const xScale = (d) => padL + (d / days) * innerW;
  const yScale = (p) => padT + (1 - (p - yMin) / (yMax - yMin)) * innerH;

  const toPath = (arr) => {
    let d = "";
    for (let i = 0; i < arr.length; i++) d += (i === 0 ? "M" : "L") + xScale(i).toFixed(1) + "," + yScale(arr[i]).toFixed(1) + " ";
    return d;
  };
  const bandPath = () => {
    let d = "";
    for (let i = 0; i <= days; i++) d += (i === 0 ? "M" : "L") + xScale(i) + "," + yScale(mc.percentiles.p95[i]) + " ";
    for (let i = days; i >= 0; i--) d += "L" + xScale(i) + "," + yScale(mc.percentiles.p05[i]) + " ";
    return d + "Z";
  };

  const gridVals = [];
  for (let i = 0; i <= 4; i++) gridVals.push(yMin + (yMax - yMin) * (i / 4));

  const s = mc.summary;
  const captionText = `${mc.nPaths.toLocaleString()} Monte Carlo paths over ${mc.days} trading days. Expected return ${pctFmt(s.expectedReturn)}, median ${pctFmt(s.medianReturn)}, 95% CI ${pctFmt(s.ci95Low)} to ${pctFmt(s.ci95High)}, probability of gain ${(s.probUp * 100).toFixed(0)}%.`;
  const chartAriaLabel = `${stock.ticker} Monte Carlo chart. ${captionText}`;

  // Build frame SVG with placeholders that will be animated in
  $("#mc-chart").innerHTML = `
    <svg viewBox="0 0 ${svgW} ${svgH}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${escapeAttr(chartAriaLabel)}">
      <rect x="0" y="0" width="${svgW}" height="${svgH}" fill="transparent"></rect>
      ${gridVals.map((v) => `
        <line x1="${padL}" x2="${svgW - padR}" y1="${yScale(v)}" y2="${yScale(v)}" stroke="#1a2029" stroke-dasharray="2 4" />
        <text x="${padL - 8}" y="${yScale(v) + 4}" fill="#7f8693" font-size="10" text-anchor="end" font-family="JetBrains Mono, monospace">$${v.toFixed(0)}</text>
      `).join("")}
      ${[0, Math.floor(days / 2), days].map((d) => `
        <text x="${xScale(d)}" y="${svgH - 12}" fill="#7f8693" font-size="10" text-anchor="middle" font-family="JetBrains Mono, monospace">DAY ${d}</text>
      `).join("")}
      <line x1="${padL}" y1="${yScale(mc.S0)}" x2="${svgW - padR}" y2="${yScale(mc.S0)}" stroke="#d6dce4" stroke-opacity="0.18" stroke-width="1" stroke-dasharray="2 2"></line>
      <text x="${svgW - padR}" y="${yScale(mc.S0) - 4}" fill="#7f8693" font-size="10" text-anchor="end" font-family="JetBrains Mono, monospace">S₀ = $${mc.S0.toFixed(2)}</text>
      <g id="paths-layer"></g>
      <path id="mc-band" d="${bandPath()}" fill="rgba(0,255,136,0.08)" stroke="none" opacity="0"></path>
      <path id="mc-p05" d="${toPath(mc.percentiles.p05)}" fill="none" stroke="#00ff88" stroke-width="1" stroke-dasharray="4 4" stroke-opacity="0.6" opacity="0"></path>
      <path id="mc-p95" d="${toPath(mc.percentiles.p95)}" fill="none" stroke="#00ff88" stroke-width="1" stroke-dasharray="4 4" stroke-opacity="0.6" opacity="0"></path>
      <path id="mc-median" d="${toPath(mc.percentiles.p50)}" fill="none" stroke="#00ff88" stroke-width="2.25" stroke-linecap="round" opacity="0"></path>
      <text id="mc-counter" x="${padL}" y="${padT + 14}" font-size="11" text-anchor="start">0 / ${mc.nPaths.toLocaleString()} PATHS</text>
    </svg>
  `;

  $("#mc-caption").textContent = captionText;
  const body = $("#mc-data-body");
  body.innerHTML = [
    ["Ticker", stock.ticker],
    ["Start price", priceFmt(mc.S0)],
    ["Paths simulated", mc.nPaths.toLocaleString()],
    ["Horizon (days)", mc.days],
    ["Expected return", pctFmt(s.expectedReturn)],
    ["Median return", pctFmt(s.medianReturn)],
    ["5th percentile", pctFmt(s.ci95Low)],
    ["95th percentile", pctFmt(s.ci95High)],
    ["Probability of gain", (s.probUp * 100).toFixed(1) + "%"],
    ["Mean final price", priceFmt(s.meanPrice)],
    ["Median final price", priceFmt(s.medianPrice)],
  ].map(([k, v]) => `<tr><th scope="row">${k}</th><td>${v}</td></tr>`).join("");

  // === Animate draw ===
  const figure = document.querySelector(".chart-wrap");
  const layer = $("#paths-layer");
  const counter = $("#mc-counter");
  const reduced = prefersReducedMotion();

  const drawPaths = () => {
    const paths = mc.samplePaths.map((arr, i) => {
      const p = document.createElementNS(SVG_NS, "path");
      p.setAttribute("d", toPath(arr));
      p.setAttribute("fill", "none");
      p.setAttribute("stroke", "#00ff88");
      p.setAttribute("stroke-width", "1");
      p.setAttribute("stroke-opacity", "0.18");
      layer.appendChild(p);
      return { el: p, idx: i };
    });

    if (reduced) {
      paths.forEach(({ el }) => { el.style.strokeDashoffset = 0; });
      counter.textContent = `${mc.nPaths.toLocaleString()} / ${mc.nPaths.toLocaleString()} PATHS`;
      ["#mc-band", "#mc-p05", "#mc-p95", "#mc-median"].forEach((sel) => $(sel).setAttribute("opacity", "1"));
      figure.classList.remove("simulating");
      if (onComplete) onComplete();
      return;
    }

    const displayStep = mc.nPaths / paths.length;
    let drawnCount = 0;
    const perPath = Math.max(480, Math.min(720, 2400 / Math.max(paths.length, 1)));
    const staggerMs = Math.max(8, Math.min(30, 1800 / Math.max(paths.length, 1)));

    paths.forEach(({ el, idx }) => {
      const len = el.getTotalLength();
      el.style.strokeDasharray = len;
      el.style.strokeDashoffset = len;
      const anim = el.animate(
        [{ strokeDashoffset: len }, { strokeDashoffset: 0 }],
        { duration: perPath, delay: idx * staggerMs, easing: "cubic-bezier(0.16,1,0.3,1)", fill: "forwards" }
      );
      anim.onfinish = () => {
        drawnCount++;
        const shown = Math.min(mc.nPaths, Math.round(drawnCount * displayStep));
        counter.textContent = `${shown.toLocaleString()} / ${mc.nPaths.toLocaleString()} PATHS`;
        if (drawnCount === paths.length) {
          const reveal = [
            ["#mc-band", 0],
            ["#mc-p05", 80],
            ["#mc-p95", 120],
            ["#mc-median", 180],
          ];
          reveal.forEach(([sel, d]) => {
            const e = $(sel);
            if (!e) return;
            e.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 420, delay: d, easing: "cubic-bezier(0.25,1,0.5,1)", fill: "forwards" });
          });
          setTimeout(() => {
            figure.classList.remove("simulating");
            if (onComplete) onComplete();
          }, 620);
          counter.textContent = `${mc.nPaths.toLocaleString()} PATHS · E[R] ${pctFmt(s.expectedReturn)}`;
        }
      };
    });
  };

  figure.classList.add("simulating");
  // Defer to next frame so DOM is settled before computing getTotalLength
  requestAnimationFrame(drawPaths);
}

function escapeAttr(s) {
  return String(s).replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderSentiment(stock) {
  const s = stock.sentiment;
  $("#sent-bar-pos").style.width = (s.positive * 100).toFixed(1) + "%";
  $("#sent-bar-neu").style.width = (s.neutral * 100).toFixed(1) + "%";
  $("#sent-bar-neg").style.width = (s.negative * 100).toFixed(1) + "%";
  $("#sent-pos-val").textContent = (s.positive * 100).toFixed(0) + "%";
  $("#sent-neu-val").textContent = (s.neutral * 100).toFixed(0) + "%";
  $("#sent-neg-val").textContent = (s.negative * 100).toFixed(0) + "%";

  const bars = $("#sentiment-bars-wrap");
  bars.setAttribute("aria-label",
    `Sentiment: ${(s.positive * 100).toFixed(0)}% positive, ${(s.neutral * 100).toFixed(0)}% neutral, ${(s.negative * 100).toFixed(0)}% negative.`);

  const list = $("#post-list");
  const posts = s.posts.length ? s.posts : [{ sent: "neu", author: "@market_data", handle: "now", text: "No recent public chatter for this ticker." }];
  list.innerHTML = posts.map((p) => `
    <li class="post ${p.sent}">
      <div class="post-meta">
        <span class="post-author">${p.author}</span>
        <span>${p.handle} · <span class="sr-only">sentiment </span>${p.sent === "pos" ? "positive" : p.sent === "neg" ? "negative" : "neutral"}</span>
      </div>
      <div class="post-text">${p.text}</div>
    </li>
  `).join("");
}

function renderStrategyBreakdown(perStrategy, stock) {
  const grid = $("#strategy-grid");
  if (!perStrategy.length) {
    grid.innerHTML = `<div class="strategy-card"><h4>No strategies selected</h4><p class="strategy-note">Pick at least one strategy above to see signals.</p></div>`;
    return;
  }
  grid.innerHTML = perStrategy.map((r) => `
    <article class="strategy-card">
      <h4>${r.name}</h4>
      <div class="strategy-value">${signalBadge(r.signal, { context: r.detail })}</div>
      <p class="strategy-note">${r.detail}</p>
    </article>
  `).join("");
}

function renderSummary() {
  const grid = $("#summary-grid");
  grid.innerHTML = state.stocks.map((s) => {
    const combined = getCombinedSignalForStock(s);
    const mcs = s.mcSummary;
    const sent = s.sentiment;
    const sentDom = sent.positive > sent.negative ? "positive" : sent.negative > sent.positive ? "negative" : "mixed";
    const outlook =
      mcs.expectedReturn > 0.03 && sent.score > 0.1 ? "strong constructive setup" :
      mcs.expectedReturn < -0.03 && sent.score < -0.1 ? "caution warranted" :
      mcs.expectedReturn > 0 ? "modestly positive skew" :
      "mixed outlook";
    return `
      <article class="summary-card">
        <div class="summary-card-head">
          <span class="summary-ticker">${s.ticker}</span>
          <span class="summary-name">${s.sector}</span>
        </div>
        <p class="summary-body">
          <strong>${s.name}</strong> trades near ${priceFmt(s.price)} (${pctFmt(s.change)} today). Monte Carlo across ${state.horizon} trading days projects an expected return of <strong>${pctFmt(mcs.expectedReturn)}</strong> with a 95% CI of ${pctFmt(mcs.ci95Low, 1)} to ${pctFmt(mcs.ci95High, 1)}. X chatter is <strong>${sentDom}</strong> (net ${pctFmt(sent.score, 0)}). Overall, ${outlook}.
        </p>
        <div class="summary-footer">
          <span>${signalBadge(combined.signal, { context: combined.detail })}</span>
          <span class="${mcs.expectedReturn >= 0 ? "delta-up" : "delta-down"}">E[R] ${pctFmt(mcs.expectedReturn)}</span>
        </div>
      </article>
    `;
  }).join("");

  // Aggregate signal for the whole watchlist
  const all = state.stocks.map((s) => getCombinedSignalForStock(s).signal);
  const buy = all.filter((x) => x === "BUY").length;
  const sell = all.filter((x) => x === "SELL").length;
  const uns = all.filter((x) => x === "UNSURE").length;
  const winner = buy > sell && buy > uns ? "BUY" : sell > buy && sell > uns ? "SELL" : "UNSURE";
  const agg = `${buy} BUY · ${uns} UNSURE · ${sell} SELL across ${all.length} starred stocks.`;
  $("#aggregate-signal").innerHTML = signalBadge(winner, { context: agg });
  $("#summary-signal").innerHTML = signalBadge(winner, { context: agg });

  if (state.prevAggregateSignal && state.prevAggregateSignal !== winner && !prefersReducedMotion()) {
    ["#aggregate-signal .badge", "#summary-signal .badge"].forEach((sel) => {
      const b = $(sel);
      if (!b) return;
      b.animate(
        [{ transform: "scale(0.92)" }, { transform: "scale(1.08)" }, { transform: "scale(1)" }],
        { duration: 360, easing: "cubic-bezier(0.22,1,0.36,1)" }
      );
    });
  }
  state.prevAggregateSignal = winner;
}

// ========== Announce (live region) ==========
function announce(msg) {
  const el = $("#live-status");
  el.textContent = "";
  // force reflow so consecutive identical messages still announce
  void el.offsetWidth;
  el.textContent = msg;
  $("#status-line").textContent = msg;
}

function renderAll() {
  renderKPIs();
  renderStocksTable();
  renderDetail();
  renderSummary();
}

// ========== Live tick simulation ==========
let tickTimer = null;
let uptimeTimer = null;
const bootTime = Date.now();

function formatUptime(ms) {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(h)}:${pad(m)}:${pad(sec)}`;
}

function startLiveTicks() {
  if (tickTimer) clearInterval(tickTimer);
  if (uptimeTimer) clearInterval(uptimeTimer);

  // Uptime ticks every second for the terminal strip
  uptimeTimer = setInterval(() => {
    setText("#term-uptime", formatUptime(Date.now() - bootTime));
    // jitter latency for atmosphere (1-4 ms)
    const lat = 1 + Math.floor(Math.random() * 4);
    setText("#term-latency", lat + "MS");
  }, 1000);

  tickTimer = setInterval(() => {
    // small random walk per stock
    for (const s of state.stocks) {
      const rng = Math.random;
      const dt = 1 / (252 * 24);
      const z = (rng() - 0.5) * 2 * 1.2;
      const delta = (s.mu - 0.5 * s.sigma ** 2) * dt + s.sigma * Math.sqrt(dt) * z;
      const newPrice = s.price * Math.exp(delta);
      s.price = Math.max(1, newPrice);
      s.history[s.history.length - 1] = s.price;
      s.change = (s.price - s.prevClose) / s.prevClose;
    }
    state.portfolio = computePortfolioKpis(state.stocks);
    const now = new Date();
    const t = $("#last-update");
    t.textContent = now.toLocaleTimeString();
    t.setAttribute("datetime", now.toISOString());
    renderKPIs();
    renderStocksTable();
  }, 3000);
}

// ========== Event wiring ==========
function wireEvents() {
  // Run button
  $("#run-btn").addEventListener("click", () => {
    const btn = $("#run-btn");
    if (btn.classList.contains("running")) return;
    btn.classList.add("running");
    btn.setAttribute("aria-busy", "true");
    const original = btn.textContent;
    btn.textContent = "Simulating";
    announce(`Running ${state.sims.toLocaleString()} simulations for ${state.selectedTicker}…`);
    setTimeout(() => {
      state.mcResult = null;
      renderDetail(() => {
        btn.classList.remove("running");
        btn.removeAttribute("aria-busy");
        btn.textContent = original;
        const er = state.mcResult && state.mcResult.summary
          ? pctFmt(state.mcResult.summary.expectedReturn)
          : "—";
        announce(`Simulation complete for ${state.selectedTicker}. Expected return ${er}.`);
      });
    }, 30);
  });

  // Sims range
  $("#sims-range").addEventListener("input", (e) => {
    state.sims = +e.target.value;
    $("#sims-value").textContent = state.sims.toLocaleString();
    e.target.setAttribute("aria-valuetext", `${state.sims.toLocaleString()} simulation paths`);
  });

  // Horizon
  $("#horizon-select").addEventListener("change", (e) => {
    state.horizon = +e.target.value;
    state.mcResult = null;
    // also update per-stock MC summaries at new horizon for table
    for (const s of state.stocks) {
      const seed = state.seed + hashString(s.ticker);
      const mc = monteCarlo({ S0: s.price, mu: s.mu, sigma: s.sigma, days: state.horizon, nPaths: 500, seed });
      s.mcSummary = mc.summary;
    }
    state.portfolio = computePortfolioKpis(state.stocks);
    renderAll();
    announce(`Horizon set to ${state.horizon} trading days.`);
  });

  // Seed
  $("#seed-input").addEventListener("change", (e) => {
    const v = e.target.value === "random" ? Math.floor(Math.random() * 1e6) : +e.target.value;
    state.seed = v;
    // Rebuild whole runtime (histories depend on seed)
    state.stocks = buildStocksRuntime();
    state.portfolio = computePortfolioKpis(state.stocks);
    state.mcResult = null;
    renderAll();
    announce(`Random seed set to ${e.target.value}.`);
  });

  // Stocks table: click + keyboard on rows + sortable headers
  const body = $("#stocks-body");
  body.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-ticker]");
    if (!tr) return;
    selectTickerFromRow(tr.dataset.ticker);
  });
  body.addEventListener("keydown", (e) => {
    const tr = e.target.closest("tr[data-ticker]");
    if (!tr) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      selectTickerFromRow(tr.dataset.ticker);
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const rows = $$("#stocks-body tr[data-ticker]");
      const idx = rows.indexOf(tr);
      const next = e.key === "ArrowDown" ? rows[idx + 1] : rows[idx - 1];
      if (next) next.focus();
    }
  });

  $$("thead th button[data-sort]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.sort;
      if (state.tableSort.key === key) {
        state.tableSort.dir = state.tableSort.dir === "asc" ? "desc" : "asc";
      } else {
        state.tableSort.key = key;
        state.tableSort.dir = key === "ticker" ? "asc" : "desc";
      }
      renderStocksTable();
      announce(`Sorted by ${key}, ${state.tableSort.dir === "asc" ? "ascending" : "descending"}.`);
    });
  });
}

function selectTickerFromRow(ticker) {
  state.selectedTicker = ticker;
  $("#ticker-select").value = ticker;
  renderDetail();
  renderStocksTable();
  // Don't steal focus — just announce
  announce(`Selected ${ticker}. Detail panel updated.`);
  // Scroll detail into view for convenience but keep focus
  $("#detail-section").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ========== Init ==========
function init() {
  state.stocks = buildStocksRuntime();
  state.portfolio = computePortfolioKpis(state.stocks);

  renderStrategyOptions();
  renderTickerSelect();

  $("#sims-value").textContent = state.sims.toLocaleString();
  $("#last-update").textContent = new Date().toLocaleTimeString();

  wireEvents();
  renderAll();
  startLiveTicks();
  announce("Dashboard ready.");
}

// Wait for DOM if script tag is at end this runs immediately anyway.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
