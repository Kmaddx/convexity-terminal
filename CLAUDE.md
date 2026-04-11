# Convexity Terminal — Project Notes for Claude

## What This Is
A Streamlit investment terminal for tracking a personal watchlist of growth/tech stocks.
Deployed on Streamlit Cloud and run locally on port 8501/8502.
GitHub repo: Kmaddx/convexity-terminal

## File Structure
- `portfolio_app.py` — main Streamlit app (~2900 lines), 8 tabs
- `data.py` — all data fetching (yfinance, Anthropic API, StockTwits, Google News RSS)
- `scoring.py` — four-pillar scoring engine + market environment scoring
- `themes.py` — theme management and auto-assignment
- `portfolio_chart.py`, `portfolio_tech.py`, `portfolio_check.py` — utilities (partially unused)

## The Four-Pillar Scoring System
Every stock gets scored 0-100 on four pillars. All must score >= 50 for "Aligned".

- **Technical** (30% weight): RSI, MA50, 52wk position, trend consistency, down-day RS
- **Fundamental** (30%): Revenue growth, gross margin, FCF, EV/Sales, insider ownership, analyst targets, Rule of 40
- **Thematic** (20%): Theme membership, ETF momentum, ticker vs SPY, short squeeze
- **Narrative** (20%): StockTwits sentiment, AI headline score (Claude Haiku), insider buying signal

Convexity Score = weighted average of four pillars.
All thresholds are in `THRESHOLDS` dict at top of `scoring.py` — tune there, not inline.

## Key Design Decisions

### Valuation: EV/Sales not P/S
- Use EV/Sales (enterprise value / revenue) as primary valuation metric
- EV/Sales accounts for debt; P/S does not
- Always compare stocks within their sector, not across sectors (S&J advice)
- `EVS_SectorPct` = each stock's EV/Sales percentile vs sector peers, computed in portfolio_app.py after data load
- P/S is kept for display reference but does not drive scoring

### Data Sources
- **yfinance** — price data, fundamentals, insider transactions (unreliable; known 401/crumb errors)
- **Anthropic Claude Haiku** — AI headline sentiment scoring and market summary
- **StockTwits** — retail sentiment (bull/bear %)
- **Google News RSS** — market headlines
- **FMP (Financial Modeling Prep)** — ticker metadata / theme assignment
- yfinance is the biggest pain point — it has intermittent 401 "Invalid Crumb" errors from Yahoo Finance
- All fetches have timeouts (15s per ticker, 30s overall) to prevent hanging
- Disk cache: `fund_cache.json` for fundamentals, `meta_cache.json` for metadata

### API Keys
- Stored in Streamlit secrets (`st.secrets`) only — never in .env files
- Keys needed: `ANTHROPIC_API_KEY`, `FMP_API_KEY`

### Performance
- ThreadPoolExecutor(4-6 workers) for parallel fundamentals and extras fetching
- `@st.cache_data` with TTLs: 300s price, 600s market env, 900s headlines, 1800s fundamentals
- `st.fragment` used for ticker add/remove to avoid full page refresh

## UI Rules
- No emojis anywhere in the UI or code — looks AI-generated
- Dark theme throughout (#0d1117 background, #58a6ff accent blue)
- Top bar for controls (no sidebar — it was fragile on Streamlit Cloud)
- Settings tab for watchlist/ticker management

## Known Issues / Ongoing Work
- yfinance 401 errors are a recurring problem; Finnhub considered as replacement but not yet done
- Historical EV/Sales range (EVS_HistPos) is computed per-ticker at load time — slow for 30+ tickers
- `portfolio_check.py` and `portfolio_tech.py` appear unused — investigate before deleting
- No test suite exists

## Running Locally
```bash
cd /Users/kieran/Projects/Investment
streamlit run portfolio_app.py
```

## Deployment
Streamlit Cloud — pushes to main branch auto-deploy.
```bash
git add -p && git commit -m "..." && git push
```

## Investment Philosophy Context
- Focus: high-growth, high-beta stocks with convexity (asymmetric upside)
- Peer comparison is essential — never compare across sectors (e.g. AMPX vs Kohl's)
- Insider buying (open market purchases only, not option exercises) is a strong signal
- EarningsBeats metric was removed — minor signal, heavy API cost
