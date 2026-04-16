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
- EV/Sales (enterprise value / revenue) is the ONLY user-facing valuation metric
- EV/Sales accounts for debt; P/S does not
- Always compare stocks within their sector, not across sectors (S&J advice)
- `EVS_SectorPct` = each stock's EV/Sales percentile vs sector peers, computed in portfolio_app.py after data load
- `PS_HistPos`, `PS_Current`, `PS_3yr_*` are still computed in data.py as silent fallback only — they never appear in the UI

### Coiled Base Setup ("money printer")
S&J heuristic: "basing + tightening + compressing with key moving averages aligning below price".
Four measurable components in `fetch_price_data` (data.py):
- **BaseTightPct** — 20d high-low as % of price (tight: ≤12%)
- **ATRContract** — 14d ATR / 60d average ATR (tightening: ≤0.80)
- **BBWidthPct** — current Bollinger width percentile in last 126d (compressing: ≤20%)
- **MAStack** — Price > MA20 > MA50 > MA200 with MA20/50 rising
- **CoiledScore** — count (0–4) of how many components fire

Incorporation:
- `calc_setup_stage` returns "Coiled Base" when score ≥3 (takes precedence over Basing)
- Technical pillar gets +7 pts at 3/4, +12 pts at 4/4 (`THRESHOLDS["coiled_tech_boost_*"]`)
- Dedicated UI section in Convexity tab lists all qualifying tickers (replaced old "Compressed Spring" chart)

### Data Sources
- **yfinance** — price data only (batch `yf.download`); fundamentals migrated to Finnhub
- **Finnhub** — fundamentals, insider transactions, news, earnings calendar (free tier)
- **xAI Grok** — headline sentiment scoring, X/social sentiment, narrative summary, X insights
- **StockTwits** — retail sentiment fallback (bull/bear %) if X sentiment unavailable
- **Google News RSS** — market headlines
- **FMP (Financial Modeling Prep)** — ticker metadata / theme assignment
- yfinance is only used for price data now — `yf.download()` on a single thread (reliable)
- Disk cache: `fund_cache.json` for fundamentals, `meta_cache.json` for metadata

### xAI / Grok API — IMPORTANT
Two separate endpoints with different formats:

**1. Chat completions** (`/v1/chat/completions`) — standard calls, no live search:
```python
payload = {"model": "grok-3-mini", "messages": [...], "max_tokens": N}
response["choices"][0]["message"]["content"]
```

**2. Responses API** (`/v1/responses`) — required for Agent Tools (live search):
```python
payload = {"model": "grok-3", "input": [...], "max_output_tokens": N, "tools": [{"type": "x_search"}, {"type": "web_search"}]}
response["output"][0]["content"][0]["text"]
```
- `search_parameters` is **deprecated** (returns 410) — always use Responses API for live search
- Tool types: `x_search` (X/Twitter), `web_search` (web)
- Docs: https://docs.x.ai/docs/guides/tools/overview

### API Keys
- Stored in Streamlit secrets (`st.secrets`) only — never in .env files
- Keys needed: `XAI_API_KEY`, `FINNHUB_API_KEY`, `FMP_API_KEY`
- `ANTHROPIC_API_KEY` is no longer used (kept in secrets.toml but code uses xAI)

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
