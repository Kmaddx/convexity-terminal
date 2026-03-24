import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
import json
import os

# ── Module imports (refactored from monolith) ────────────────────────────────
from scoring import (
    _safe, calc_rsi, calc_atr_pct, calc_setup_stage,
    calc_four_pillars, calc_convexity_score,
    calc_market_env_score, calc_execution_window,
)
from data import (
    fetch_price_data, fetch_spy_daily, fetch_spy_returns,
    fetch_market_environment, calc_downday_rs,
    fetch_etf_benchmark_data, fetch_fundamentals, fetch_extras,
    fetch_stocktwits, score_scanner_candidates,
    scan_yahoo_screener, scan_yahoo_predefined, scan_finviz, scan_yahoo_trending,
    claude_summarize, score_headlines_ai,
    fetch_market_headlines, ai_market_summary,
)
from themes import (
    load_themes, save_themes, get_ticker_themes, get_theme_tickers,
    get_theme_etfs, get_all_etf_tickers, _get_anthropic_key,
    DEFAULT_THEMES,
)

st.set_page_config(
    page_title="Convexity Terminal",
    page_icon="◣",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base layout ── */
    .block-container { padding-top: 0.75rem; padding-bottom: 0rem; max-width: 100%; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.4rem; }
    .main .block-container { font-family: 'Inter', -apple-system, sans-serif; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
    div[data-testid="stToolbar"] { display: none !important; }
    .stAppToolbar { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }
    /* Hide sidebar completely — all management is in Settings tab */
    section[data-testid="stSidebar"] { display: none !important; }
    button[data-testid="stSidebarNavToggle"] { display: none !important; }
    button[data-testid="collapsedControl"] { display: none !important; }

    /* ── Metrics cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        padding: 10px 14px; border-radius: 8px;
        border: 1px solid #30363d;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        font-family: 'Inter', sans-serif; font-size: 0.7rem;
        color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    }

    /* ── Tabs ── */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 500;
        letter-spacing: 0.3px; text-transform: uppercase;
        padding: 8px 16px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #58a6ff !important;
        color: #58a6ff !important;
    }
    div[data-baseweb="tab-border"] { background-color: #21262d !important; }

    /* ── Dataframe tables ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d; border-radius: 8px; overflow: hidden;
    }
    div[data-testid="stDataFrame"] table { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }

    /* ── Expanders ── */
    details[data-testid="stExpander"] {
        border: 1px solid #30363d; border-radius: 8px;
        background-color: #161b22;
    }
    details[data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif; font-weight: 500;
    }

    /* ── Section headers ── */
    .section-header {
        display: flex; align-items: center; gap: 8px;
        padding: 8px 0 6px 0; margin-top: 4px;
        border-bottom: 2px solid #21262d;
        font-family: 'Inter', sans-serif; font-weight: 600; font-size: 1rem;
        color: #e6edf3; letter-spacing: 0.3px;
    }
    .section-header .accent {
        width: 3px; height: 20px; border-radius: 2px; flex-shrink: 0;
    }

    /* ── Alert cards ── */
    .alert-card {
        padding: 10px 14px; border-radius: 8px; margin: 4px 0;
        font-family: 'Inter', sans-serif; font-size: 0.82rem;
        border-left: 3px solid; line-height: 1.5;
    }
    .alert-card.green { background: #0d1f17; border-color: #2ecc71; color: #a3d9b1; }
    .alert-card.red { background: #1f0d0d; border-color: #e74c3c; color: #d9a3a3; }
    .alert-card.amber { background: #1f1a0d; border-color: #f39c12; color: #d9c9a3; }
    .alert-card.blue { background: #0d141f; border-color: #3498db; color: #a3c4d9; }

    /* ── KPI row ── */
    .kpi-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d; border-radius: 8px;
        padding: 12px 16px; text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .kpi-label {
        font-family: 'Inter', sans-serif; font-size: 0.65rem;
        color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;
        font-weight: 700; line-height: 1.2;
    }
    .kpi-sub {
        font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #8b949e;
        margin-top: 2px;
    }

    /* ── Plotly chart containers ── */
    div[data-testid="stPlotlyChart"] {
        border: 1px solid #21262d; border-radius: 8px; overflow: hidden;
    }

    /* ── Gradient divider ── */
    .grad-divider {
        height: 1px; margin: 12px 0;
        background: linear-gradient(90deg, transparent 0%, #30363d 20%, #30363d 80%, transparent 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* ── Download buttons ── */
    button[data-testid="stDownloadButton"] {
        font-family: 'Inter', sans-serif; font-size: 0.75rem;
        border: 1px solid #30363d; border-radius: 6px;
        background-color: #161b22;
    }

    /* ── Captions ── */
    div[data-testid="stCaptionContainer"] {
        font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper functions for styled UI elements ──
def section_header(title, accent_color="#58a6ff"):
    """Render a styled section header with accent bar."""
    st.markdown(
        f'<div class="section-header">'
        f'<span class="accent" style="background-color:{accent_color};"></span>'
        f'{title}</div>',
        unsafe_allow_html=True,
    )

def alert_card(text, color="blue"):
    """Render a styled alert card. color: green/red/amber/blue."""
    st.markdown(f'<div class="alert-card {color}">{text}</div>', unsafe_allow_html=True)

def grad_divider():
    """Render a subtle gradient divider."""
    st.markdown('<div class="grad-divider"></div>', unsafe_allow_html=True)

DEFAULT_TICKERS = [
    "PLTR", "RKLB", "TSLA", "HOOD", "ASTS"
]

TICKERS_FILE     = os.path.join(os.path.dirname(__file__), "tickers.json")
WATCHLISTS_FILE  = os.path.join(os.path.dirname(__file__), "watchlists.json")
SCORES_FILE      = os.path.join(os.path.dirname(__file__), "score_history.json")

if "themes" not in st.session_state:
    st.session_state.themes = load_themes()

# ── Removed: theme rules, metadata cache, FMP/Anthropic key helpers,
#    auto_assign_themes, theme helpers — all moved to themes.py ──


def load_watchlists():
    """Load named watchlists. Migrates old tickers.json if present."""
    if os.path.exists(WATCHLISTS_FILE):
        try:
            with open(WATCHLISTS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            pass
    # Migrate from old flat tickers.json if it exists
    if os.path.exists(TICKERS_FILE):
        try:
            with open(TICKERS_FILE) as f:
                old = json.load(f)
            if isinstance(old, list):
                return {"All Tickers": old}
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            pass
    return {"All Tickers": DEFAULT_TICKERS.copy()}

def save_watchlists(wl):
    try:
        with open(WATCHLISTS_FILE, "w") as f:
            json.dump(wl, f, indent=2)
    except (IOError, TypeError, OSError):
        pass

def parse_yahoo_csv(uploaded_file):
    """Parse a Yahoo Finance watchlist CSV export. Returns list of ticker symbols."""
    try:
        df = pd.read_csv(uploaded_file)
        # Yahoo exports use "Symbol" column
        for col in ["Symbol", "symbol", "Ticker", "ticker"]:
            if col in df.columns:
                return [s.strip().upper() for s in df[col].dropna().astype(str).tolist()
                        if s.strip() and s.strip() != "nan"]
        # Fallback: first column
        return [s.strip().upper() for s in df.iloc[:, 0].dropna().astype(str).tolist()
                if s.strip() and s.strip() != "nan"]
    except (pd.errors.ParserError, ValueError, KeyError, AttributeError):
        return []

if "watchlists" not in st.session_state:
    st.session_state.watchlists = load_watchlists()
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = list(st.session_state.watchlists.keys())[0]

# Convenience: current tickers = active watchlist's tickers
st.session_state.tickers = st.session_state.watchlists.get(
    st.session_state.active_watchlist, DEFAULT_TICKERS
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def fmt_mcap(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    if v >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"

def rsi_label(v):
    if v >= 70: return "Overbought"
    if v <= 30: return "Oversold"
    return "Neutral"

def pos_color(v):
    if v >= 70: return "#27ae60"
    if v >= 40: return "#f39c12"
    return "#e74c3c"

DARK = dict(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="Inter, -apple-system, sans-serif", size=12),
    xaxis_gridcolor="#21262d", yaxis_gridcolor="#21262d",
    xaxis_linecolor="#30363d", yaxis_linecolor="#30363d",
    xaxis_zeroline=False, yaxis_zeroline=False,
    hoverlabel=dict(bgcolor="#1c2333", bordercolor="#30363d", font=dict(family="Inter, sans-serif", size=12, color="#e6edf3")),
)

def log_scores(df):
    """Append today's scores to a local JSON file for trend tracking."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        history = {}
        if os.path.exists(SCORES_FILE):
            with open(SCORES_FILE) as f:
                history = json.load(f)
        for _, row in df.iterrows():
            t = row["Ticker"]
            if t not in history:
                history[t] = []
            # Don't duplicate today's entry
            entry = {
                    "date": today,
                    "convexity": round(float(_safe(row.get("ConvexityScore"))), 1),
                    "momentum": round(float(_safe(row.get("MomentumScore"))), 1),
                    "asymmetry": round(float(_safe(row.get("AsymmetryScore"))), 1),
                    "stool": round(float(_safe(row.get("VGScore"))), 1),
                    "price": round(float(_safe(row.get("Price"))), 2),
                    "rs_label": str(row.get("RS_Label", "")) if row.get("RS_Label") else "",
                    "setup_stage": str(row.get("SetupStage", "")) if row.get("SetupStage") else "",
                }
            if history[t] and history[t][-1].get("date") == today:
                history[t][-1] = entry
            else:
                history[t].append(entry)
            # Keep last 90 days max
            history[t] = history[t][-90:]
        with open(SCORES_FILE, "w") as f:
            json.dump(history, f, indent=1)
    except (IOError, TypeError, OSError):
        pass

def load_score_history():
    """Load score history from local JSON."""
    if os.path.exists(SCORES_FILE):
        try:
            with open(SCORES_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            pass
    return {}

# ── Sidebar ──────────────────────────────────────────────────────────────────

# ── Watchlist selection (must happen before data load) ────────────────────────
wl_names = list(st.session_state.watchlists.keys())
active_idx = wl_names.index(st.session_state.active_watchlist) if st.session_state.active_watchlist in wl_names else 0
selected = st.session_state.tickers

# ── Load all data once ───────────────────────────────────────────────────────

with st.status("◣ Loading Convexity Terminal...", expanded=True) as _status:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _status.update(label="◣ Scanning markets...", state="running")

    # Phase 1: Price data first (needed for filtering), plus independent fetches in parallel
    st.write("Fetching price data + market environment...")
    _all_etf_tickers = get_all_etf_tickers(st.session_state.themes)
    _anthropic_key = _get_anthropic_key()
    _tickers_tuple = tuple(st.session_state.tickers)
    _etf_tuple = tuple(_all_etf_tickers)

    with ThreadPoolExecutor(max_workers=8) as _pool:
        _f_price = _pool.submit(fetch_price_data, st.session_state.tickers)
        _f_fund = _pool.submit(fetch_fundamentals, st.session_state.tickers)
        _f_spy_ret = _pool.submit(fetch_spy_returns)
        _f_spy_daily = _pool.submit(fetch_spy_daily)
        _f_env = _pool.submit(fetch_market_environment)
        _f_headlines = _pool.submit(fetch_market_headlines)
        _f_etf = _pool.submit(fetch_etf_benchmark_data, _etf_tuple) if _all_etf_tickers else None
        _f_extras = _pool.submit(fetch_extras, _tickers_tuple)
        _f_st = _pool.submit(fetch_stocktwits, _tickers_tuple)

        # Collect results as they complete
        df_price = _f_price.result()
        if df_price.empty:
            _status.update(label="◣ Failed", state="error")
            import data as _data_mod
            _err = _data_mod._price_fetch_error
            _is_yahoo_issue = any(x in _err for x in ["401", "Crumb", "Unauthorized"])
            if _is_yahoo_issue:
                st.error(
                    "Yahoo Finance is temporarily unavailable (authentication error).  \n"
                    "This is a known intermittent issue on Yahoo's end — usually resolves within 15–30 minutes.  \n\n"
                    "**Try refreshing in a few minutes.**"
                )
            else:
                st.error("No data loaded. Check your internet connection and try again.")
            if st.button("Retry Now", type="primary"):
                st.cache_data.clear()
                st.rerun()
            st.stop()

        st.write("Loading fundamentals & signals...")
        df_fund = _f_fund.result()
        spy_ret = _f_spy_ret.result()
        spy_close, spy_daily_ret = _f_spy_daily.result()
        market_env = _f_env.result()
        _market_headlines = _f_headlines.result()
        df_etf = _f_etf.result() if _f_etf else pd.DataFrame()
        df_extras = _f_extras.result()
        st_data = _f_st.result()

    st.write("Calculating scores...")
    env_total, env_pillars, env_decision, env_decision_sub, env_warnings, env_tailwinds = calc_market_env_score(market_env)

    _market_ai_summary = ai_market_summary(tuple(
        (h["title"], h["source"]) for h in _market_headlines
    ), _anthropic_key) if _anthropic_key and _market_headlines else None

    df_rs = calc_downday_rs(_tickers_tuple, spy_close, spy_daily_ret)

    df_price = df_price[df_price["Ticker"].isin(selected)].copy()

    # AI headline sentiment (uses headlines already fetched in df_extras)
    _ai_sentiment = {}
    if _anthropic_key and not df_extras.empty:
        _headlines_map = {}
        for _, _erow in df_extras.iterrows():
            _t = _erow.get("Ticker")
            _h = _erow.get("Headlines")
            if _t and _h:
                _headlines_map[_t] = _h
        if _headlines_map:
            _ai_sentiment = score_headlines_ai(
                tuple(sorted(_headlines_map.items())), _anthropic_key
            )

    _status.update(label="◣ Terminal Ready", state="complete", expanded=False)

# ── Merge all data ──
df_all = df_price.merge(df_fund, on="Ticker", how="left")
df_all = df_all.merge(
    df_extras[["Ticker", "InsiderSignal", "InsiderNet", "InsiderBuys"]],
    on="Ticker", how="left"
)
df_all["ST_BullPct"] = df_all["Ticker"].map(lambda t: st_data.get(t, {}).get("bull_pct"))
df_all["ST_MsgCount"] = df_all["Ticker"].map(lambda t: st_data.get(t, {}).get("msg_count", 0))

# AI headline sentiment
df_all["AI_HeadlineScore"] = df_all["Ticker"].map(lambda t: _ai_sentiment.get(t, {}).get("score"))
df_all["AI_HeadlineSummary"] = df_all["Ticker"].map(lambda t: _ai_sentiment.get(t, {}).get("summary", ""))

# Merge relative strength data
if not df_rs.empty:
    df_all = df_all.merge(
        df_rs[["Ticker", "RS_Score", "RS_Rank", "RS_Label", "DownDayRS",
               "DownDayWinRate", "RecentDDExcess", "EarlyBottom", "DaysSinceLow",
               "RS_1m", "RS_3m"]],
        on="Ticker", how="left"
    )
else:
    for col in ["RS_Score", "RS_Rank", "RS_Label", "DownDayRS",
                "DownDayWinRate", "RecentDDExcess", "EarlyBottom", "DaysSinceLow",
                "RS_1m", "RS_3m"]:
        df_all[col] = None

# ── Four Pillars (the single scoring spine) ──
df_all["SetupStage"] = df_all.apply(calc_setup_stage, axis=1)
_pillar_results = df_all.apply(
    lambda r: calc_four_pillars(r, st.session_state.themes, spy_ret, df_etf, st_data, _ai_sentiment), axis=1
)
df_all["PillarTech"]    = _pillar_results.apply(lambda d: d["technical"])
df_all["PillarFund"]    = _pillar_results.apply(lambda d: d["fundamental"])
df_all["PillarTheme"]   = _pillar_results.apply(lambda d: d["thematic"])
df_all["PillarNarr"]    = _pillar_results.apply(lambda d: d["narrative"])
df_all["PillarAligned"] = _pillar_results.apply(lambda d: d["aligned"])

# Convexity Score derived FROM pillars (not parallel)
df_all["ConvexityScore"] = df_all.apply(
    lambda r: calc_convexity_score(r["PillarTech"], r["PillarFund"], r["PillarTheme"], r["PillarNarr"]),
    axis=1
)
# Backward-compat aliases (used in UI tables)
df_all["MomentumScore"] = df_all["PillarTech"]  # momentum absorbed into Technical
df_all["AsymmetryScore"] = df_all["ConvexityScore"]  # alias
df_all["VGScore"] = df_all["PillarFund"]  # VG absorbed into Fundamental

# ── Ensure all expected columns exist (graceful fallback when APIs fail) ──
_expected_cols = {
    # Price data
    "Price": None, "Chg": None, "Pos52": None, "Ret1m": None, "Ret3m": None,
    "Spark30": None, "RSI": None, "ATR_pct": None, "Beta": None,
    "vsMA50": None, "vsMA200": None,
    # Fundamental data
    "NumAnalysts": 0, "Consensus": None, "Upside": None, "AnalystUpside": None,
    "ShortPct": None, "FCF_yield": None, "ROE": None, "Revenue_Growth": None,
    "Rule40": None, "FCFPositive": False, "RevGrowthPct": None,
    "InsiderPct": 0, "InstitPct": None, "DaysToCover": None,
    "CashRunwayMonths": None, "PS_Current": None, "PS_HistPos": None,
    "PS_3yr_Min": None, "PS_3yr_Max": None, "PS_3yr_Avg": None,
    "EV_EBITDA": None, "GrossMargin": None, "FCFValue": None,
    "PostMktChg": None, "PreMktChg": None, "OvernightChg": None, "DayChgPct": None,
    "TargetLow": None, "TargetHigh": None, "TargetMean": None,
    "RelVol": None,
    # Extras
    "NextEarnings": None, "InsiderSignal": None, "InsiderNet": 0,
    "InsiderBuys": 0, "Headlines": None,
    # Sentiment
    "ST_BullPct": None, "ST_MsgCount": 0,
    "AI_HeadlineScore": None, "AI_HeadlineSummary": "",
    # Relative strength
    "RS_Score": None, "RS_Rank": None, "RS_Label": None,
    "DownDayRS": None, "DownDayWinRate": None, "RecentDDExcess": None,
    "EarlyBottom": None, "DaysSinceLow": None, "RS_1m": None, "RS_3m": None,
    # Scores (in case scoring fails)
    "ConvexityScore": 0, "MomentumScore": 0, "AsymmetryScore": 0, "VGScore": 0,
    "SetupStage": "Unknown", "PillarTech": 0, "PillarFund": 0,
    "PillarTheme": 0, "PillarNarr": 0, "PillarAligned": False,
    # Theme
    "Theme": None,
}
for _col, _default in _expected_cols.items():
    if _col not in df_all.columns:
        df_all[_col] = _default

# Auto-assign themes to each ticker
df_all["Theme"] = df_all["Ticker"].apply(
    lambda t: ", ".join(get_ticker_themes(st.session_state.themes, t))
)

# Execution window from portfolio RS data
exec_window = calc_execution_window(df_all)

# Log scores for history tracking
log_scores(df_all)

# Failed tickers warning
_fetched = set(df_price["Ticker"].tolist())
_failed  = [t for t in st.session_state.tickers if t not in _fetched]
if _failed:
    st.warning(f"Failed to fetch data for: **{', '.join(_failed)}**. These tickers are excluded from analysis.")
if len(_fetched) == 0:
    st.error(
        "⚠️ **No data loaded.** Yahoo Finance is rate-limiting this server.  \n"
        "This usually happens on Streamlit Cloud's shared infrastructure.  \n\n"
        "**Options:**  \n"
        "1. Wait a few minutes and refresh  \n"
        "2. Run locally: `streamlit run portfolio_app.py`  \n"
        "3. Clone from [GitHub](https://github.com/Kmaddx/convexity-terminal)"
    )
    st.stop()

# Data quality check — detect when fundamentals didn't load (rate-limited)
_fund_cols_check = ["RevGrowthPct", "GrossMargin", "Beta"]
_has_fund_data = any(
    df_all[col].notna().any() and (df_all[col] != 0).any()
    for col in _fund_cols_check if col in df_all.columns
)
_data_quality = "Full" if _has_fund_data else "Limited"

st.title("◣ Convexity Terminal")
_data_ts = datetime.now().strftime('%Y-%m-%d %H:%M')
st.caption(f"As of {_data_ts}  |  {len(df_price)} tickers")

# ── Fragment: Top bar ticker add (independent rerun) ────────────────────────────

@st.fragment
def _topbar_ticker_add_fragment():
    """Add ticker from top bar without full page reload."""
    new_t = st.text_input("Add ticker", placeholder="+ Add ticker",
                           key="top_add_ticker", label_visibility="collapsed").upper().strip()
    if new_t and new_t not in st.session_state.tickers:
        st.session_state.tickers.append(new_t)
        st.session_state.watchlists[st.session_state.active_watchlist] = st.session_state.tickers
        save_watchlists(st.session_state.watchlists)


# ── Top bar — compact controls ───────────────────────────────────────────────
_tb1, _tb2, _tb3, _tb4 = st.columns([2.5, 2, 2, 1])
with _tb1:
    chosen_wl = st.selectbox("Watchlist", wl_names, index=active_idx, key="wl_select",
                              label_visibility="collapsed")
    if chosen_wl != st.session_state.active_watchlist:
        st.session_state.active_watchlist = chosen_wl
        st.session_state.tickers = st.session_state.watchlists[chosen_wl]
        st.cache_data.clear()
        st.rerun()
with _tb2:
    _topbar_ticker_add_fragment()
with _tb3:
    deep_dive_ticker = st.selectbox("Deep Dive", ["--"] + sorted(st.session_state.tickers),
                                     key="deep_dive", label_visibility="collapsed")
with _tb4:
    if st.button("Refresh", use_container_width=True, key="btn_refresh_top"):
        st.cache_data.clear()
        st.rerun()

if not _has_fund_data:
    st.warning(
        "⚠️ **Limited data** — Yahoo Finance rate-limited fundamental data. "
        "Pillar scores may be uniform. Run locally for full analysis: "
        "`git clone https://github.com/Kmaddx/convexity-terminal && streamlit run portfolio_app.py`"
    )

# ── Watchlist Health Score ────────────────────────────────────────────────
n_total = len(df_all)
if n_total > 0:
    pct_rs_leader = (df_all["RS_Label"] == "Leader").sum() / n_total * 100 if "RS_Label" in df_all.columns else 0
    pct_trending = df_all["SetupStage"].isin(["Trending", "Emerging"]).sum() / n_total * 100
    pct_aligned = (df_all["PillarAligned"] == True).sum() / n_total * 100
    health_avg = (pct_rs_leader + pct_trending + pct_aligned) / 3
    if health_avg >= 40:
        health_label = "Strong environment -- lean in"
        health_color = "green"
    elif health_avg >= 20:
        health_label = "Selective -- be patient"
        health_color = "orange"
    else:
        health_label = "Weak tape -- preserve capital"
        health_color = "red"

    # Color for KPI values based on health
    def _kpi_color(pct):
        if pct >= 40: return "#2ecc71"
        if pct >= 20: return "#f39c12"
        return "#e74c3c"

    _health_colors = {"green": "#2ecc71", "orange": "#f39c12", "red": "#e74c3c"}
    _hc = _health_colors.get(health_color, "#e6edf3")

    st.markdown(
        f"""<div style="display:flex;gap:12px;margin:4px 0 8px 0;">
        <div class="kpi-card" style="flex:1;">
            <div class="kpi-label">RS Leaders</div>
            <div class="kpi-value" style="color:{_kpi_color(pct_rs_leader)};">{pct_rs_leader:.0f}%</div>
            <div class="kpi-sub">{int((df_all["RS_Label"] == "Leader").sum()) if "RS_Label" in df_all.columns else 0} of {n_total}</div>
        </div>
        <div class="kpi-card" style="flex:1;">
            <div class="kpi-label">Trending / Emerging</div>
            <div class="kpi-value" style="color:{_kpi_color(pct_trending)};">{pct_trending:.0f}%</div>
            <div class="kpi-sub">{int(df_all["SetupStage"].isin(["Trending", "Emerging"]).sum())} of {n_total}</div>
        </div>
        <div class="kpi-card" style="flex:1;">
            <div class="kpi-label">Four-Pillar Aligned</div>
            <div class="kpi-value" style="color:{_kpi_color(pct_aligned)};">{pct_aligned:.0f}%</div>
            <div class="kpi-sub">{int((df_all["PillarAligned"] == True).sum())} of {n_total}</div>
        </div>
        <div class="kpi-card" style="flex:1.3;">
            <div class="kpi-label">Market Read</div>
            <div class="kpi-value" style="color:{_hc};font-size:1.1rem;">{health_label}</div>
            <div class="kpi-sub">Health score: {health_avg:.0f}</div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Single-Ticker Deep Dive (above tabs) ─────────────────────────────────────

if deep_dive_ticker != "--" and deep_dive_ticker in df_all["Ticker"].values:
    t_row = df_all[df_all["Ticker"] == deep_dive_ticker].iloc[0]
    theme = ", ".join(get_ticker_themes(st.session_state.themes, deep_dive_ticker)) or "Untagged"

    st.subheader(f"Deep Dive: {deep_dive_ticker}")
    dd1, dd2, dd3, dd4, dd5 = st.columns(5)
    dd1.metric("Price", f"${t_row['Price']}")
    dd2.metric("Convexity", f"{t_row['ConvexityScore']:.0f}/100")
    dd3.metric("Momentum", f"{t_row['MomentumScore']:.0f}/100")
    dd4.metric("RSI", f"{t_row['RSI']:.0f}", rsi_label(t_row['RSI']))
    dd5.metric("52wk Pos", f"{t_row['Pos52']:.0f}%")

    dd6, dd7, dd8, dd9, dd10 = st.columns(5)
    dd6.metric("Theme", theme)
    _beta_val = _safe(t_row.get('Beta'))
    _beta_reliable = t_row.get('BetaReliable', True)
    _beta_label = f"{_beta_val:.2f}" if _beta_val else "N/A"
    if not _beta_reliable:
        _beta_label = f"⚠️ {_beta_label}"
    dd7.metric("Beta", _beta_label, help="⚠️ Beta may be unreliable (missing, negative, or >5). Fallback of 1.5 used in scoring." if not _beta_reliable else None)
    dd8.metric("P/S", f"{_safe(t_row.get('PS_Current')):.1f}x" if _safe(t_row.get('PS_Current')) else "N/A")
    dd9.metric("Analyst Upside", f"{_safe(t_row.get('AnalystUpside')):+.0f}%" if _safe(t_row.get('AnalystUpside')) else "N/A")
    dd10.metric("Rev Growth", f"{_safe(t_row.get('RevGrowthPct')):+.0f}%" if _safe(t_row.get('RevGrowthPct')) else "N/A")

    dd11, dd12, dd13, dd14, dd15, dd16 = st.columns(6)
    dd11.metric("ATR%", f"{t_row['ATR_pct']:.1f}%")
    dd12.metric("Short %", f"{_safe(t_row.get('ShortPct')):.1f}%" if _safe(t_row.get('ShortPct')) else "N/A")
    dd13.metric("FCF+", "Yes" if t_row.get("FCFPositive") else "No")
    dd14.metric("Rule of 40", f"{_safe(t_row.get('Rule40')):.0f}" if _safe(t_row.get('Rule40')) else "N/A")
    dd15.metric("Next Earnings", t_row.get("NextEarnings") or "N/A")
    _dd_bull = t_row.get("ST_BullPct")
    dd16.metric("Sentiment", f"{_dd_bull:.0f}% bull" if pd.notna(_dd_bull) and _dd_bull else "N/A")

    # Relative strength
    if spy_ret:
        rs_1m = round(_safe(t_row.get("Ret1m")) - spy_ret.get("1m", 0), 1)
        rs_3m = round(_safe(t_row.get("Ret3m")) - spy_ret.get("3m", 0), 1)
        rs_6m = round(_safe(t_row.get("Ret6m")) - spy_ret.get("6m", 0), 1)
        st.caption(f"vs SPY — 1m: {rs_1m:+.1f}%  |  3m: {rs_3m:+.1f}%  |  6m: {rs_6m:+.1f}%")

    # Four-pillar score breakdown
    with st.expander("Pillar Breakdown"):
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            st.markdown("**Technical**")
            st.markdown(f"- Score: **{_safe(t_row.get('PillarTech')):.0f}**/100")
            st.markdown(f"- RSI: {_safe(t_row.get('RSI')):.0f}")
            st.markdown(f"- vs MA50: {_safe(t_row.get('vsMA50')):+.1f}%")
            st.markdown(f"- 52wk Pos: {_safe(t_row.get('Pos52')):.0f}%")
            st.markdown(f"- DD Win Rate: {_safe(t_row.get('DownDayWinRate')):.0f}%")
        with bc2:
            st.markdown("**Fundamental**")
            st.markdown(f"- Score: **{_safe(t_row.get('PillarFund')):.0f}**/100")
            st.markdown(f"- Rev Growth: {_safe(t_row.get('RevGrowthPct')):+.0f}%")
            st.markdown(f"- Gross Margin: {_safe(t_row.get('GrossMargin')):.0f}%")
            st.markdown(f"- Insider %: {_safe(t_row.get('InsiderPct')):.1f}%")
            st.markdown(f"- FCF+: {'Yes' if t_row.get('FCFPositive') else 'No'}")
        with bc3:
            st.markdown("**Thematic**")
            st.markdown(f"- Score: **{_safe(t_row.get('PillarTheme')):.0f}**/100")
            st.markdown(f"- Theme: {theme}")
            st.markdown(f"- Short %: {_safe(t_row.get('ShortPct')):.1f}%")
        with bc4:
            st.markdown("**Narrative**")
            st.markdown(f"- Score: **{_safe(t_row.get('PillarNarr')):.0f}**/100")
            _dd_bull_bd = t_row.get("ST_BullPct")
            st.markdown(f"- Sentiment: {f'{_dd_bull_bd:.0f}% bull' if pd.notna(_dd_bull_bd) and _dd_bull_bd else 'N/A'}")
            _ai_hs = t_row.get("AI_HeadlineScore")
            if pd.notna(_ai_hs) and _ai_hs != 0:
                _ai_label = "bullish" if _ai_hs > 0.2 else ("bearish" if _ai_hs < -0.2 else "neutral")
                _ai_color = "green" if _ai_hs > 0.2 else ("red" if _ai_hs < -0.2 else "orange")
                st.markdown(f"- AI News: **:{_ai_color}[{_ai_label}]** ({_ai_hs:+.2f})")
            _ai_summary = t_row.get("AI_HeadlineSummary", "")
            if _ai_summary:
                st.caption(f"AI: {_ai_summary}")
            st.markdown(f"- Insider: {t_row.get('InsiderSignal', 'N/A')}")

    # Price chart for deep dive ticker
    try:
        dd_hist = yf.download(deep_dive_ticker, period="6mo", interval="1d",
                              progress=False, auto_adjust=True)
        if not dd_hist.empty:
            dd_close = dd_hist["Close"].squeeze()
            dd_vol = dd_hist["Volume"].squeeze()
            dd_ma50 = dd_close.rolling(50).mean()

            from plotly.subplots import make_subplots
            fig_dd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig_dd.add_trace(go.Scatter(
                x=dd_close.index, y=dd_close.values, mode="lines",
                name="Price", line=dict(color="#3498db", width=2),
            ), row=1, col=1)
            fig_dd.add_trace(go.Scatter(
                x=dd_ma50.index, y=dd_ma50.values, mode="lines",
                name="50-day MA", line=dict(color="#f39c12", width=1, dash="dash"),
            ), row=1, col=1)
            fig_dd.add_trace(go.Bar(
                x=dd_vol.index, y=dd_vol.values, name="Volume",
                marker_color="#555", opacity=0.5,
            ), row=2, col=1)
            fig_dd.update_layout(
                height=350, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=50, r=20, t=30, b=30), **DARK,
            )
            fig_dd.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor="#21262d")
            fig_dd.update_yaxes(title_text="Vol", row=2, col=1, gridcolor="#21262d")
            st.plotly_chart(fig_dd, use_container_width=True)
    except (AttributeError, ValueError, KeyError, TypeError):
        pass

    # Score history trend
    score_hist = load_score_history()
    if deep_dive_ticker in score_hist and len(score_hist[deep_dive_ticker]) > 1:
        with st.expander("Score History Trend"):
            sh = score_hist[deep_dive_ticker]
            sh_df = pd.DataFrame(sh)
            fig_sh = go.Figure()
            for col, color in [("convexity", "#9b59b6"), ("momentum", "#27ae60"),
                               ("asymmetry", "#3498db"), ("stool", "#f39c12")]:
                if col in sh_df.columns:
                    fig_sh.add_trace(go.Scatter(
                        x=sh_df["date"], y=sh_df[col], mode="lines+markers",
                        name=col.title(), line=dict(color=color, width=2),
                        marker=dict(size=5),
                    ))
            fig_sh.update_layout(
                height=280, yaxis_title="Score",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=50, r=20, t=30, b=30), **DARK,
            )
            st.plotly_chart(fig_sh, use_container_width=True)

    grad_divider()

# ── TABS (visible at top, right after KPI row) ──────────────────────────────
tab_dash, tab_conv, tab_themes, tab_str, tab_charts, tab_sig, tab_scan, tab_settings = st.tabs([
    "Dashboard", "Convexity", "Themes", "Strength", "Charts", "Signals", "Scanner", "Settings",
])

# Stage color map (used across multiple tabs)
_stage_colors = {
    "Basing": "#3498db", "Emerging": "#2ecc71", "Trending": "#27ae60",
    "Extended": "#e67e22", "Breaking Down": "#e74c3c", "Neutral": "#888",
}
def _color_stage(val):
    return f"color: {_stage_colors.get(val, '#888')}"
def _color_aligned(val):
    if val:
        return "background-color: #1a3a2a; color: #2ecc71"
    return "background-color: #5b2c2c; color: #ff6b6b"

with tab_dash:
    # ══════════════════════════════════════════════════════════════════════════
    # MARKET ENVIRONMENT — "Should I Be Trading?"
    # ══════════════════════════════════════════════════════════════════════════
    section_header("Market Conditions", "#58a6ff")

    # AI Market Summary — styled to match app aesthetic
    if _market_ai_summary:
        import re as _re
        _summary_text = _market_ai_summary.strip()
        # Parse sections from the AI output
        _sections = {}
        _current_key = None
        _current_lines = []
        for _line in _summary_text.split("\n"):
            _stripped = _line.strip()
            # Detect section headers (e.g. "**MARKET READ**", "## MARKET READ", "1. **KEY CATALYSTS**")
            _header_match = _re.match(
                r'^(?:#+ *|\d+\.\s*)?(?:\*\*)?([A-Z][A-Z &/\-]+?)(?:\*\*)?:?\s*$', _stripped)
            if _header_match:
                if _current_key:
                    _sections[_current_key] = "\n".join(_current_lines).strip()
                _current_key = _header_match.group(1).strip()
                _current_lines = []
            elif _current_key is not None:
                _current_lines.append(_line)
            else:
                # Content before any header — treat as market read
                _current_lines.append(_line)
        if _current_key:
            _sections[_current_key] = "\n".join(_current_lines).strip()
        elif _current_lines:
            _sections["MARKET READ"] = "\n".join(_current_lines).strip()

        # Render styled HTML
        _section_icons = {}
        _ai_html = (
            '<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;'
            'padding:16px 20px;margin-bottom:12px;">'
            '<div style="font-size:0.65rem;color:#8b949e;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:10px;">AI Market Intelligence</div>'
        )
        for _sname, _sbody in _sections.items():
            _icon = _section_icons.get(_sname, "")
            # Convert markdown bold to HTML bold and list items
            _body_html = _sbody.replace("**", "<b>", 1)
            while "**" in _body_html:
                _body_html = _body_html.replace("**", "</b>", 1).replace("**", "<b>", 1)
            _body_html = _body_html.replace("\n", "<br>")
            _ai_html += (
                f'<div style="margin-bottom:12px;">'
                f'<div style="font-size:0.75rem;font-weight:600;color:#58a6ff;'
                f'margin-bottom:4px;">{(_icon + " ") if _icon else ""}{_sname}</div>'
                f'<div style="font-size:0.8rem;color:#c9d1d9;line-height:1.5;">'
                f'{_body_html}</div></div>'
            )
        _ai_html += '</div>'
        st.markdown(_ai_html, unsafe_allow_html=True)

    # Decision badge + total score
    _env_colors = {"OPPORTUNITY": "#2ecc71", "SELECTIVE": "#f39c12", "DEFENSIVE": "#e74c3c"}
    _env_bg = {"OPPORTUNITY": "#0d2818", "SELECTIVE": "#2d2a0d", "DEFENSIVE": "#2d0d0d"}
    _dec_color = _env_colors.get(env_decision, "#e6edf3")
    _dec_bg = _env_bg.get(env_decision, "#161b22")

    # Build a single row of 7 cards: Decision, Score, then each pillar
    def _env_card(label, value, subtitle, color="#e6edf3", bg="#161b22", border="#30363d"):
        return (
            f'<div style="background:{bg};border:1px solid {border};border-radius:8px;'
            f'padding:10px 12px;text-align:center;flex:1;min-width:0;">'
            f'<div style="font-family:\'Inter\',sans-serif;font-size:0.68rem;color:#8b949e;'
            f'text-transform:uppercase;letter-spacing:0.5px;">{label}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.3rem;'
            f'font-weight:700;color:{color};line-height:1.3;">{value}</div>'
            f'<div style="font-family:\'Inter\',sans-serif;font-size:0.6rem;color:#6e7681;">'
            f'{subtitle}</div></div>'
        )

    def _pillar_color(score):
        if score >= 60: return "#2ecc71"
        if score >= 40: return "#f39c12"
        return "#e74c3c"

    _env_cards_html = '<div style="display:flex;gap:8px;margin:6px 0 10px 0;">'
    # Decision card
    _env_cards_html += _env_card("Market Posture", env_decision, env_decision_sub,
                                  color=_dec_color, bg=_dec_bg, border=_dec_color)
    # Score card
    _env_cards_html += _env_card("Score", str(env_total), "/ 100", color=_dec_color)
    # Pillar cards
    for pname, pdata in env_pillars.items():
        _ps = pdata["score"]
        _pw = pdata["weight"]
        _pc = _pillar_color(_ps)
        _env_cards_html += _env_card(pname, str(_ps), f"wt: {_pw}%", color=_pc)
    _env_cards_html += '</div>'
    st.markdown(_env_cards_html, unsafe_allow_html=True)

    # ── Portfolio-level signals ──
    _portfolio_warnings = list(env_warnings)
    _portfolio_tailwinds = list(env_tailwinds)

    # Leaders rolling over: top-5 convexity tickers with negative 1m return
    _top5 = df_all.nlargest(5, "ConvexityScore")
    _leaders_down = _top5[_top5["Ret1m"].apply(lambda x: _safe(x, 0) < -5)]
    if len(_leaders_down) >= 2:
        _ld_names = ", ".join(_leaders_down["Ticker"].tolist())
        _portfolio_warnings.append(f"Leaders rolling over — {_ld_names} down >5% in 1m")
    elif len(_leaders_down) == 0:
        _leaders_up = _top5[_top5["Ret1m"].apply(lambda x: _safe(x, 0) > 0)]
        if len(_leaders_up) >= 3:
            _portfolio_tailwinds.append(f"Leaders holding — {len(_leaders_up)}/5 top convexity picks positive 1m")

    # Failing breakouts vs working breakouts
    _near_high = df_all[df_all["Pos52"].apply(lambda x: _safe(x) >= 85)]
    _failed_bo = _near_high[_near_high["Ret1m"].apply(lambda x: _safe(x, 0) < -3)]
    _working_bo = _near_high[_near_high["Ret1m"].apply(lambda x: _safe(x, 0) > 0)]
    if len(_failed_bo) >= 1:
        _fb_names = ", ".join(_failed_bo["Ticker"].tolist())
        _portfolio_warnings.append(f"Failing breakouts — {_fb_names} near highs but fading")
    if len(_working_bo) >= 2:
        _wb_names = ", ".join(_working_bo["Ticker"].tolist())
        _portfolio_tailwinds.append(f"Breakouts working — {_wb_names} near highs and extending")

    # Expandable detail panel
    with st.expander("Environment Detail"):
        det_c1, det_c2, det_c3 = st.columns(3)

        with det_c1:
            st.markdown("**Volatility**")
            vix_trend = "Rising" if market_env.get("vix_rising") else "Falling"
            vix_trend_color = "red" if market_env.get("vix_rising") else "green"
            st.markdown(f"- VIX: **{market_env.get('vix_level', 'N/A')}** "
                        f"(vs 20d MA: {market_env.get('vix_ma20', 'N/A')})")
            st.markdown(f"- VIX Trend: **:{vix_trend_color}[{vix_trend}]**")
            _vix_pct = market_env.get('vix_pct_rank')
            st.markdown(f"- VIX 1Y Percentile: **{f'{_vix_pct:.0f}th' if isinstance(_vix_pct, (int, float)) else 'N/A'}**")

            st.markdown("**Trend**")
            def _trend_label(val):
                if val is None: return "N/A", "gray"
                return (f"{val:+.1f}%", "green" if val > 0 else "red")
            for label, key in [("SPX vs 20d", "spx_vs_20d"), ("SPX vs 50d", "spx_vs_50d"), ("SPX vs 200d", "spx_vs_200d")]:
                v, c = _trend_label(market_env.get(key))
                st.markdown(f"- {label}: **:{c}[{v}]**")
            qqq_v, qqq_c = _trend_label(market_env.get("qqq_vs_50d"))
            st.markdown(f"- QQQ vs 50d: **:{qqq_c}[{qqq_v}]**")

        with det_c2:
            st.markdown("**Breadth**")
            st.markdown(f"- Sectors > 50d MA: **{market_env.get('sectors_above_50d', '?')}** / 11")
            st.markdown(f"- Sectors positive today: **{market_env.get('sectors_positive_1d', '?')}** / 11")
            if market_env.get("sector_leader"):
                st.markdown(f"- Leader (5d): **{market_env.get('sector_leader')}**")
            if market_env.get("sector_laggard"):
                st.markdown(f"- Laggard (5d): **{market_env.get('sector_laggard')}**")

            st.markdown("**Macro**")
            tnx_trend = "Rising" if market_env.get("tnx_rising") else "Falling"
            tnx_color = "red" if market_env.get("tnx_rising") else "green"
            st.markdown(f"- 10Y Yield: **{market_env.get('tnx_yield', 'N/A')}%** "
                        f"(:**{tnx_color}[{tnx_trend}]**)")
            dxy_trend = "Strengthening" if market_env.get("dxy_strengthening") else "Weakening"
            dxy_color = "red" if market_env.get("dxy_strengthening") else "green"
            st.markdown(f"- DXY: **{market_env.get('dxy_level', 'N/A')}** "
                        f"(:**{dxy_color}[{dxy_trend}]**)")

            st.markdown("**Internals**")
            _dist = market_env.get("distribution_days")
            if _dist is not None:
                _dc = "red" if _dist >= 5 else ("orange" if _dist >= 3 else "green")
                st.markdown(f"- Distribution days (25d): **:{_dc}[{_dist}]**")
            _uvdr = market_env.get("up_down_vol_ratio")
            if _uvdr is not None:
                _vc = "green" if _uvdr >= 1.1 else ("red" if _uvdr < 0.9 else "orange")
                st.markdown(f"- Up/Down vol ratio: **:{_vc}[{_uvdr:.2f}x]**")
            _fd = market_env.get("fade_days_10d")
            if _fd is not None:
                _fc = "red" if _fd >= 5 else ("orange" if _fd >= 3 else "green")
                st.markdown(f"- Fade days (10d): **:{_fc}[{_fd}]**")

        with det_c3:
            st.markdown("**Sector Performance (5d)**")
            sectors = market_env.get("sectors", {})
            if sectors:
                sorted_sectors = sorted(sectors.items(), key=lambda x: x[1]["ret_5d"], reverse=True)
                for etf, data in sorted_sectors:
                    r5 = data["ret_5d"]
                    sc = "green" if r5 > 0 else "red"
                    st.markdown(f"- {etf}: **:{sc}[{r5:+.1f}%]**")

        # ── Headwinds & Tailwinds ──
        if _portfolio_warnings or _portfolio_tailwinds:
            st.markdown("---")
            _hw_col, _tw_col = st.columns(2)
            with _hw_col:
                st.markdown(f"**Headwinds** ({len(_portfolio_warnings)})")
                if _portfolio_warnings:
                    for _w in _portfolio_warnings:
                        st.markdown(f"- :red[{_w}]")
                else:
                    st.markdown("- :green[None — clear skies]")
            with _tw_col:
                st.markdown(f"**Tailwinds** ({len(_portfolio_tailwinds)})")
                if _portfolio_tailwinds:
                    for _t in _portfolio_tailwinds:
                        st.markdown(f"- :green[{_t}]")
                else:
                    st.markdown("- :orange[None — no wind at your back]")

    st.caption(
        "**65+** = OPPORTUNITY (lean in) · **45-64** = SELECTIVE (quality setups only) · "
        "**<45** = DEFENSIVE (preserve capital)"
    )
    grad_divider()

    # ── Active Alerts ─────────────────────────────────────────────────────────
    section_header("Alerts", "#f39c12")
    overbought = df_price[df_price["RSI"] >= 70]["Ticker"].tolist()
    oversold   = df_price[df_price["RSI"] <= 30]["Ticker"].tolist()
    breakouts  = df_price[df_price["Breakout"] == True]["Ticker"].tolist()
    vol_spikes = df_price[df_price["RelVol"] >= 2.0]["Ticker"].tolist()

    _has_alerts = False
    if oversold:
        alert_card(f"<strong>Oversold</strong> (RSI &le; 30): {', '.join(oversold)} &mdash; potential entry", "green")
        _has_alerts = True
    if overbought:
        alert_card(f"<strong>Overbought</strong> (RSI &ge; 70): {', '.join(overbought)} &mdash; consider trimming", "red")
        _has_alerts = True
    if breakouts:
        alert_card(f"<strong>52wk Breakout</strong> (&ge; 95%): {', '.join(breakouts)} &mdash; momentum confirmation", "blue")
        _has_alerts = True
    if vol_spikes:
        alert_card(f"<strong>Volume Spike</strong> (&ge; 2x avg): {', '.join(vol_spikes)} &mdash; unusual activity", "amber")
        _has_alerts = True
    insider_buyers = df_all[df_all["InsiderSignal"] == "Buying"]["Ticker"].tolist()
    if insider_buyers:
        alert_card(f"<strong>Insider Buying</strong> (last 90 days): {', '.join(insider_buyers)}", "green")
        _has_alerts = True
    # Founder-level insider ownership
    high_insider = df_all[df_all["InsiderPct"].notna() & (df_all["InsiderPct"] >= 10)][["Ticker", "InsiderPct"]].to_dict("records")
    if high_insider:
        hi_str = ", ".join([f"{r['Ticker']} ({r['InsiderPct']:.0f}%)" for r in high_insider])
        alert_card(f"<strong>High Insider Ownership</strong> (&ge;10%): {hi_str} &mdash; founder alignment", "green")
        _has_alerts = True
    low_insider = df_all[df_all["InsiderPct"].notna() & (df_all["InsiderPct"] < 1) & (df_all["InsiderPct"] > 0)]["Ticker"].tolist()
    if low_insider:
        alert_card(f"<strong>Low Insider Ownership</strong> (&lt;1%): {', '.join(low_insider)} &mdash; limited skin in game", "amber")
        _has_alerts = True
    rs_leaders = df_all[df_all["RS_Label"] == "Leader"]["Ticker"].tolist() if "RS_Label" in df_all.columns else []
    early_bots = df_all[df_all["EarlyBottom"] == True]["Ticker"].tolist() if "EarlyBottom" in df_all.columns else []
    if rs_leaders:
        alert_card(f"<strong>RS Leaders</strong> (strongest on down-days): {', '.join(rs_leaders)}", "green")
        _has_alerts = True
    if early_bots:
        eb_not_leaders = [t for t in early_bots if t not in rs_leaders]
        if eb_not_leaders:
            alert_card(f"<strong>Early Bottoming</strong> (low before SPY): {', '.join(eb_not_leaders)}", "blue")
            _has_alerts = True

    # Character Change Detection
    _score_history = load_score_history()
    if _score_history:
        for _, row in df_all.iterrows():
            t = row["Ticker"]
            hist = _score_history.get(t, [])
            if len(hist) < 2:
                continue
            current_rs = str(row.get("RS_Label", ""))
            current_stage = str(row.get("SetupStage", ""))
            past_entries = [h for h in hist[:-1] if h.get("rs_label")][-5:]
            past_stage_entries = [h for h in hist[:-1] if h.get("setup_stage")][-5:]
            if current_rs in ("Holding", "Lagging") and past_entries:
                prev_rs_labels = [h["rs_label"] for h in past_entries]
                if "Leader" in prev_rs_labels:
                    alert_card(f"<strong>RS Downgrade</strong> {t}: was Leader, now {current_rs}", "red")
                    _has_alerts = True
            if current_stage in ("Basing", "Breaking Down") and past_stage_entries:
                prev_stages = [h["setup_stage"] for h in past_stage_entries]
                if "Trending" in prev_stages:
                    alert_card(f"<strong>Stage Degradation</strong> {t}: was Trending, now {current_stage}", "amber")
                    _has_alerts = True

    # Upcoming earnings within 7 days as alert
    _earn_check = df_all[df_all["NextEarnings"].notna() & (df_all["NextEarnings"] != "")].copy()
    if not _earn_check.empty:
        _earn_check["_ed"] = pd.to_datetime(_earn_check["NextEarnings"], errors="coerce")
        _today_dt = pd.Timestamp.now().normalize()
        _earn_soon = _earn_check[(_earn_check["_ed"] >= _today_dt) & (_earn_check["_ed"] <= _today_dt + pd.Timedelta(days=7))]
        if not _earn_soon.empty:
            _earn_tickers = [f"{r['Ticker']} ({r['NextEarnings']})" for _, r in _earn_soon.iterrows()]
            alert_card(f"<strong>Earnings This Week</strong>: {', '.join(_earn_tickers)}", "amber")
            _has_alerts = True

    # Extreme sentiment alerts
    if "ST_BullPct" in df_all.columns:
        _very_bullish = df_all[df_all["ST_BullPct"].notna() & (df_all["ST_BullPct"] >= 80)]["Ticker"].tolist()
        _very_bearish = df_all[df_all["ST_BullPct"].notna() & (df_all["ST_BullPct"] <= 25)]["Ticker"].tolist()
        if _very_bullish:
            alert_card(f"<strong>Extreme Bullish Sentiment</strong> (&ge;80% bull): {', '.join(_very_bullish)} &mdash; crowded trade risk", "amber")
            _has_alerts = True
        if _very_bearish:
            alert_card(f"<strong>Extreme Bearish Sentiment</strong> (&le;25% bull): {', '.join(_very_bearish)} &mdash; contrarian opportunity?", "green")
            _has_alerts = True

    if not _has_alerts:
        st.info("No active alerts. Portfolio is in a neutral zone.")

    grad_divider()

    # ── Portfolio Overview ─────────────────────────────────────────────────────
    section_header("Portfolio Overview", "#58a6ff")

    overview_df = df_all.copy()
    overview_df["Theme"] = overview_df["Ticker"].apply(
        lambda t: ", ".join(get_ticker_themes(st.session_state.themes, t)) or "")

    # Four-pillar score: average of the four pillars (more granular than yes/no alignment)
    overview_df["FourPillar"] = overview_df.apply(
        lambda r: round((_safe(r.get("PillarTech")) + _safe(r.get("PillarFund")) +
                         _safe(r.get("PillarTheme")) + _safe(r.get("PillarNarr"))) / 4, 0), axis=1)

    overview_cols = ["Ticker", "Price", "DayChgPct", "OvernightChg", "Spark30", "Ret1m",
                     "SetupStage", "RS_Label",
                     "FourPillar", "ConvexityScore", "MomentumScore", "RSI", "Pos52",
                     "ST_BullPct", "AnalystUpside", "RevGrowthPct", "InsiderPct", "PS_Current", "ShortPct", "Theme"]
    # Ensure required columns exist
    for _oc in ["RS_Label", "Spark30", "OvernightChg", "DayChgPct", "ST_BullPct", "Ret1m"]:
        if _oc not in overview_df.columns:
            overview_df[_oc] = None
    overview_disp = overview_df[[c for c in overview_cols if c in overview_df.columns]].copy()
    overview_disp = overview_disp.rename(columns={
        "DayChgPct": "Day %", "OvernightChg": "Ext Hrs %", "Spark30": "30d", "Ret1m": "30d %",
        "SetupStage": "Stage", "RS_Label": "RS",
        "FourPillar": "4-Pillar", "ConvexityScore": "Convexity", "MomentumScore": "Momentum",
        "Pos52": "52wk Pos", "ST_BullPct": "Sentiment",
        "AnalystUpside": "Upside %",
        "RevGrowthPct": "Rev Growth %", "InsiderPct": "Insider %", "PS_Current": "P/S", "ShortPct": "Short %",
    })
    overview_disp = overview_disp.sort_values("Convexity", ascending=False).reset_index(drop=True)

    # ── Color-coding helper ──
    def _color_pct(val):
        """Green/red for percentage columns."""
        if pd.isna(val): return ""
        if val > 0: return "color: #2ecc71"
        if val < 0: return "color: #ff6b6b"
        return ""

    def _color_score(val):
        """Green/amber/red for score columns (0–100)."""
        if pd.isna(val): return ""
        if val >= 65: return "background-color: #1a3a2a; color: #2ecc71"
        if val >= 45: return "background-color: #2a2a1a; color: #f39c12"
        return "background-color: #3a1a1a; color: #ff6b6b"

    def _color_rsi(val):
        """RSI color: overbought/oversold."""
        if pd.isna(val): return ""
        if val >= 70: return "color: #ff6b6b"
        if val <= 30: return "color: #2ecc71"
        return ""

    pct_cols = [c for c in ["Day %", "Ext Hrs %", "30d %", "Upside %", "Rev Growth %"] if c in overview_disp.columns]
    score_cols = [c for c in ["4-Pillar", "Convexity", "Momentum"] if c in overview_disp.columns]

    overview_styled = (overview_disp.style
        .map(_color_pct, subset=pct_cols)
        .map(_color_score, subset=score_cols)
        .map(_color_rsi, subset=["RSI"] if "RSI" in overview_disp.columns else [])
        .format({
            "Price": "${:.2f}",
            "Day %": lambda x: f"{x:+.2f}%" if pd.notna(x) else "",
            "Ext Hrs %": lambda x: f"{x:+.2f}%" if pd.notna(x) else "",
            "30d %": lambda x: f"{x:+.1f}%" if pd.notna(x) else "",
            "RSI": lambda x: f"{x:.0f}" if pd.notna(x) else "",
            "52wk Pos": lambda x: f"{x:.0f}" if pd.notna(x) else "",
            "4-Pillar": lambda x: f"{x:.0f}" if pd.notna(x) else "",
            "Convexity": lambda x: f"{x:.0f}" if pd.notna(x) else "",
            "Momentum": lambda x: f"{x:.0f}" if pd.notna(x) else "",
            "Sentiment": lambda x: f"{x:.0f}%" if pd.notna(x) else "",
            "Upside %": lambda x: f"{x:+.0f}%" if pd.notna(x) else "",
            "Rev Growth %": lambda x: f"{x:+.0f}%" if pd.notna(x) else "",
            "Insider %": lambda x: f"{x:.1f}%" if pd.notna(x) else "",
            "P/S": lambda x: f"{x:.1f}" if pd.notna(x) else "",
            "Short %": lambda x: f"{x:.1f}%" if pd.notna(x) else "",
        }, na_rep="")
    )

    _overview_col_config = {
        "30d": st.column_config.LineChartColumn("30d Trend", width="small"),
    }
    st.dataframe(
        overview_styled,
        column_config=_overview_col_config,
        use_container_width=True, hide_index=True,
        height=min(800, 38 * len(overview_disp) + 40),
    )
    st.download_button("Export Overview CSV",
                       overview_disp.drop(columns=["30d"], errors="ignore").to_csv(index=False),
                       "portfolio_overview.csv", "text/csv", key="dl_overview")

    # ── Portfolio Map — RSI vs 52-Week Position ──
    grad_divider()
    section_header("Portfolio Map — RSI vs 52-Week Position", "#f39c12")
    st.caption("Bubble size = ATR volatility. Bottom-left = weak. Top-right = strong. Bottom-right = pullback opportunity.")
    fig5 = go.Figure()
    df_price["vsMA200_str"] = df_price["vsMA200"].apply(lambda x: f"{x:+.1f}%" if x is not None else "N/A")
    df_price.apply(lambda row: fig5.add_trace(go.Scatter(
        x=[row["Pos52"]], y=[row["RSI"]],
        mode="markers+text", text=[row["Ticker"]], textposition="top center",
        marker=dict(size=row["ATR_pct"]*4, color=pos_color(row["Pos52"]),
                    line=dict(width=1, color="#0d1117")),
        hovertemplate=(
            f"<b>{row['Ticker']}</b><br>Price: ${row['Price']}<br>"
            f"RSI: {row['RSI']}<br>52wk Pos: {row['Pos52']:.1f}%<br>"
            f"ATR%: {row['ATR_pct']}<br>vs MA50: {row['vsMA50']:+.1f}%<br>"
            f"vs MA200: {row['vsMA200_str']}<extra></extra>"
        ),
        showlegend=False,
    )), axis=1)
    fig5.add_hline(y=70, line_dash="dash", line_color="#e74c3c", line_width=0.8,
                   annotation_text="Overbought", annotation_font_color="#e74c3c")
    fig5.add_hline(y=30, line_dash="dash", line_color="#2ecc71", line_width=0.8,
                   annotation_text="Oversold", annotation_font_color="#2ecc71")
    fig5.add_vline(x=50, line_dash="dot", line_color="#555", line_width=0.8)
    for (x, y, lbl) in [(82,78,"STRONG"),(12,78,"RECOVERING"),(82,25,"PULLBACK"),(12,25,"WEAK")]:
        fig5.add_annotation(x=x, y=y, text=lbl, showarrow=False,
                            font=dict(color="#555", size=11, family="monospace"))
    fig5.update_layout(
        height=500,
        xaxis=dict(range=[-5,105], title="52-Week Range Position (%)", gridcolor="#21262d"),
        yaxis=dict(range=[15,90], title="RSI (14)", gridcolor="#21262d"),
        margin=dict(l=60,r=20,t=20,b=60), **DARK,
    )
    st.plotly_chart(fig5, use_container_width=True)

    grad_divider()
    st.caption(f"Data as of {_data_ts}")

# ── TAB: Convexity ───────────────────────────────────────────────────────────

with tab_conv:
    # ── Four-Pillar Alignment (moved from Dashboard) ─────────────────────────
    section_header("Four-Pillar Alignment", "#58a6ff")
    st.caption("Technical + Fundamental + Thematic + Narrative — all four must score >= 50 for full alignment.")

    aligned_tickers = df_all[df_all["PillarAligned"] == True]
    aligned_count = len(aligned_tickers)
    if aligned_count > 0:
        st.success(f"**{aligned_count} fully aligned ticker{'s' if aligned_count != 1 else ''}** — all 4 pillars >= 50: "
                   f"**{', '.join(aligned_tickers['Ticker'].tolist())}**")
        for _, arow in aligned_tickers.sort_values("PillarTech", ascending=False).iterrows():
            stage = arow.get("SetupStage", "Neutral")
            st.markdown(
                f"**{arow['Ticker']}**  |  Stage: `{stage}`  |  "
                f"Tech **{arow['PillarTech']:.0f}**  |  Fund **{arow['PillarFund']:.0f}**  |  "
                f"Theme **{arow['PillarTheme']:.0f}**  |  Narr **{arow['PillarNarr']:.0f}**"
            )
    else:
        st.info("No tickers currently have all four pillars aligned (>= 50).")

    with st.expander("Four-Pillar Detail", expanded=True):
        pillar_df = df_all.copy()
        pillar_df["_strong_count"] = (
            (pillar_df["PillarTech"] >= 50).astype(int) +
            (pillar_df["PillarFund"] >= 50).astype(int) +
            (pillar_df["PillarTheme"] >= 50).astype(int) +
            (pillar_df["PillarNarr"] >= 50).astype(int)
        )
        pillar_df["_avg_pillar"] = (
            pillar_df["PillarTech"] + pillar_df["PillarFund"] +
            pillar_df["PillarTheme"] + pillar_df["PillarNarr"]
        ) / 4
        pillar_df = pillar_df.sort_values(["_strong_count", "_avg_pillar"], ascending=[False, False])
        pillar_df["4-Pillar"] = pillar_df["_avg_pillar"].round(0)
        pillar_disp = pillar_df[["Ticker", "Price", "SetupStage", "PillarTech", "PillarFund",
                                  "PillarTheme", "PillarNarr", "4-Pillar", "PillarAligned"]].copy()
        pillar_disp = pillar_disp.rename(columns={
            "SetupStage": "Stage", "PillarTech": "Technical", "PillarFund": "Fundamental",
            "PillarTheme": "Thematic", "PillarNarr": "Narrative", "PillarAligned": "Aligned",
        }).reset_index(drop=True)
        pillar_disp["Aligned"] = pillar_disp["Aligned"].apply(lambda x: "Yes" if x else "No")
        pillar_styled = (pillar_disp.style
            .background_gradient(subset=["Technical", "Fundamental", "Thematic", "Narrative", "4-Pillar"],
                                 cmap="RdYlGn", vmin=0, vmax=100)
            .map(_color_stage, subset=["Stage"])
            .map(lambda v: _color_aligned(v == "Yes"), subset=["Aligned"])
            .format({"Price": "${:.2f}", "Technical": "{:.0f}", "Fundamental": "{:.0f}",
                      "Thematic": "{:.0f}", "Narrative": "{:.0f}", "4-Pillar": "{:.0f}"}, na_rep="N/A")
        )
        st.dataframe(pillar_styled, width="stretch", hide_index=True)

    grad_divider()

    # ── Top Convexity Picks ──────────────────────────────────────────────────
    section_header("Top Convexity Picks", "#f1c40f")
    st.caption("Highest combined convexity score — the best risk/reward setups right now.")
    top5 = df_all.sort_values("ConvexityScore", ascending=False).head(5)
    for rank_i, (_, row) in enumerate(top5.iterrows(), 1):
        reasons = []
        ps_pos = _safe(row.get("PS_HistPos"))
        if ps_pos > 0 and ps_pos <= 30:
            reasons.append(f"compressed P/S ({ps_pos:.0f}% of 3yr range)")
        upside = _safe(row.get("AnalystUpside"))
        if upside > 20:
            reasons.append(f"+{upside:.0f}% analyst upside")
        rev = _safe(row.get("RevGrowthPct"))
        if rev > 15:
            reasons.append(f"{rev:+.0f}% rev growth")
        short = _safe(row.get("ShortPct"))
        if short > 8:
            reasons.append(f"{short:.0f}% short (squeeze fuel)")
        if not row.get("FCFPositive") and rev and rev > 20:
            reasons.append("pre-EBITDA flip")
        if _safe(row.get("RSI"), 50) <= 40:
            reasons.append(f"RSI oversold ({row['RSI']:.0f})")
        reason_str = " | ".join(reasons[:3]) if reasons else "balanced score across all factors"
        theme = ", ".join(get_ticker_themes(st.session_state.themes, row["Ticker"]))
        theme_tag = f"  `{theme}`" if theme else ""
        st.markdown(
            f"**#{rank_i}  {row['Ticker']}**{theme_tag}  —  "
            f"Convexity **{row['ConvexityScore']:.0f}**  |  ${row['Price']}  |  {reason_str}"
        )

    grad_divider()

    # ── Convexity Analysis ───────────────────────────────────────────────────
    section_header("Convexity Analysis", "#2ecc71")
    st.caption(
        "Two complementary lenses on asymmetric upside. "
        "**Value-Growth Score**: three-legged stool framework: undervalued + growing + trending (the foundation). "
        "**Asymmetry Score**: catalyst fuel — analyst upside, beta, short squeeze potential, volatility."
    )

    df_conv = df_all

    # Header metrics
    fcf_pos   = df_conv[df_conv["FCFPositive"] == True]["Ticker"].tolist()
    ps_lows   = df_conv[df_conv["PS_HistPos"].notna() & (df_conv["PS_HistPos"] <= 25)]["Ticker"].tolist()
    rule40_ok = df_conv[df_conv["Rule40"].notna() & (df_conv["Rule40"] >= 40)]["Ticker"].tolist()
    near_flip = df_conv[
        (df_conv["FCFPositive"] == False) &
        (df_conv["CashRunwayMonths"].isna() | (df_conv["CashRunwayMonths"] > 6)) &
        (df_conv["RevGrowthPct"].notna()) & (df_conv["RevGrowthPct"] > 20)
    ]["Ticker"].tolist()

    # "Priced for perfection" flag — P/S in top quartile of 3yr range AND RSI >= 65
    perfection = df_conv[
        df_conv["PS_HistPos"].notna() &
        (df_conv["PS_HistPos"] >= 75) &
        (df_conv["RSI"] >= 65)
    ]["Ticker"].tolist()

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("FCF Positive",         len(fcf_pos),   ", ".join(fcf_pos)   or "None")
    h2.metric("P/S at 3yr Low (<25%)", len(ps_lows),   ", ".join(ps_lows)   or "None")
    h3.metric("Rule of 40 (>=40)",    len(rule40_ok), ", ".join(rule40_ok) or "None")
    h4.metric("Near EBITDA Flip",     len(near_flip), ", ".join(near_flip) or "None",
              help="FCF negative but growing >20% revenue with >6 months runway")
    h5.metric("Priced for Perfection", len(perfection), ", ".join(perfection) or "None",
              delta_color="inverse",
              help="P/S in top 25% of 3yr range AND RSI >= 65 — dangerous setup, negative asymmetry")

    grad_divider()

    # Four-Pillar stacked bar chart
    st.markdown("#### Pillar Breakdown by Ticker")
    st.caption("Each bar shows the four pillar scores. Convexity = weighted sum (Tech 30% + Fund 30% + Theme 20% + Narr 20%).")

    pillar_sorted = df_conv.sort_values("ConvexityScore", ascending=False)
    pillar_names = ["PillarTech", "PillarFund", "PillarTheme", "PillarNarr"]
    pillar_labels = ["Technical", "Fundamental", "Thematic", "Narrative"]
    pillar_colors = ["#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    fig_pillars = go.Figure()
    for col, label, color in zip(pillar_names, pillar_labels, pillar_colors):
        fig_pillars.add_trace(go.Bar(
            name=label,
            x=pillar_sorted["Ticker"],
            y=pillar_sorted[col],
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.0f}}/100<extra></extra>",
        ))
    fig_pillars.update_layout(
        barmode="group", height=400,
        xaxis_title="", yaxis_title="Score (0-100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=50, b=40), **DARK,
    )
    fig_pillars.add_hline(y=50, line_dash="dash", line_color="#f39c12",
                          annotation_text="Alignment threshold", annotation_font_color="#f39c12")
    st.plotly_chart(fig_pillars, use_container_width=True)

    grad_divider()

    # Compressed Spring chart (P/S historical position)
    has_ps_hist = df_conv[df_conv["PS_HistPos"].notna()].sort_values("PS_HistPos")
    if not has_ps_hist.empty:
        st.markdown("#### Compressed Spring — P/S Position in 3-Year Range")
        st.caption(
            "How expensive is the stock today vs its own history? "
            "Left = historically cheap (compressed spring). Right = historically expensive."
        )
        fig_ps = go.Figure()
        for _, row in has_ps_hist.iterrows():
            color = "#2ecc71" if row["PS_HistPos"] <= 25 else "#f39c12" if row["PS_HistPos"] <= 60 else "#e74c3c"
            fig_ps.add_trace(go.Bar(
                x=[row["PS_HistPos"]], y=[row["Ticker"]],
                orientation="h", marker_color=color, showlegend=False,
                text=f"  {row['PS_HistPos']:.0f}%  |  {row['PS_Current']:.1f}x P/S  (range: {row['PS_3yr_Min']:.1f}–{row['PS_3yr_Max']:.1f}x)",
                textposition="outside",
                hovertemplate=(
                    f"<b>{row['Ticker']}</b><br>"
                    f"Current P/S: {row['PS_Current']:.2f}x<br>"
                    f"3yr Min: {row['PS_3yr_Min']:.2f}x<br>"
                    f"3yr Max: {row['PS_3yr_Max']:.2f}x<br>"
                    f"3yr Avg: {row['PS_3yr_Avg']:.2f}x<br>"
                    f"Position in range: {row['PS_HistPos']:.0f}%<extra></extra>"
                ),
            ))
        fig_ps.add_vline(x=25, line_dash="dash", line_color="#2ecc71",
                         annotation_text="Compressed zone", annotation_font_color="#2ecc71")
        fig_ps.update_layout(
            height=max(350, len(has_ps_hist) * 34),
            xaxis=dict(range=[0, 130], title="Position in 3-year P/S range (%)"),
            margin=dict(l=60, r=20, t=20, b=40), **DARK,
        )
        st.plotly_chart(fig_ps, use_container_width=True)
    else:
        st.info("Historical P/S data loading — check back shortly or refresh.")

    # Priced for Perfection warning
    if perfection:
        grad_divider()
        st.markdown("#### Priced for Perfection — Negative Asymmetry")
        st.warning(
            f"**{', '.join(perfection)}** — these stocks have P/S in the top 25% of their 3-year range "
            "AND RSI >= 65. This is the opposite of a compressed spring: expensive + overbought = "
            "downside risk outweighs upside. Consider trimming or tightening stops."
        )
        pfp_tbl = df_conv[df_conv["Ticker"].isin(perfection)][
            ["Ticker", "Price", "RSI", "PS_Current", "PS_HistPos", "PS_3yr_Min", "PS_3yr_Max"]
        ].copy()
        pfp_tbl["PS_Current"] = pfp_tbl["PS_Current"].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A")
        pfp_tbl["PS_HistPos"] = pfp_tbl["PS_HistPos"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A")
        pfp_tbl["PS_3yr_Min"] = pfp_tbl["PS_3yr_Min"].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "—")
        pfp_tbl["PS_3yr_Max"] = pfp_tbl["PS_3yr_Max"].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "—")
        pfp_tbl = pfp_tbl.rename(columns={
            "PS_Current": "P/S", "PS_HistPos": "P/S Position",
            "PS_3yr_Min": "3yr Low", "PS_3yr_Max": "3yr High",
        })
        st.dataframe(pfp_tbl, width="stretch", hide_index=True)

    grad_divider()

    # Valuation metrics table
    st.markdown("#### Valuation Metrics")
    val_tbl = df_conv[[
        "Ticker", "Price", "PS_Current", "PS_3yr_Min", "PS_3yr_Avg",
        "EV_EBITDA", "GrossMargin", "RevGrowthPct", "Rule40",
        "FCFPositive", "CashRunwayMonths", "InsiderPct", "InstitPct", "DaysToCover",
    ]].copy().sort_values("PS_Current")

    val_tbl["FCFPositive"] = val_tbl["FCFPositive"].apply(lambda x: "Yes" if x else "No")

    val_tbl = val_tbl.rename(columns={
        "PS_Current": "P/S", "PS_3yr_Min": "P/S 3yr Low", "PS_3yr_Avg": "P/S 3yr Avg",
        "EV_EBITDA": "EV/EBITDA", "GrossMargin": "Gross Margin",
        "RevGrowthPct": "Rev Growth", "FCFPositive": "FCF+",
        "CashRunwayMonths": "Runway", "InsiderPct": "Insider%",
        "InstitPct": "Instit%", "DaysToCover": "Days to Cover",
    })

    # Gradient columns (numeric) — lower P/S is better, higher Rev Growth / Rule40 is better
    val_grad_cols_green = [c for c in ["Rev Growth", "Rule40"] if c in val_tbl.columns and val_tbl[c].notna().any()]
    val_grad_cols_rev   = [c for c in ["P/S"] if c in val_tbl.columns and val_tbl[c].notna().any()]

    val_styled = val_tbl.style
    if val_grad_cols_green:
        val_styled = val_styled.background_gradient(subset=val_grad_cols_green, cmap="RdYlGn", vmin=-20, vmax=60)
    if val_grad_cols_rev:
        val_styled = val_styled.background_gradient(subset=val_grad_cols_rev, cmap="RdYlGn_r")
    val_styled = val_styled.format({
        "Price": "${:.2f}",
        "P/S": lambda x: f"{x:.1f}x" if pd.notna(x) and x else "N/A",
        "P/S 3yr Low": lambda x: f"{x:.1f}x" if pd.notna(x) and x else "—",
        "P/S 3yr Avg": lambda x: f"{x:.1f}x" if pd.notna(x) and x else "—",
        "EV/EBITDA": lambda x: f"{x:.1f}x" if pd.notna(x) and x else "N/A",
        "Gross Margin": lambda x: f"{x:.0f}%" if pd.notna(x) and x else "N/A",
        "Rev Growth": lambda x: f"{x:+.0f}%" if pd.notna(x) and x else "N/A",
        "Rule40": lambda x: f"{x:.0f}" if pd.notna(x) and x else "N/A",
        "Runway": lambda x: f"{x:.0f}mo" if pd.notna(x) and x else "N/A",
        "Insider%": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        "Instit%": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        "Days to Cover": lambda x: f"{x:.1f}d" if pd.notna(x) and x else "N/A",
    }, na_rep="N/A")
    st.dataframe(val_styled, width="stretch", hide_index=True)
    st.download_button("Export Valuation CSV", val_tbl.to_csv(index=False),
                       "valuation_metrics.csv", "text/csv", key="dl_val")

    grad_divider()

    # EBITDA Turn Watchlist
    st.markdown("#### EBITDA Turn Watchlist")
    st.caption(
        "Companies not yet FCF positive but growing fast with adequate runway — "
        "the 'pre-institutional green light' setup."
    )
    ebitda_flip = df_conv[
        (df_conv["FCFPositive"] == False) &
        df_conv["RevGrowthPct"].notna()
    ].sort_values("RevGrowthPct", ascending=False)[
        ["Ticker", "Price", "RevGrowthPct", "Rule40", "CashRunwayMonths", "PS_Current"]
    ].copy()
    ebitda_flip = ebitda_flip.rename(columns={
        "RevGrowthPct": "Rev Growth", "CashRunwayMonths": "Cash Runway", "PS_Current": "P/S",
    })
    if not ebitda_flip.empty:
        eb_grad_cols = [c for c in ["Rev Growth"] if ebitda_flip[c].notna().any()]
        eb_styled = ebitda_flip.style
        if eb_grad_cols:
            eb_styled = eb_styled.background_gradient(subset=eb_grad_cols, cmap="RdYlGn", vmin=-20, vmax=80)
        eb_styled = eb_styled.format({
            "Price": "${:.2f}",
            "Rev Growth": lambda x: f"{x:+.0f}%" if pd.notna(x) else "N/A",
            "Rule40": lambda x: f"{x:.0f}" if pd.notna(x) and x else "N/A",
            "Cash Runway": lambda x: f"{x:.0f} months" if pd.notna(x) and x else "N/A",
            "P/S": lambda x: f"{x:.1f}x" if pd.notna(x) and x else "N/A",
        }, na_rep="N/A")
        st.dataframe(eb_styled, width="stretch", hide_index=True)
    else:
        st.info("No pre-flip candidates in current selection.")

    grad_divider()

    # Rankings (merged from former Rankings tab)
    section_header("Convexity vs Momentum Rankings", "#f1c40f")
    st.caption(
        "**Convexity** (primary) = average of Asymmetry + Value-Growth scores — "
        "identifies the best risk/reward setups. "
        "**Momentum** (secondary) = stocks already trending."
    )

    df_ranked = df_all

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Convexity Ranking")
        conv = df_ranked.sort_values("ConvexityScore", ascending=False).reset_index(drop=True)
        conv.index += 1
        fig_conv = go.Figure(go.Bar(
            x=conv["ConvexityScore"], y=conv["Ticker"], orientation="h",
            marker_color=["#9b59b6" if s >= 60 else "#3498db" if s >= 40 else "#555"
                          for s in conv["ConvexityScore"]],
            text=conv["ConvexityScore"].apply(lambda x: f"{x:.0f}"), textposition="outside",
            hovertemplate="<b>%{y}</b><br>Convexity: %{x:.1f}<extra></extra>",
        ))
        fig_conv.add_vline(x=50, line_dash="dot", line_color="#555")
        fig_conv.update_layout(height=max(400,len(conv)*32),
                               xaxis=dict(range=[0,115], title="Score"),
                               yaxis=dict(autorange="reversed"),
                               margin=dict(l=60,r=20,t=20,b=40), showlegend=False, **DARK)
        st.plotly_chart(fig_conv, use_container_width=True)
        conv_tbl = conv[["Ticker","Price","ConvexityScore","PillarTech","PillarFund","PillarTheme","PillarNarr","AnalystUpside","NextEarnings"]].copy()
        conv_tbl["4-Pillar"] = ((conv_tbl["PillarTech"] + conv_tbl["PillarFund"] +
                                 conv_tbl["PillarTheme"] + conv_tbl["PillarNarr"]) / 4).round(0)
        conv_tbl["NextEarnings"] = conv_tbl["NextEarnings"].fillna("N/A")
        conv_tbl = conv_tbl.rename(columns={
            "ConvexityScore":"Convexity","PillarTech":"Tech","PillarFund":"Fund",
            "PillarTheme":"Theme","PillarNarr":"Narr",
            "AnalystUpside":"Upside","NextEarnings":"Next Earnings",
        })
        conv_tbl = conv_tbl[["Ticker","Price","Convexity","4-Pillar","Tech","Fund","Theme","Narr","Upside","Next Earnings"]]
        st.dataframe(conv_tbl, width="stretch", hide_index=False,
                     column_config={
                         "Convexity": st.column_config.ProgressColumn("Convexity", min_value=0, max_value=100, format="%d"),
                         "4-Pillar": st.column_config.ProgressColumn("4-Pillar", min_value=0, max_value=100, format="%d"),
                         "Tech": st.column_config.ProgressColumn("Tech", min_value=0, max_value=100, format="%d"),
                         "Fund": st.column_config.ProgressColumn("Fund", min_value=0, max_value=100, format="%d"),
                         "Theme": st.column_config.ProgressColumn("Theme", min_value=0, max_value=100, format="%d"),
                         "Narr": st.column_config.ProgressColumn("Narr", min_value=0, max_value=100, format="%d"),
                         "Upside": st.column_config.NumberColumn("Upside", format="%+.0f%%"),
                         "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                     })
        st.download_button("Export Convexity CSV", conv_tbl.to_csv(index=False),
                           "convexity_ranking.csv", "text/csv", key="dl_conv")

    with col_right:
        st.markdown("#### Momentum Ranking")
        mom = df_ranked.sort_values("MomentumScore", ascending=False).reset_index(drop=True)
        mom.index += 1
        fig_mom = go.Figure(go.Bar(
            x=mom["MomentumScore"], y=mom["Ticker"], orientation="h",
            marker_color=["#27ae60" if s >= 65 else "#f39c12" if s >= 45 else "#e74c3c"
                          for s in mom["MomentumScore"]],
            text=mom["MomentumScore"].apply(lambda x: f"{x:.0f}"), textposition="outside",
            hovertemplate="<b>%{y}</b><br>Momentum: %{x:.1f}<extra></extra>",
        ))
        fig_mom.add_vline(x=50, line_dash="dot", line_color="#555")
        fig_mom.update_layout(height=max(400,len(mom)*32),
                               xaxis=dict(range=[0,115], title="Score"),
                               yaxis=dict(autorange="reversed"),
                               margin=dict(l=60,r=20,t=20,b=40), showlegend=False, **DARK)
        st.plotly_chart(fig_mom, use_container_width=True)
        mom_tbl = mom[["Ticker","Price","RSI","Pos52","NextEarnings","MomentumScore"]].copy()
        mom_tbl["NextEarnings"] = mom_tbl["NextEarnings"].fillna("N/A")
        mom_tbl = mom_tbl.rename(columns={"Pos52":"52wk Pos","NextEarnings":"Next Earnings","MomentumScore":"Score"})
        st.dataframe(mom_tbl, width="stretch", hide_index=False,
                     column_config={
                         "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                         "52wk Pos": st.column_config.NumberColumn("52wk Pos", format="%.1f%%"),
                         "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
                         "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                     })
        st.download_button("Export Momentum CSV", mom_tbl.to_csv(index=False),
                           "momentum_ranking.csv", "text/csv", key="dl_mom")

    grad_divider()
    st.markdown("#### Re-rating Candidates")
    st.caption("Tickers ranked 3+ places higher on Convexity than Momentum — potential catalyst plays not yet priced in.")
    conv_rank = {row["Ticker"]: i+1 for i, row in conv.iterrows()}
    mom_rank  = {row["Ticker"]: i+1 for i, row in mom.iterrows()}
    divs = [{"Ticker": t,
             "Momentum Rank": mom_rank.get(t,"-"),
             "Convexity Rank": conv_rank.get(t,"-"),
             "Rank Gain": mom_rank.get(t,0) - conv_rank.get(t,0)}
            for t in df_ranked["Ticker"]
            if mom_rank.get(t,0) - conv_rank.get(t,0) >= 3]
    if divs:
        st.dataframe(pd.DataFrame(divs).sort_values("Rank Gain", ascending=False),
                     width="stretch", hide_index=True)
    else:
        st.info("No major divergences right now.")

# ── TAB: Themes ──────────────────────────────────────────────────────────────

with tab_themes:
    section_header("Theme / Sub-Sector Analysis", "#9b59b6")
    st.caption(
        "Custom sub-sector groupings benchmarked against representative ETFs. "
        "See where capital is flowing and how your picks compare to the sector."
    )

    themes_data = st.session_state.themes
    _etf_tickers_map = get_theme_etfs(themes_data)
    _all_theme_names = list(themes_data.get("themes", {}).keys())

    # Build ticker-to-theme mapping using auto-assignment
    # This replaces the old manual _user_tickers_map
    _user_tickers_map = {th: [] for th in _all_theme_names}
    _user_tickers_map["Other"] = []
    for _t in st.session_state.tickers:
        _t_themes = get_ticker_themes(themes_data, _t)
        for _th in _t_themes:
            if _th in _user_tickers_map:
                _user_tickers_map[_th].append(_t)
            elif _th == "Other":
                _user_tickers_map["Other"].append(_t)

    # Theme management
    with st.expander("Manage Themes", expanded=False):
        mgmt_col1, mgmt_col2 = st.columns(2)
        with mgmt_col1:
            st.markdown("**Move ticker to theme**")
            move_ticker = st.selectbox("Ticker", sorted(st.session_state.tickers), key="theme_move_ticker")
            _auto_themes = get_ticker_themes(themes_data, move_ticker)
            current_theme = ", ".join(_auto_themes) if _auto_themes else "Other"
            st.caption(f"Currently in: {current_theme}")
            target_theme = st.selectbox("Move to", _all_theme_names, key="theme_target")
            if st.button("Move", use_container_width=True, key="theme_move_btn"):
                td = themes_data.get("themes", {})
                for th in td:
                    if move_ticker in td[th].get("tickers", []):
                        td[th]["tickers"].remove(move_ticker)
                if target_theme in td:
                    if move_ticker not in td[target_theme].get("tickers", []):
                        td[target_theme].setdefault("tickers", []).append(move_ticker)
                themes_data["themes"] = td
                st.session_state.themes = themes_data
                save_themes(themes_data)
                st.rerun()
        with mgmt_col2:
            st.markdown("**Create new theme**")
            new_theme = st.text_input("Theme name", placeholder="e.g. Drones, Quantum", key="new_theme_input")
            if st.button("Create Theme", use_container_width=True, key="theme_create_btn"):
                if new_theme and new_theme not in themes_data.get("themes", {}):
                    themes_data.setdefault("themes", {})[new_theme] = {"etfs": [], "tickers": []}
                    st.session_state.themes = themes_data
                    save_themes(themes_data)
                    st.rerun()

    df_theme = df_all

    # ── (a) Theme Momentum Map (scatter plot) ──
    st.markdown("#### Theme Momentum Map")
    st.caption(
        "Each dot is a theme, positioned by its ETF benchmark returns. "
        "Green = you hold tickers in this theme. Grey = no holdings."
    )
    if not df_etf.empty and spy_ret:
        scatter_rows = []
        for theme_name in _all_theme_names:
            etf_list = _etf_tickers_map.get(theme_name, [])
            etf_sub = df_etf[df_etf["Ticker"].isin(etf_list)]
            if etf_sub.empty:
                continue
            avg_3m = etf_sub["Ret3m"].dropna().mean() if etf_sub["Ret3m"].notna().any() else None
            avg_ytd = etf_sub["RetYTD"].dropna().mean() if etf_sub["RetYTD"].notna().any() else None
            avg_1y = etf_sub["Ret1y"].dropna().mean() if etf_sub["Ret1y"].notna().any() else None
            user_tks = _user_tickers_map.get(theme_name, [])
            has_holdings = any(t in df_theme["Ticker"].values for t in user_tks)
            scatter_rows.append({
                "Theme": theme_name,
                "3m Return": round(avg_3m, 1) if avg_3m is not None else None,
                "YTD Return": round(avg_ytd, 1) if avg_ytd is not None else None,
                "1y Return": round(avg_1y, 1) if avg_1y is not None else None,
                "ETFs": ", ".join(etf_list),
                "Holdings": has_holdings,
            })
        if scatter_rows:
            df_scatter = pd.DataFrame(scatter_rows).dropna(subset=["3m Return"])
            if not df_scatter.empty:
                # Use 3m as X, 1y (or YTD if 1y missing) as Y
                df_scatter["Y Axis"] = df_scatter["1y Return"].fillna(df_scatter["YTD Return"])
                df_scatter = df_scatter.dropna(subset=["Y Axis"])
                if not df_scatter.empty:
                    fig_scatter = go.Figure()
                    held = df_scatter[df_scatter["Holdings"]]
                    not_held = df_scatter[~df_scatter["Holdings"]]
                    if not not_held.empty:
                        fig_scatter.add_trace(go.Scatter(
                            x=not_held["3m Return"], y=not_held["Y Axis"],
                            mode="markers+text", text=not_held["Theme"],
                            textposition="top center", textfont=dict(size=10, color="#888"),
                            marker=dict(size=12, color="#555", opacity=0.6, line=dict(width=1, color="#888")),
                            name="No Holdings",
                            hovertemplate="<b>%{text}</b><br>3m: %{x:+.1f}%<br>1y: %{y:+.1f}%<br>ETFs: " +
                                          not_held["ETFs"].tolist().__repr__() + "<extra></extra>" if False else
                                          "<b>%{text}</b><br>3m: %{x:+.1f}%<br>1y: %{y:+.1f}%<extra></extra>",
                        ))
                    if not held.empty:
                        fig_scatter.add_trace(go.Scatter(
                            x=held["3m Return"], y=held["Y Axis"],
                            mode="markers+text", text=held["Theme"],
                            textposition="top center", textfont=dict(size=11, color="#2ecc71"),
                            marker=dict(size=14, color="#2ecc71", opacity=0.9, line=dict(width=1, color="#27ae60")),
                            name="Your Holdings",
                            hovertemplate="<b>%{text}</b><br>3m: %{x:+.1f}%<br>1y: %{y:+.1f}%<extra></extra>",
                        ))
                    fig_scatter.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
                    fig_scatter.add_vline(x=0, line_dash="dot", line_color="#444", line_width=1)
                    fig_scatter.update_layout(
                        height=500,
                        xaxis_title="3-Month Return (%)",
                        yaxis_title="1-Year Return (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(l=50, r=30, t=50, b=50), **DARK,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("ETF benchmark data not available for momentum map.")

    # ── (b) Theme Relative Strength (ETF-based) ──
    grad_divider()
    st.markdown("#### Theme Relative Strength")
    st.caption(
        "ETF benchmark performance vs SPY. Sorted by 1-month relative strength to show what's leading now. "
        "Trend compares 1m vs 3m to detect acceleration (capital flowing in) or deceleration (capital flowing out)."
    )
    if spy_ret and not df_etf.empty:
        trend_rows = []
        for theme_name in _all_theme_names:
            etf_list = _etf_tickers_map.get(theme_name, [])
            etf_sub = df_etf[df_etf["Ticker"].isin(etf_list)]
            if etf_sub.empty:
                continue
            etf_1m = etf_sub["Ret1m"].dropna().mean() if etf_sub["Ret1m"].notna().any() else None
            etf_3m = etf_sub["Ret3m"].dropna().mean() if etf_sub["Ret3m"].notna().any() else None
            if etf_1m is None or etf_3m is None:
                continue
            rs_1m = round(etf_1m - spy_ret.get("1m", 0), 1)
            rs_3m = round(etf_3m - spy_ret.get("3m", 0), 1)
            if rs_1m > rs_3m + 1:
                momentum_status = "Accelerating"
            elif rs_1m < rs_3m - 1:
                momentum_status = "Decelerating"
            else:
                momentum_status = "Stable"
            # RS Rating based on 1m relative strength
            if rs_1m >= 10:
                rs_rating = "Strong"
            elif rs_1m >= 3:
                rs_rating = "Leading"
            elif rs_1m >= -3:
                rs_rating = "Inline"
            elif rs_1m >= -10:
                rs_rating = "Lagging"
            else:
                rs_rating = "Weak"
            # Your picks vs ETF (1m)
            user_tks = _user_tickers_map.get(theme_name, [])
            active_user = [t for t in user_tks if t in df_theme["Ticker"].values]
            picks_vs_etf = None
            if active_user:
                user_sub = df_theme[df_theme["Ticker"].isin(active_user)]
                user_1m = user_sub["Ret1m"].dropna().mean() if user_sub["Ret1m"].notna().any() else None
                if user_1m is not None:
                    picks_vs_etf = round(user_1m - etf_1m, 1)
            trend_rows.append({
                "Theme": theme_name,
                "ETF(s)": ", ".join(etf_list),
                "RS Rating": rs_rating,
                "1m vs SPY": rs_1m,
                "3m vs SPY": rs_3m,
                "Trend": momentum_status,
                "Picks vs ETF": picks_vs_etf,
            })
        if trend_rows:
            df_trend = pd.DataFrame(trend_rows).sort_values("1m vs SPY", ascending=False)

            # ── Call-outs: strongest and weakest themes ──
            leaders = df_trend[df_trend["1m vs SPY"] >= 3]
            laggards = df_trend[df_trend["1m vs SPY"] <= -3]
            accelerating = df_trend[df_trend["Trend"] == "Accelerating"]

            callout_parts = []
            if not leaders.empty:
                top_names = leaders["Theme"].tolist()
                callout_parts.append(
                    f"**Leading SPY:** {', '.join(top_names)}"
                )
            if not accelerating.empty:
                accel_names = [t for t in accelerating["Theme"].tolist() if t not in leaders.get("Theme", pd.Series()).tolist()]
                if accel_names:
                    callout_parts.append(
                        f"**Accelerating:** {', '.join(accel_names)}"
                    )
            if not laggards.empty:
                lag_names = laggards["Theme"].tolist()
                callout_parts.append(
                    f"**Lagging SPY:** {', '.join(lag_names)}"
                )

            if callout_parts:
                st.markdown(" | ".join(callout_parts))

            def _color_rs_rating(val):
                colors = {
                    "Strong": "background-color: #0d4a2a; color: #2ecc71",
                    "Leading": "background-color: #1a3a2a; color: #2ecc71",
                    "Inline": "background-color: #2a2a2a; color: #aaa",
                    "Lagging": "background-color: #3a2a1a; color: #f39c12",
                    "Weak": "background-color: #5b2c2c; color: #ff6b6b",
                }
                return colors.get(val, "")

            def _color_trend(val):
                if val == "Accelerating":
                    return "background-color: #1a3a2a; color: #2ecc71"
                if val == "Decelerating":
                    return "background-color: #5b2c2c; color: #ff6b6b"
                return ""

            def _color_vs_spy(val):
                if pd.isna(val): return ""
                if val >= 5: return "background-color: #1a3a2a; color: #2ecc71"
                if val >= 3: return "background-color: #1a3a2a; color: #8ade8a"
                if val <= -5: return "background-color: #5b2c2c; color: #ff6b6b"
                if val <= -3: return "background-color: #3a2a1a; color: #f39c12"
                return ""

            def _color_picks(val):
                if pd.isna(val): return ""
                if val >= 3: return "background-color: #1a3a2a; color: #2ecc71"
                if val <= -3: return "background-color: #5b2c2c; color: #ff6b6b"
                return ""

            trend_styled = (df_trend.style
                .map(_color_rs_rating, subset=["RS Rating"])
                .map(_color_trend, subset=["Trend"])
                .map(_color_vs_spy, subset=["1m vs SPY", "3m vs SPY"])
                .map(_color_picks, subset=["Picks vs ETF"])
                .format({
                    "1m vs SPY": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                    "3m vs SPY": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                    "Picks vs ETF": lambda x: f"{x:+.1f}%" if pd.notna(x) else "--",
                }, na_rep="N/A")
            )
            st.dataframe(trend_styled, width="stretch", hide_index=True)
            st.caption(
                "**RS Rating:** Strong (>+10%) / Leading (+3 to +10%) / Inline (-3 to +3%) / Lagging (-10 to -3%) / Weak (<-10%)  \n"
                "**Trend:** Accelerating = 1m RS improving vs 3m RS  |  Decelerating = 1m RS fading vs 3m RS  \n"
                "**Picks vs ETF:** your tickers' 1m return minus the ETF benchmark"
            )
        else:
            st.info("Not enough ETF return data to compute momentum trends.")
    else:
        st.info("SPY or ETF data not available for momentum trends.")

    # ── (c) Theme Pillar Breakdown (holdings only) ──
    grad_divider()
    st.markdown("#### Theme Pillar Breakdown")
    st.caption("Average four-pillar scores for themes where you hold tickers. Shows where each theme is strong or weak.")

    theme_score_rows = []
    for theme_name in _all_theme_names:
        user_tks = _user_tickers_map.get(theme_name, [])
        active = [t for t in user_tks if t in df_theme["Ticker"].values]
        if not active:
            continue
        sub = df_theme[df_theme["Ticker"].isin(active)]
        theme_score_rows.append({
            "Theme": theme_name,
            "Tickers": len(active),
            "Technical": round(sub["PillarTech"].mean(), 1),
            "Fundamental": round(sub["PillarFund"].mean(), 1),
            "Thematic": round(sub["PillarTheme"].mean(), 1),
            "Narrative": round(sub["PillarNarr"].mean(), 1),
            "Convexity": round(sub["ConvexityScore"].mean(), 1),
            "Members": ", ".join(active),
        })

    if theme_score_rows:
        df_scores = pd.DataFrame(theme_score_rows).sort_values("Convexity", ascending=False)

        _pillar_colors = {
            "Technical": "#3498db", "Fundamental": "#2ecc71",
            "Thematic": "#9b59b6", "Narrative": "#f39c12",
        }
        fig_th = go.Figure()
        for pillar, color in _pillar_colors.items():
            fig_th.add_trace(go.Bar(
                name=pillar, x=df_scores["Theme"], y=df_scores[pillar],
                marker_color=color,
                hovertemplate="<b>%{x}</b><br>" + pillar + ": %{y:.0f}/100<extra></extra>",
            ))
        fig_th.update_layout(
            barmode="group", height=420,
            yaxis_title="Pillar Score (0-100)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=30, t=50, b=40), **DARK,
        )
        fig_th.add_hline(y=50, line_dash="dot", line_color="#555", annotation_text="Neutral",
                         annotation_position="bottom right", annotation_font_color="#666")
        st.plotly_chart(fig_th, use_container_width=True)

    # ── (d) Theme Drilldown — flat table grouped by theme ──
    grad_divider()
    st.markdown("#### Theme Drilldown")
    st.caption("All holdings grouped by theme with ETF benchmark context. Sorted by convexity within each theme.")

    _drill_rows = []
    for theme_name in _all_theme_names:
        user_tks = _user_tickers_map.get(theme_name, [])
        members = [t for t in user_tks if t in df_theme["Ticker"].values]
        if not members:
            continue
        etf_list = _etf_tickers_map.get(theme_name, [])
        etf_sub = df_etf[df_etf["Ticker"].isin(etf_list)] if not df_etf.empty else pd.DataFrame()
        etf_1m = round(etf_sub["Ret1m"].dropna().mean(), 1) if not etf_sub.empty and etf_sub["Ret1m"].notna().any() else None
        etf_3m = round(etf_sub["Ret3m"].dropna().mean(), 1) if not etf_sub.empty and etf_sub["Ret3m"].notna().any() else None

        sub = df_theme[df_theme["Ticker"].isin(members)].sort_values("ConvexityScore", ascending=False)
        for _, r in sub.iterrows():
            _r1m = r.get("Ret1m")
            _r3m = r.get("Ret3m")
            _drill_rows.append({
                "Theme": theme_name,
                "Ticker": r["Ticker"],
                "Price": r.get("Price"),
                "RSI": r.get("RSI"),
                "52wk Pos": r.get("Pos52"),
                "Convexity": r.get("ConvexityScore"),
                "Momentum": r.get("MomentumScore"),
                "1m Ret": _r1m,
                "3m Ret": _r3m,
                "vs ETF 1m": round(_r1m - etf_1m, 1) if pd.notna(_r1m) and etf_1m is not None else None,
                "ETF 1m": etf_1m,
                "ETF 3m": etf_3m,
            })

    if _drill_rows:
        df_drill = pd.DataFrame(_drill_rows)

        def _drill_color_vs_etf(val):
            if pd.isna(val): return ""
            if val >= 5: return "background-color: #1a3a2a; color: #2ecc71"
            if val >= 0: return "background-color: #1a3a2a; color: #8ade8a"
            if val <= -5: return "background-color: #5b2c2c; color: #ff6b6b"
            return "background-color: #3a2a1a; color: #f39c12"

        _drill_grad = [c for c in ["Convexity", "Momentum"] if df_drill[c].notna().any()]
        drill_styled = df_drill.style
        if _drill_grad:
            drill_styled = drill_styled.background_gradient(subset=_drill_grad, cmap="RdYlGn", vmin=20, vmax=80)
        drill_styled = (drill_styled
            .map(_drill_color_vs_etf, subset=["vs ETF 1m"])
            .format({
                "Price": "${:.2f}",
                "RSI": "{:.0f}",
                "52wk Pos": "{:.0f}%",
                "Convexity": "{:.0f}",
                "Momentum": "{:.0f}",
                "1m Ret": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                "3m Ret": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                "vs ETF 1m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "--",
                "ETF 1m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "--",
                "ETF 3m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "--",
            }, na_rep="N/A")
        )
        st.dataframe(drill_styled, width="stretch", hide_index=True, height=min(len(df_drill) * 38 + 40, 600))

    # Untagged tickers
    all_themed = set()
    for tks in _user_tickers_map.values():
        all_themed.update(tks)
    untagged = [t for t in st.session_state.tickers if t not in all_themed and t in df_theme["Ticker"].values]
    if untagged:
        st.caption(f"Untagged tickers: {', '.join(untagged)} -- use 'Manage Themes' above to assign them.")
    if not theme_score_rows and not scatter_rows:
        st.info("No themes configured or no ETF data available. Use 'Manage Themes' above to create sub-sector groups.")

# ── TAB: Charts ─────────────────────────────────────────────────────────────

with tab_charts:
    section_header("Price Charts", "#3498db")

    from plotly.subplots import make_subplots

    # Timeframe toggle
    chart_tf = st.radio("Timeframe", ["Daily", "Weekly"], horizontal=True, key="chart_timeframe")
    _is_weekly = chart_tf == "Weekly"

    if _is_weekly:
        _ch_period, _ch_interval = "2y", "1wk"
        _ch_ma_short, _ch_ma_long = 10, 40
        _ch_ma_short_label, _ch_ma_long_label = "10-WMA", "40-WMA"
        st.caption("2-year weekly price action with 10-week and 40-week moving averages.")
    else:
        _ch_period, _ch_interval = "6mo", "1d"
        _ch_ma_short, _ch_ma_long = 20, 50
        _ch_ma_short_label, _ch_ma_long_label = "20-MA", "50-MA"
        st.caption("6-month daily price action with 20-day and 50-day moving averages.")

    # Build selectable options: SPY + QQQ first, then portfolio tickers
    _chart_options = ["SPY", "QQQ"] + [t for t in df_price["Ticker"].tolist() if t not in ("SPY", "QQQ")]
    _chart_default = "SPY" if "SPY" not in df_price["Ticker"].tolist() else df_price["Ticker"].tolist()[0]
    chart_ticker = st.selectbox(
        "Select ticker to chart",
        _chart_options,
        index=0,
        key="chart_ticker"
    )

    if chart_ticker:
        try:
            ch_data = yf.download(chart_ticker, period=_ch_period, interval=_ch_interval,
                                  progress=False, auto_adjust=True)
            if ch_data.empty:
                st.caption(f"{chart_ticker}: no data")
            else:
                ch_close = ch_data["Close"].squeeze()
                ch_vol = ch_data["Volume"].squeeze()
                ch_ma_s = ch_close.rolling(_ch_ma_short).mean()
                ch_ma_l = ch_close.rolling(_ch_ma_long).mean()

                # Get current metrics for subtitle
                cr = df_all[df_all["Ticker"] == chart_ticker]
                subtitle = ""
                if not cr.empty:
                    cr = cr.iloc[0]
                    subtitle = f"RSI {cr['RSI']:.0f}  |  52wk {cr['Pos52']:.0f}%  |  ATR {cr['ATR_pct']:.1f}%"

                fig_ch = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.78, 0.22], vertical_spacing=0.03)
                fig_ch.add_trace(go.Scatter(
                    x=ch_close.index, y=ch_close.values, mode="lines",
                    name="Price", line=dict(color="#3498db", width=2),
                ), row=1, col=1)
                fig_ch.add_trace(go.Scatter(
                    x=ch_ma_s.index, y=ch_ma_s.values, mode="lines",
                    name=_ch_ma_short_label, line=dict(color="#2ecc71", width=1, dash="dot"),
                ), row=1, col=1)
                fig_ch.add_trace(go.Scatter(
                    x=ch_ma_l.index, y=ch_ma_l.values, mode="lines",
                    name=_ch_ma_long_label, line=dict(color="#f39c12", width=1, dash="dash"),
                ), row=1, col=1)
                fig_ch.add_trace(go.Bar(
                    x=ch_vol.index, y=ch_vol.values, name="Volume",
                    marker_color="#555", opacity=0.5, showlegend=False,
                ), row=2, col=1)
                fig_ch.update_layout(
                    height=450,
                    title=dict(text=f"{chart_ticker}  <span style='font-size:12px;color:#888'>{subtitle}</span>",
                               font=dict(size=16)),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
                    margin=dict(l=50, r=15, t=50, b=30), **DARK,
                )
                fig_ch.update_yaxes(title_text="$", row=1, col=1, gridcolor="#21262d")
                fig_ch.update_yaxes(row=2, col=1, gridcolor="#21262d")
                st.plotly_chart(fig_ch, use_container_width=True)
        except (AttributeError, ValueError, KeyError, TypeError):
            st.caption(f"{chart_ticker}: chart unavailable")

# ── TAB: Strength ────────────────────────────────────────────────────────────

with tab_str:
    section_header("Relative Strength — Who's Holding Up?", "#2ecc71")
    st.caption(
        "Stocks that refuse to go lower on market down-days are being accumulated. "
        "**Leaders** hold up best when SPY sells off — they often lead the next rally. "
        "**Down-Day Win Rate** = % of SPY's worst days where the stock outperformed."
    )

    rs_view = df_all[df_all["RS_Score"].notna()].copy() if "RS_Score" in df_all.columns else pd.DataFrame()

    if not rs_view.empty:
        rs_view = rs_view.sort_values("RS_Score", ascending=False)

        # ── Leader / Laggard callout ──
        leaders = rs_view[rs_view["RS_Label"] == "Leader"]["Ticker"].tolist()
        laggards = rs_view[rs_view["RS_Label"] == "Lagging"]["Ticker"].tolist()
        early_bottoms = rs_view[rs_view["EarlyBottom"] == True]["Ticker"].tolist()
        col_l, col_lag, col_eb = st.columns(3)
        with col_l:
            st.success(f"**Leaders ({len(leaders)})**: {', '.join(leaders) if leaders else 'None'}")
        with col_lag:
            st.error(f"**Lagging ({len(laggards)})**: {', '.join(laggards) if laggards else 'None'}")
        with col_eb:
            st.info(f"**Early Bottom ({len(early_bottoms)})**: {', '.join(early_bottoms) if early_bottoms else 'None'}")

        # ── RS Score chart (horizontal bar, color-coded by label) ──
        label_colors = {"Leader": "#2ecc71", "Holding": "#f1c40f", "Lagging": "#e74c3c"}
        fig_rs = go.Figure()
        for label, color in label_colors.items():
            subset = rs_view[rs_view["RS_Label"] == label]
            if subset.empty:
                continue
            fig_rs.add_trace(go.Bar(
                name=label,
                y=subset["Ticker"], x=subset["RS_Score"], orientation="h",
                marker_color=color,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "RS Score: %{x:.0f}<br>"
                    + "Down-Day Win Rate: %{customdata[0]:.0f}%<br>"
                    + "Down-Day Excess: %{customdata[1]:+.2f}%<br>"
                    + "Recent DD Excess: %{customdata[2]:+.2f}%<br>"
                    + "Early Bottom: %{customdata[3]}<extra></extra>"
                ),
                customdata=subset[["DownDayWinRate", "DownDayRS", "RecentDDExcess", "EarlyBottom"]].values,
            ))
        fig_rs.update_layout(
            barmode="stack", height=max(400, len(rs_view) * 32),
            xaxis=dict(title="RS Score (0-100)", range=[0, 105]),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=60, r=20, t=50, b=40), **DARK,
        )
        st.plotly_chart(fig_rs, use_container_width=True)

        # ── Down-Day Win Rate scatter ──
        grad_divider()
        st.markdown("##### Down-Day Behaviour Detail")
        fig_dd = go.Figure()
        for _, row in rs_view.iterrows():
            color = label_colors.get(row.get("RS_Label", "Holding"), "#f1c40f")
            fig_dd.add_trace(go.Scatter(
                x=[row["DownDayWinRate"]], y=[row["RecentDDExcess"]],
                mode="markers+text", text=[row["Ticker"]], textposition="top center",
                marker=dict(size=14, color=color, line=dict(width=1, color="#0d1117")),
                hovertemplate=(
                    f"<b>{row['Ticker']}</b><br>"
                    f"Win Rate: {row['DownDayWinRate']:.0f}%<br>"
                    f"Recent Excess: {row['RecentDDExcess']:+.2f}%/day<br>"
                    f"RS Score: {row['RS_Score']:.0f}<extra></extra>"
                ),
                showlegend=False,
            ))
        fig_dd.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
        fig_dd.add_vline(x=50, line_dash="dot", line_color="#555", line_width=1)
        for (x, y, lbl) in [(75, 0.4, "STRONG LEADER"), (25, 0.4, "RECOVERING"),
                             (75, -0.4, "FADING"), (25, -0.4, "WEAK")]:
            fig_dd.add_annotation(x=x, y=y, text=lbl, showarrow=False,
                                  font=dict(color="#555", size=11, family="monospace"))
        fig_dd.update_layout(
            height=450,
            xaxis=dict(title="Down-Day Win Rate (%)", gridcolor="#21262d"),
            yaxis=dict(title="Recent Down-Day Excess Return (%/day)", gridcolor="#21262d"),
            margin=dict(l=60, r=20, t=20, b=60), **DARK,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── RS Table (colour-coded) ──
        grad_divider()
        rs_tbl = rs_view[["Ticker", "RS_Score", "RS_Rank", "RS_Label", "DownDayWinRate",
                          "DownDayRS", "RecentDDExcess", "EarlyBottom", "DaysSinceLow",
                          "RS_1m", "RS_3m"]].copy()
        rs_tbl.columns = ["Ticker", "RS Score", "Rank", "Status", "DD Win%",
                          "DD Excess", "Recent DD", "Early Bottom", "Days Since Low",
                          "1m vs SPY", "3m vs SPY"]

        def _color_rs_label(val):
            if val == "Leader":
                return "background-color: #1a3a2a; color: #2ecc71"
            elif val == "Lagging":
                return "background-color: #5b2c2c; color: #ff6b6b"
            return "background-color: #3d3520; color: #f1c40f"

        styled_rs = (rs_tbl.style
            .map(_color_rs_label, subset=["Status"])
            .background_gradient(cmap="RdYlGn", subset=["RS Score", "DD Win%", "1m vs SPY", "3m vs SPY"])
            .format({
                "RS Score": "{:.0f}", "DD Win%": "{:.0f}%",
                "DD Excess": "{:+.2f}%", "Recent DD": "{:+.2f}%",
                "Days Since Low": "{:.0f}",
                "1m vs SPY": "{:+.1f}%", "3m vs SPY": "{:+.1f}%",
            })
        )
        st.dataframe(styled_rs, use_container_width=True, hide_index=True)

        if spy_ret:
            st.caption(f"SPY returns — 1m: {spy_ret.get('1m', 'N/A'):+.1f}%  |  3m: {spy_ret.get('3m', 'N/A'):+.1f}%  |  6m: {spy_ret.get('6m', 'N/A'):+.1f}%")
    else:
        st.info("Relative strength data unavailable — need sufficient price history.")

# ── TAB: Signals ─────────────────────────────────────────────────────────────

with tab_sig:
    # ── Signal Summary (quick-glance metrics) ────────────────────────────────
    section_header("Signal Summary", "#58a6ff")

    _sig_insider_buying = df_all[df_all["InsiderSignal"] == "Buying"]["Ticker"].tolist()
    _sig_analyst_upside = df_all[df_all["AnalystUpside"].notna() & (df_all["AnalystUpside"] > 20)]["Ticker"].tolist()
    _sig_short_squeeze = df_all[df_all["ShortPct"].notna() & (df_all["ShortPct"] > 10)]["Ticker"].tolist()
    _sig_earn_14d = []
    _earn_tmp = df_all[df_all["NextEarnings"].notna() & (df_all["NextEarnings"] != "")].copy()
    if not _earn_tmp.empty:
        _earn_tmp["_ed"] = pd.to_datetime(_earn_tmp["NextEarnings"], errors="coerce")
        _today_sig = pd.Timestamp.now().normalize()
        _earn_14 = _earn_tmp[(_earn_tmp["_ed"] >= _today_sig) & (_earn_tmp["_ed"] <= _today_sig + pd.Timedelta(days=14))]
        _sig_earn_14d = _earn_14["Ticker"].tolist()

    _ss1, _ss2, _ss3, _ss4 = st.columns(4)
    _ss1.metric("Insider Buying", len(_sig_insider_buying),
                ", ".join(_sig_insider_buying) if _sig_insider_buying else "None")
    _ss2.metric("Analyst Upside >20%", len(_sig_analyst_upside),
                ", ".join(_sig_analyst_upside[:5]) if _sig_analyst_upside else "None")
    _ss3.metric("Short >10% (Squeeze)", len(_sig_short_squeeze),
                ", ".join(_sig_short_squeeze) if _sig_short_squeeze else "None")
    _ss4.metric("Earnings <14d", len(_sig_earn_14d),
                ", ".join(_sig_earn_14d) if _sig_earn_14d else "None")

    grad_divider()

    # ── Setup Stage Summary ──────────────────────────────────────────────────
    section_header("Setup Stage Summary", "#3498db")
    st.caption("Where each ticker sits in the swing-trading lifecycle.")
    _stage_descriptions = {
        "Basing": "watch for entry",
        "Emerging": "early momentum",
        "Trending": "hold/ride",
        "Extended": "consider trimming",
        "Breaking Down": "review thesis",
        "Neutral": "no clear stage",
    }
    for stage_name in ["Basing", "Emerging", "Trending", "Extended", "Breaking Down", "Neutral"]:
        stage_tickers = df_all[df_all["SetupStage"] == stage_name]["Ticker"].tolist()
        if stage_tickers:
            desc = _stage_descriptions.get(stage_name, "")
            color = _stage_colors.get(stage_name, "#888")
            st.markdown(
                f"<span style='color:{color};font-weight:bold'>{stage_name}</span> "
                f"({desc}): **{', '.join(stage_tickers)}**",
                unsafe_allow_html=True,
            )

    grad_divider()

    # ── Upcoming Earnings ────────────────────────────────────────────────────
    section_header("Upcoming Earnings", "#f39c12")
    earn_data = df_all[df_all["NextEarnings"].notna() & (df_all["NextEarnings"] != "")][
        ["Ticker", "NextEarnings", "ConvexityScore", "RSI", "Price"]
    ].copy()
    if not earn_data.empty:
        earn_data["EarningsDate"] = pd.to_datetime(earn_data["NextEarnings"], errors="coerce")
        earn_data = earn_data.dropna(subset=["EarningsDate"]).sort_values("EarningsDate")
        today_dt = pd.Timestamp.now().normalize()
        earn_upcoming = earn_data[earn_data["EarningsDate"] >= today_dt - pd.Timedelta(days=1)]
        if not earn_upcoming.empty:
            fig_earn = go.Figure()
            for _, erow in earn_upcoming.iterrows():
                days_away = (erow["EarningsDate"] - today_dt).days
                color = "#e74c3c" if days_away <= 7 else "#f39c12" if days_away <= 21 else "#3498db"
                fig_earn.add_trace(go.Scatter(
                    x=[erow["EarningsDate"]], y=[erow["Ticker"]],
                    mode="markers+text", text=[f"  {days_away}d"],
                    textposition="middle right", textfont=dict(size=11, color=color),
                    marker=dict(size=14, color=color, symbol="diamond"),
                    hovertemplate=(
                        f"<b>{erow['Ticker']}</b><br>"
                        f"Earnings: {erow['NextEarnings']}<br>"
                        f"Days away: {days_away}<br>"
                        f"Convexity: {erow['ConvexityScore']:.0f}<br>"
                        f"RSI: {erow['RSI']:.0f}<extra></extra>"
                    ),
                    showlegend=False,
                ))
            fig_earn.add_vline(x=today_dt.isoformat(), line_dash="dash", line_color="#2ecc71")
            fig_earn.add_annotation(x=today_dt.isoformat(), y=1, yref="paper",
                                    text="Today", showarrow=False,
                                    font=dict(color="#2ecc71", size=11))
            fig_earn.update_layout(
                height=max(200, len(earn_upcoming) * 35 + 60),
                xaxis=dict(title="Date", gridcolor="#21262d"),
                margin=dict(l=60, r=40, t=20, b=40), **DARK,
            )
            st.plotly_chart(fig_earn, use_container_width=True)
            st.caption("Red = this week  |  Orange = next 3 weeks  |  Blue = later")
        else:
            st.info("No upcoming earnings dates found.")
    else:
        st.info("No earnings date data available.")

    grad_divider()

    # ── Analyst Targets, News & Sentiment ────────────────────────────────────
    section_header("Analyst Targets, News & Sentiment", "#e74c3c")

    df_sig = df_all.copy()

    # ── Insider Activity (new) ──
    st.markdown("#### Insider Activity")
    st.caption("People sell for many reasons, but only buy for one. Recent insider purchases (last 90 days).")
    _insider_buy_tickers = df_sig[df_sig["InsiderSignal"] == "Buying"]
    if not _insider_buy_tickers.empty:
        insider_rows = []
        for _, irow in _insider_buy_tickers.iterrows():
            buys_list = irow.get("InsiderBuys", [])
            if isinstance(buys_list, list) and buys_list:
                for b in buys_list:
                    val = b.get("value", 0)
                    insider_rows.append({
                        "Ticker": irow["Ticker"],
                        "Price": irow["Price"],
                        "Insider": b.get("insider", "Unknown"),
                        "Role": b.get("position", ""),
                        "Date": b.get("date", ""),
                        "Shares": f"{b.get('shares', 0):,}",
                        "Value": f"${val:,.0f}" if val else "N/A",
                    })
            else:
                insider_rows.append({
                    "Ticker": irow["Ticker"],
                    "Price": irow["Price"],
                    "Insider": "—",
                    "Role": "—",
                    "Date": "—",
                    "Shares": "—",
                    "Value": "—",
                })
        if insider_rows:
            df_insider_tbl = pd.DataFrame(insider_rows)
            def _color_insider_val(val):
                if val == "N/A" or val == "—":
                    return ""
                try:
                    v = float(val.replace("$","").replace(",",""))
                    if v >= 500_000: return "background-color: #1a3a2a; color: #2ecc71; font-weight: bold"
                    if v >= 100_000: return "background-color: #1a3a2a; color: #2ecc71"
                    return ""
                except (TypeError, ValueError, AttributeError):
                    return ""
            insider_styled = df_insider_tbl.style.map(_color_insider_val, subset=["Value"])
            st.dataframe(insider_styled, width="stretch", hide_index=True)
    else:
        st.info("No insider purchases detected in the last 90 days across your watchlist.")
    _insider_sell_tickers = df_sig[df_sig["InsiderSignal"] == "Selling"]["Ticker"].tolist()
    if _insider_sell_tickers:
        st.caption(f"Insider selling detected: {', '.join(_insider_sell_tickers)}")

    grad_divider()

    # Analyst price targets chart
    st.markdown("#### Analyst Price Targets")
    tgt_df = df_sig[df_sig["TargetMean"].notna() & (df_sig["TargetMean"] > 0)]
    if not tgt_df.empty:
        fig_tgt = go.Figure()
        for _, row in tgt_df.iterrows():
            t = row["Ticker"]
            tl = row.get("TargetLow") or row["Price"] * 0.9
            th = row.get("TargetHigh") or row["Price"] * 1.2
            fig_tgt.add_trace(go.Scatter(
                x=[t, t], y=[tl, th],
                mode="lines", line=dict(color="#444", width=8),
                showlegend=False, hoverinfo="skip",
            ))
            fig_tgt.add_trace(go.Scatter(
                x=[t], y=[row["TargetMean"]],
                mode="markers", marker=dict(symbol="diamond", color="#f39c12", size=14),
                showlegend=False,
                hovertemplate=f"<b>{t}</b><br>Mean: ${row['TargetMean']:.2f}<br>Upside: {row['AnalystUpside']:+.0f}%<extra></extra>",
            ))
            fig_tgt.add_trace(go.Scatter(
                x=[t], y=[row["Price"]],
                mode="markers", marker=dict(symbol="circle", color="white", size=10),
                showlegend=False,
                hovertemplate=f"<b>{t}</b> current: ${row['Price']:.2f}<extra></extra>",
            ))
        fig_tgt.update_layout(height=380, yaxis_title="Price ($)",
                               margin=dict(l=40,r=20,t=20,b=40), showlegend=False, **DARK)
        st.plotly_chart(fig_tgt, use_container_width=True)
    else:
        st.info("No analyst price targets available.")

    grad_divider()

    # Full signal table
    st.markdown("#### Full Signal Table")
    def fmt_rec(r):
        icons = {"strong_buy":"Strong Buy","buy":"Buy","hold":"Hold",
                 "sell":"Sell","strong_sell":"Strong Sell"}
        return icons.get(str(r).lower().replace(" ","_"), r or "N/A")

    sig_rows = []
    for _, row in df_sig.iterrows():
        sent = st_data.get(row["Ticker"], {})
        sig_rows.append({
            "Ticker": row["Ticker"], "Price": row["Price"],
            "Beta": row.get("Beta"), "ShortPct": row.get("ShortPct"),
            "AnalystUpside": row.get("AnalystUpside"),
            "TargetMean": row.get("TargetMean"),
            "NumAnalysts": row.get("NumAnalysts"),
            "Recommendation": row.get("Recommendation",""),
            "RevenueGrowth": row.get("RevenueGrowth"), "MarketCap": row.get("MarketCap"),
            "RSI": row["RSI"], "ATR_pct": row["ATR_pct"], "RelVol": row.get("RelVol"),
            "InsiderSignal": row.get("InsiderSignal","N/A"),
            "NextEarnings": row.get("NextEarnings"),
            "BullPct": sent.get("bull_pct"), "MsgCount": sent.get("msg_count", 0),
        })

    tbl = pd.DataFrame(sig_rows)
    tbl["Recommendation"] = tbl["Recommendation"].apply(fmt_rec)
    tbl["MarketCap"]      = tbl["MarketCap"].apply(fmt_mcap)
    tbl["NextEarnings"]   = tbl["NextEarnings"].fillna("N/A")
    tbl = tbl.rename(columns={
        "AnalystUpside":"Upside","TargetMean":"Mean Target",
        "NumAnalysts":"# Analysts","Recommendation":"Consensus",
        "RevenueGrowth":"Rev Growth","MarketCap":"Mkt Cap",
        "ATR_pct":"ATR%","RelVol":"Rel Vol","InsiderSignal":"Insider",
        "NextEarnings":"Next Earnings",
        "BullPct":"ST Bulls","MsgCount":"ST Msgs","ShortPct":"Short%",
    })

    def _sig_color_upside(val):
        if pd.isna(val):
            return ""
        if val >= 30:
            return "background-color: #1a3a2a; color: #2ecc71"
        if val <= -10:
            return "background-color: #5b2c2c; color: #ff6b6b"
        return ""

    def _sig_color_short(val):
        if pd.isna(val):
            return ""
        if val >= 15:
            return "background-color: #5b2c2c; color: #ff6b6b"
        if val >= 8:
            return "background-color: #3a2c1a; color: #f39c12"
        return ""

    def _sig_color_rsi(val):
        if pd.isna(val):
            return ""
        if val >= 70:
            return "background-color: #5b2c2c; color: #ff6b6b"
        if val <= 30:
            return "background-color: #1a3a2a; color: #2ecc71"
        return ""

    def _sig_color_insider(val):
        v = str(val).lower()
        if v == "buying":
            return "background-color: #1a3a2a; color: #2ecc71; font-weight: bold"
        if v == "selling":
            return "background-color: #3a2c1a; color: #f39c12"
        return ""

    sig_styled = (tbl.style
        .map(_sig_color_upside, subset=["Upside"])
        .map(_sig_color_short, subset=["Short%"])
        .map(_sig_color_rsi, subset=["RSI"])
        .map(_sig_color_insider, subset=["Insider"])
        .format({
            "Price": "${:.2f}",
            "Beta": lambda x: f"{x:.2f}" if pd.notna(x) and x else "N/A",
            "Short%": lambda x: f"{x:.1f}%" if pd.notna(x) and x else "N/A",
            "Upside": lambda x: f"{x:+.0f}%" if pd.notna(x) and x else "N/A",
            "Mean Target": lambda x: f"${x:.2f}" if pd.notna(x) and x else "N/A",
            "Rev Growth": lambda x: f"{x:+.0f}%" if pd.notna(x) and x else "N/A",
            "RSI": "{:.0f}",
            "ATR%": "{:.1f}%",
            "Rel Vol": lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A",
            "ST Bulls": lambda x: f"{x:.0f}% bull" if pd.notna(x) and x else "N/A",
        }, na_rep="N/A")
    )
    st.dataframe(sig_styled, width="stretch", hide_index=True)
    st.download_button("Export Signals CSV", tbl.to_csv(index=False),
                       "signals_data.csv", "text/csv", key="dl_sig")
    st.caption(
        "Insider = net buys vs sells last 90 days.  "
        "ST Bulls = StockTwits bull% from last 30 tagged messages."
    )
    st.caption("⚠️ Short interest data is typically 2 weeks stale (FINRA bi-monthly reporting).")

    grad_divider()

    # ── AI Summary (merged from former AI Summary tab) ──
    section_header("AI Summary — Claude-powered Investment Brief", "#9b59b6")
    st.caption(
        "Synthesizes technical data, analyst targets, insider activity, "
        "StockTwits sentiment, and recent news into a concise investment brief. "
        "Add your Anthropic API key in the sidebar (~$0.001 per ticker)."
    )

    api_key = _get_anthropic_key()
    if not api_key:
        st.warning(
            "Anthropic API key not found.  \n"
            "Add `ANTHROPIC_API_KEY` to your `.env` file locally or Streamlit Cloud secrets.  \n"
            "Get one at **console.anthropic.com**."
        )
    else:
        col_single, col_all = st.columns([2, 1])

        with col_single:
            st.markdown("#### Analyze single ticker")
            selected_ticker = st.selectbox("Select ticker", df_price["Ticker"].tolist())

        with col_all:
            st.markdown("#### Analyze full portfolio")
            st.caption("~30 sec, costs ~$0.02 total")
            run_all = st.button("Run All", use_container_width=True)

        def build_summary(t):
            r  = df_price[df_price["Ticker"] == t].iloc[0]
            fd = df_fund[df_fund["Ticker"] == t].iloc[0].to_dict() if not df_fund[df_fund["Ticker"] == t].empty else {}
            ex = df_extras[df_extras["Ticker"] == t].iloc[0] if not df_extras[df_extras["Ticker"] == t].empty else {}
            sent = st_data.get(t, {})
            return claude_summarize(
                ticker=t,
                price=r["Price"], rsi=r["RSI"], atr_pct=r["ATR_pct"],
                pos52=r["Pos52"], vs50=r["vsMA50"], vs200=r.get("vsMA200"),
                short_pct=fd.get("ShortPct"), beta=fd.get("Beta"),
                upside=fd.get("AnalystUpside"), recommendation=fd.get("Recommendation",""),
                mcap=fmt_mcap(fd.get("MarketCap")), rev_growth=fd.get("RevenueGrowth"),
                insider_signal=ex.get("InsiderSignal") if hasattr(ex, "get") else None,
                earnings_beats=None,
                next_earnings=fd.get("NextEarnings"),
                st_bull_pct=sent.get("bull_pct"), st_msgs=sent.get("msg_count", 0),
                headlines=ex.get("Headlines") if hasattr(ex, "get") else [],
                api_key=api_key,
            )

        if st.button(f"Analyze {selected_ticker}", use_container_width=True):
            with st.spinner(f"Generating brief for ${selected_ticker}..."):
                result = build_summary(selected_ticker)
            st.markdown(result)
            st.caption(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · model: claude-haiku-4-5")

        if run_all:
            grad_divider()
            for t in df_price["Ticker"].tolist():
                with st.spinner(f"Analyzing ${t}..."):
                    out = build_summary(t)
                with st.expander(f"**${t}**", expanded=False):
                    st.markdown(out)
            st.caption(f"All analyses generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Scanner
# ══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    section_header("Opportunity Scanner", "#e74c3c")
    st.caption(
        "Scan external universes for asymmetric small-cap opportunities. "
        "Candidates are scored using the same Convexity, Momentum, and Relative Strength models."
    )

    # ── Filter controls ──
    scan_c1, scan_c2, scan_c3, scan_c4 = st.columns(4)
    with scan_c1:
        scan_min_mcap = st.select_slider(
            "Min Market Cap",
            options=[50, 100, 200, 500, 1000],
            value=200,
            format_func=lambda x: f"${x}M",
        )
    with scan_c2:
        scan_max_mcap = st.select_slider(
            "Max Market Cap",
            options=[1, 2, 5, 10, 20, 50],
            value=10,
            format_func=lambda x: f"${x}B",
        )
    with scan_c3:
        scan_min_rev = st.slider("Min Rev Growth %", 0, 50, 10)
    with scan_c4:
        scan_sources = st.multiselect(
            "Universe Sources",
            ["Yahoo Screener", "Most Actives", "Small Cap Gainers",
             "Aggressive Small Caps", "FINVIZ", "Yahoo Trending"],
            default=["Yahoo Screener", "Small Cap Gainers"],
        )

    # Map source names to functions
    SOURCE_MAP = {
        "Yahoo Screener": ("custom", None),
        "Most Actives": ("predefined", "most_actives"),
        "Small Cap Gainers": ("predefined", "small_cap_gainers"),
        "Aggressive Small Caps": ("predefined", "aggressive_small_caps"),
        "FINVIZ": ("finviz", None),
        "Yahoo Trending": ("trending", None),
    }

    run_scan = st.button("Run Scan", type="primary", use_container_width=True)

    if run_scan:
        with st.spinner("Sourcing universe from selected feeds..."):
            universe = set()
            source_counts = {}
            for src in scan_sources:
                kind, key = SOURCE_MAP[src]
                if kind == "custom":
                    ticks = scan_yahoo_screener(
                        min_mcap=scan_min_mcap * 1_000_000,
                        max_mcap=scan_max_mcap * 1_000_000_000,
                        min_rev_growth=scan_min_rev / 100,
                    )
                elif kind == "predefined":
                    ticks = scan_yahoo_predefined(key)
                elif kind == "finviz":
                    ticks = scan_finviz()
                elif kind == "trending":
                    ticks = scan_yahoo_trending()
                else:
                    ticks = []
                source_counts[src] = len(ticks)
                universe.update(ticks)

            # Remove tickers already in portfolio
            existing = set(st.session_state.get("tickers", []))
            universe -= existing

        # Show sourcing summary
        src_parts = [f"{src}: {ct}" for src, ct in source_counts.items()]
        alert_card(
            f"Universe: {len(universe)} unique candidates "
            f"({', '.join(src_parts)}). "
            f"Excluded {len(existing)} existing portfolio tickers.",
            color="blue",
        )

        if not universe:
            st.warning("No candidates found. Try broadening your filters.")
        else:
            # Cap at 80 to keep scoring fast
            universe_list = sorted(universe)[:80]
            if len(universe) > 80:
                st.caption(f"Capped at 80 candidates (from {len(universe)}) for performance.")

            with st.spinner(f"Scoring {len(universe_list)} candidates — fetching data & computing scores..."):
                spy_c, spy_r = fetch_spy_daily()
                df_scan = score_scanner_candidates(universe_list, spy_c, spy_r)

            if df_scan.empty:
                st.warning("Could not fetch data for any candidates.")
            else:
                # Filter by minimum score
                min_score = st.slider("Min Scan Score", 0, 80, 25, key="scan_min_score")
                df_show = df_scan[df_scan["ScanScore"] >= min_score].copy()

                st.markdown(f"**{len(df_show)}** candidates scoring >= {min_score}")

                if not df_show.empty:
                    # Format display table
                    display_cols = ["Ticker", "Price", "MCap", "ScanScore", "Convexity",
                                    "Momentum", "RS_Score", "RS_Label", "SetupStage",
                                    "RSI", "Pos52", "vsMA50", "RevGrowth",
                                    "ShortPct", "Beta", "AnalystUpside"]
                    available_cols = [c for c in display_cols if c in df_show.columns]
                    df_display = df_show[available_cols].copy()

                    # Format market cap
                    if "MCap" in df_display.columns:
                        df_display["MCap"] = df_display["MCap"].apply(
                            lambda x: fmt_mcap(x) if pd.notna(x) and x else "N/A"
                        )

                    def _color_scan_score(val):
                        if pd.isna(val): return ""
                        if val >= 55: return "background-color: #1a472a; color: #2ecc71"
                        if val >= 40: return "background-color: #1a3a1a; color: #7dcea0"
                        if val >= 25: return "background-color: #2d2a0d; color: #f39c12"
                        return "background-color: #2d0d0d; color: #e74c3c"

                    def _color_scan_stage(val):
                        colors = {
                            "Emerging": "background-color: #0d2818; color: #2ecc71",
                            "Trending": "background-color: #0d1f2d; color: #3498db",
                            "Basing": "background-color: #2d2a0d; color: #f39c12",
                            "Neutral": "",
                            "Extended": "background-color: #2d1a0d; color: #e67e22",
                            "Breaking Down": "background-color: #2d0d0d; color: #e74c3c",
                        }
                        return colors.get(val, "")

                    def _color_scan_rs(val):
                        if val == "Leader": return "background-color: #0d2818; color: #2ecc71"
                        if val == "Holding": return "background-color: #2d2a0d; color: #f39c12"
                        if val == "Lagging": return "background-color: #2d0d0d; color: #e74c3c"
                        return ""

                    styled = df_display.style.map(
                        _color_scan_score, subset=["ScanScore"]
                    )
                    if "SetupStage" in available_cols:
                        styled = styled.map(_color_scan_stage, subset=["SetupStage"])
                    if "RS_Label" in available_cols:
                        styled = styled.map(_color_scan_rs, subset=["RS_Label"])

                    fmt_dict = {
                        "Price": "${:.2f}",
                        "ScanScore": "{:.1f}",
                        "Convexity": "{:.1f}",
                        "Momentum": "{:.1f}",
                        "RS_Score": lambda x: f"{x:.0f}" if pd.notna(x) else "—",
                        "RSI": "{:.0f}",
                        "Pos52": "{:.0f}%",
                        "vsMA50": "{:+.1f}%",
                        "RevGrowth": lambda x: f"{x:+.0f}%" if pd.notna(x) else "N/A",
                        "ShortPct": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        "Beta": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        "AnalystUpside": lambda x: f"{x:+.0f}%" if pd.notna(x) else "N/A",
                    }
                    active_fmt = {k: v for k, v in fmt_dict.items() if k in available_cols}
                    styled = styled.format(active_fmt, na_rep="—")

                    st.dataframe(styled, hide_index=True, use_container_width=True)

                    # ── Add to Watchlist ──
                    grad_divider()
                    section_header("Add to Watchlist", "#2ecc71")
                    add_c1, add_c2, add_c3 = st.columns([2, 2, 1])
                    with add_c1:
                        add_tickers = st.multiselect(
                            "Select tickers to add",
                            df_show["Ticker"].tolist(),
                            key="scan_add_tickers",
                        )
                    with add_c2:
                        wl_names = list(watchlists.keys())
                        target_wl = st.selectbox(
                            "Target watchlist",
                            wl_names,
                            key="scan_target_wl",
                        )
                    with add_c3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Add Selected", key="scan_add_btn"):
                            if add_tickers and target_wl:
                                added = []
                                for t in add_tickers:
                                    if t not in watchlists[target_wl]:
                                        watchlists[target_wl].append(t)
                                        added.append(t)
                                if added:
                                    save_watchlists(watchlists)
                                    st.success(f"Added {', '.join(added)} to '{target_wl}'")
                                    st.rerun()
                                else:
                                    st.info("All selected tickers already in watchlist.")
                            else:
                                st.warning("Select tickers and a watchlist first.")

                    # Export
                    st.download_button(
                        "Export Scanner Results CSV",
                        df_show.to_csv(index=False),
                        f"scanner_results_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                        key="dl_scanner",
                    )

# ── Fragment: Ticker Management (independent rerun) ────────────────────────────

@st.fragment
def _ticker_management_fragment():
    """Ticker add/remove UI — reruns independently without full page reload."""
    st.markdown("#### Manage Tickers")
    st.caption(f"Active watchlist: **{st.session_state.active_watchlist}** ({len(st.session_state.tickers)} tickers)")

    _add_t = st.text_input("Add ticker", placeholder="e.g. NVDA", key="settings_add_ticker").upper().strip()
    if st.button("Add Ticker", use_container_width=True, key="settings_btn_add"):
        if _add_t and _add_t not in st.session_state.tickers:
            st.session_state.tickers.append(_add_t)
            st.session_state.watchlists[st.session_state.active_watchlist] = st.session_state.tickers
            save_watchlists(st.session_state.watchlists)
            st.success(f"Added {_add_t}")
        elif _add_t in st.session_state.tickers:
            st.warning(f"{_add_t} already in list.")

    _rem_t = st.selectbox("Remove ticker", ["--"] + sorted(st.session_state.tickers), key="settings_remove")
    if st.button("Remove Ticker", use_container_width=True, key="settings_btn_remove"):
        if _rem_t != "--":
            st.session_state.tickers.remove(_rem_t)
            st.session_state.watchlists[st.session_state.active_watchlist] = st.session_state.tickers
            save_watchlists(st.session_state.watchlists)
            st.success(f"Removed {_rem_t}")

    grad_divider()

    # Current tickers display
    st.markdown("**Current tickers:**")
    st.caption(", ".join(sorted(st.session_state.tickers)) if st.session_state.tickers else "No tickers")


# ── TAB: Settings ────────────────────────────────────────────────────────────

with tab_settings:
    section_header("Portfolio Settings", "#8b949e")

    _set_c1, _set_c2 = st.columns(2)

    with _set_c1:
        # ── Ticker Management (in fragment for faster updates) ──
        _ticker_management_fragment()

    with _set_c2:
        # ── Watchlist Management ──
        st.markdown("#### Manage Watchlists")

        _new_wl = st.text_input("New watchlist name", placeholder="e.g. Holdings", key="settings_new_wl")
        if st.button("Create Watchlist", use_container_width=True, key="settings_btn_create_wl"):
            if _new_wl and _new_wl not in st.session_state.watchlists:
                st.session_state.watchlists[_new_wl] = []
                save_watchlists(st.session_state.watchlists)
                st.session_state.active_watchlist = _new_wl
                st.session_state.tickers = []
                st.cache_data.clear()
                st.rerun()

        grad_divider()

        # Import Yahoo Finance CSV
        st.markdown("**Import from Yahoo Finance**")
        st.caption("Export your watchlist from Yahoo Finance (CSV), then upload here.")
        _uploaded = st.file_uploader("Upload CSV", type=["csv"], key="settings_csv", label_visibility="collapsed")
        _import_target = st.selectbox("Import into", wl_names, key="settings_import_target")
        if st.button("Import", use_container_width=True, key="settings_btn_import") and _uploaded:
            _imported = parse_yahoo_csv(_uploaded)
            if _imported:
                _existing = st.session_state.watchlists.get(_import_target, [])
                _merged = list(dict.fromkeys(_existing + _imported))
                st.session_state.watchlists[_import_target] = _merged
                save_watchlists(st.session_state.watchlists)
                if _import_target == st.session_state.active_watchlist:
                    st.session_state.tickers = _merged
                st.cache_data.clear()
                st.success(f"Imported {len(_imported)} tickers into '{_import_target}'")
                st.rerun()
            else:
                st.error("No tickers found in CSV. Expected a 'Symbol' column.")

        grad_divider()

        # Rename
        _rename_val = st.text_input("Rename current watchlist", value=st.session_state.active_watchlist, key="settings_rename")
        if st.button("Rename", use_container_width=True, key="settings_btn_rename"):
            if _rename_val and _rename_val != st.session_state.active_watchlist and _rename_val not in st.session_state.watchlists:
                _tickers_copy = st.session_state.watchlists.pop(st.session_state.active_watchlist)
                st.session_state.watchlists[_rename_val] = _tickers_copy
                st.session_state.active_watchlist = _rename_val
                st.session_state.tickers = _tickers_copy
                save_watchlists(st.session_state.watchlists)
                st.rerun()

        # Duplicate
        _dup_name = st.text_input("Duplicate as", placeholder="e.g. Watchlist Copy", key="settings_dup")
        if st.button("Duplicate Current", use_container_width=True, key="settings_btn_dup"):
            if _dup_name and _dup_name not in st.session_state.watchlists:
                st.session_state.watchlists[_dup_name] = st.session_state.tickers.copy()
                save_watchlists(st.session_state.watchlists)
                st.rerun()

        # Delete (only if more than one)
        if len(wl_names) > 1:
            _del_wl = st.selectbox("Delete watchlist",
                                    ["--"] + [w for w in wl_names if w != st.session_state.active_watchlist],
                                    key="settings_del")
            if st.button("Delete", use_container_width=True, key="settings_btn_del"):
                if _del_wl != "--" and _del_wl in st.session_state.watchlists:
                    del st.session_state.watchlists[_del_wl]
                    save_watchlists(st.session_state.watchlists)
                    st.rerun()

    grad_divider()

    # ── Cache & Data Info ──
    st.markdown("#### Data")
    _info_c1, _info_c2 = st.columns(2)
    with _info_c1:
        st.caption(f"Cache TTL — Price: 5 min | Fund: 30 min | Sentiment: 15 min")
        st.caption(f"Last load: {datetime.now().strftime('%H:%M:%S')}")
    with _info_c2:
        if st.button("Clear Cache & Reload", use_container_width=True, key="settings_clear_cache"):
            st.cache_data.clear()
            st.rerun()
