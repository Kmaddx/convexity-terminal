import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
import json
import os

st.set_page_config(
    page_title="Convexity Terminal",
    page_icon="◣",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base layout ── */
    .block-container { padding-top: 0.75rem; padding-bottom: 0rem; max-width: 100%; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.4rem; }
    .main .block-container { font-family: 'Inter', -apple-system, sans-serif; }

    /* ── Hide Streamlit chrome (keep sidebar toggle only) ── */
    #MainMenu, footer { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
    /* Hide Streamlit Cloud toolbar (share/star/edit/deploy) */
    div[data-testid="stToolbar"] { display: none !important; }
    .stAppToolbar { display: none !important; }
    /* Minimal header — just the sidebar toggle, no background */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
        height: 2.5rem !important;
    }
    /* Hide the running man / status indicator */
    div[data-testid="stStatusWidget"] { display: none !important; }

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
THEMES_FILE      = os.path.join(os.path.dirname(__file__), "themes.json")
SCORES_FILE      = os.path.join(os.path.dirname(__file__), "score_history.json")

DEFAULT_THEMES = {
    "themes": {
        "Space & Satellite": {"etfs": ["UFO", "ARKX"], "tickers": ["RKLB", "ASTS"]},
        "Uranium / Nuclear": {"etfs": ["URA"], "tickers": []},
        "AI Semis": {"etfs": ["SMH", "SOXX"], "tickers": []},
        "Crypto / Digital Assets": {"etfs": ["BITQ", "BITO"], "tickers": []},
        "Biotech": {"etfs": ["XBI"], "tickers": []},
        "Critical Materials": {"etfs": ["REMX"], "tickers": ["UAMY"]},
        "Defense & Drones": {"etfs": ["ITA"], "tickers": []},
        "Renewables / Grid": {"etfs": ["QCLN"], "tickers": []},
        "Non-Profitable Tech": {"etfs": ["ARKK"], "tickers": []},
        "Quantum Computing": {"etfs": ["QTUM"], "tickers": []},
        "Copper & Metals": {"etfs": ["COPX", "XME"], "tickers": []},
        "Robotics / Broad AI": {"etfs": ["BOTZ", "ROBO"], "tickers": []},
        "Fintech": {"etfs": ["FINX"], "tickers": ["HOOD"]},
        "EV & Autonomy": {"etfs": ["DRIV"], "tickers": ["TSLA"]},
        "Infrastructure": {"etfs": ["PAVE"], "tickers": []},
        "Mag 7": {"etfs": ["MAGS"], "tickers": []},
        "Cybersecurity": {"etfs": ["HACK"], "tickers": []},
        "Edge AI": {"etfs": ["BOTZ"], "tickers": []},
        "AI & Data": {"etfs": ["AIQ"], "tickers": ["PLTR"]},
        "AI Software": {"etfs": ["IGV"], "tickers": []},
        "Clean Energy": {"etfs": ["ICLN", "TAN"], "tickers": []},
    }
}

def _migrate_themes(raw):
    """Migrate old flat format {name: [tickers]} to new {themes: {name: {etfs:[], tickers:[]}}}."""
    # Already new format
    if "themes" in raw and isinstance(raw["themes"], dict):
        # Validate each entry has etfs/tickers keys
        for name, val in raw["themes"].items():
            if isinstance(val, list):
                raw["themes"][name] = {"etfs": [], "tickers": val}
            elif isinstance(val, dict):
                val.setdefault("etfs", [])
                val.setdefault("tickers", [])
        return raw
    # Old format: {name: [tickers]}
    new = {"themes": {}}
    # Map old theme names to ETF benchmarks for migration
    etf_map = {
        "Space & Satellite": ["UFO", "ARKX"],
        "Fintech": ["FINX"],
        "Critical Materials": ["REMX"],
        "EV & Autonomy": ["DRIV"],
        "AI & Data": ["AIQ"],
        "AI/HPC": ["SMH"],
        "Nuclear": ["URA"],
        "Digital Assets": ["BITQ", "BITO"],
        "Edge AI": ["BOTZ"],
        "Drones": ["ITA"],
        "BioTech": ["XBI"],
        "BTC Mining": ["BITQ"],
        "Clean Energy": ["ICLN", "TAN"],
        "Photonics & Sensors": ["BOTZ"],
    }
    for name, tickers in raw.items():
        if isinstance(tickers, list):
            new["themes"][name] = {
                "etfs": etf_map.get(name, []),
                "tickers": tickers,
            }
        elif isinstance(tickers, dict):
            tickers.setdefault("etfs", [])
            tickers.setdefault("tickers", [])
            new["themes"][name] = tickers
    return new

def load_themes():
    if os.path.exists(THEMES_FILE):
        try:
            with open(THEMES_FILE) as f:
                raw = json.load(f)
            migrated = _migrate_themes(raw)
            # Save back if migration occurred
            if raw != migrated:
                save_themes(migrated)
            return migrated
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_THEMES))

def save_themes(themes_data):
    try:
        with open(THEMES_FILE, "w") as f:
            json.dump(themes_data, f, indent=2)
    except Exception:
        pass

# ── Auto-theme mapping ──
# Maps tickers to themes using yfinance industry + business description keywords.
# Priority order: description keywords (most specific) > industry match > fallback "Other"
THEME_RULES = {
    "Space & Satellite": {
        "industries": [],
        "desc_keywords": ["satellite", "rocket lab", "launch vehicle", "orbital", "spaceflight",
                          "low earth orbit", "space station", "spaceport"],
        "name_keywords": ["space", "rocket"],
    },
    "Uranium / Nuclear": {
        "industries": [],
        "desc_keywords": ["uranium", "nuclear fuel", "isotope separation", "enrichment",
                          "small modular reactor", "nuclear reactor"],
    },
    "AI Semis": {
        "industries": ["semiconductors", "semiconductor equipment & materials"],
        "desc_keywords": ["semiconductor fabricat", "chip manufactur", "semiconductor company",
                          "designs and sells semiconductor"],
    },
    "Crypto / Digital Assets": {
        "industries": [],
        "desc_keywords": ["bitcoin", "cryptocurrency", "blockchain", "stablecoin",
                          "digital asset", "crypto mining", "bitcoin mining"],
    },
    "Biotech": {
        "industries": ["biotechnology"],
        "desc_keywords": ["biotechnology company", "immunotherapy", "oncology drug"],
    },
    "Critical Materials": {
        "industries": ["other industrial metals & mining", "coking coal"],
        "desc_keywords": ["antimony", "lithium mining", "rare earth", "critical mineral",
                          "critical metal", "strategic mineral"],
    },
    "Defense & Drones": {
        "industries": [],
        "desc_keywords": ["defense contract", "defense system", "drone", "military",
                          "unmanned aerial", "defense department", "tactical"],
    },
    "Fintech": {
        "industries": [],
        "desc_keywords": ["fintech", "trading platform", "brokerage service",
                          "payment processing", "neobank", "commission-free trading"],
    },
    "EV & Autonomy": {
        "industries": ["auto manufacturers"],
        "desc_keywords": ["electric vehicle", "ev charging",
                          "battery electric", "manufactures electric"],
    },
    "Cybersecurity": {
        "industries": [],
        "desc_keywords": ["cybersecurity", "cyber security", "threat detection",
                          "endpoint security", "zero trust"],
    },
    "AI & Data": {
        "industries": [],
        "desc_keywords": ["artificial intelligence", "machine learning platform",
                          "ai infrastructure", "ai industry", "generative ai",
                          "data analytics platform", "full-stack infrastructure for ai"],
    },
    "Edge AI / IoT": {
        "industries": [],
        "desc_keywords": ["edge computing", "internet of things", "embedded system",
                          "industrial computing", "edge ai", "ruggedized computing"],
    },
    "Clean Energy": {
        "industries": ["solar"],
        "desc_keywords": ["solar panel", "wind farm", "clean energy company",
                          "renewable energy company"],
    },
    "Robotics & Automation": {
        "industries": [],
        "desc_keywords": ["robotics company", "industrial automation", "autonomous robot"],
    },
    "Quantum Computing": {
        "industries": [],
        "desc_keywords": ["quantum comput", "quantum processor", "qubit"],
    },
}

META_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta_cache.json")

def _load_meta_cache():
    try:
        if os.path.exists(META_CACHE_FILE):
            with open(META_CACHE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_meta_cache(cache):
    try:
        with open(META_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

_META_DISK_CACHE = _load_meta_cache()

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_ticker_metadata(ticker):
    """Fetch industry, description, and name for theme auto-assignment.
    Falls back to disk cache when rate-limited."""
    try:
        info = yf.Ticker(ticker).info
        # Check if we got real data
        if info and any(info.get(k) for k in ["industry", "sector", "longBusinessSummary"]):
            result = {
                "industry": (info.get("industry") or "").lower(),
                "sector": (info.get("sector") or "").lower(),
                "desc": (info.get("longBusinessSummary") or "").lower(),
                "name": (info.get("shortName") or info.get("longName") or "").lower(),
            }
            # Update disk cache
            _META_DISK_CACHE[ticker] = result
            _save_meta_cache(_META_DISK_CACHE)
            return result
    except Exception:
        pass
    # Fallback to disk cache
    if ticker in _META_DISK_CACHE:
        return _META_DISK_CACHE[ticker]
    return {"industry": "", "sector": "", "desc": "", "name": ""}

def auto_assign_themes(ticker, metadata=None):
    """Auto-assign a ticker to themes based on industry + business description + company name.
    Falls back to yfinance sector if no specific theme matches."""
    if metadata is None:
        metadata = _fetch_ticker_metadata(ticker)
    industry = metadata.get("industry", "")
    desc = metadata.get("desc", "")
    name = metadata.get("name", "").lower()
    matches = []
    for theme_name, rules in THEME_RULES.items():
        # Check industry match
        if industry and industry in [i.lower() for i in rules.get("industries", [])]:
            matches.append(theme_name)
            continue
        # Check description keywords (most specific)
        if desc and any(kw in desc for kw in rules.get("desc_keywords", [])):
            matches.append(theme_name)
            continue
        # Check company name keywords
        if name and any(kw in name for kw in rules.get("name_keywords", [])):
            matches.append(theme_name)
    if matches:
        return matches
    # Fallback: use yfinance sector as a catch-all theme
    sector = metadata.get("sector", "").title()
    return [sector] if sector else ["Other"]

def get_theme_tickers(themes_data):
    """Extract {theme_name: [user_tickers]} from the themes structure."""
    themes_dict = themes_data.get("themes", {})
    return {name: val.get("tickers", []) for name, val in themes_dict.items()}

def get_ticker_themes(themes_data, ticker):
    """Return themes for a ticker. Uses auto-assignment based on industry/description,
    with manual overrides from themes.json taking priority."""
    # Check manual overrides first
    themes_dict = themes_data.get("themes", {})
    manual = [name for name, val in themes_dict.items() if ticker in val.get("tickers", [])]
    if manual:
        return manual
    # Auto-assign from metadata
    return auto_assign_themes(ticker)

def get_theme_etfs(themes_data):
    """Extract {theme_name: [etf_tickers]} from the new themes structure."""
    themes_dict = themes_data.get("themes", {})
    return {name: val.get("etfs", []) for name, val in themes_dict.items()}

def get_all_etf_tickers(themes_data):
    """Get unique set of all ETF benchmark tickers across all themes."""
    etfs = set()
    for val in themes_data.get("themes", {}).values():
        etfs.update(val.get("etfs", []))
    return sorted(etfs)

if "themes" not in st.session_state:
    st.session_state.themes = load_themes()

def load_watchlists():
    """Load named watchlists. Migrates old tickers.json if present."""
    if os.path.exists(WATCHLISTS_FILE):
        try:
            with open(WATCHLISTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    # Migrate from old flat tickers.json if it exists
    if os.path.exists(TICKERS_FILE):
        try:
            with open(TICKERS_FILE) as f:
                old = json.load(f)
            if isinstance(old, list):
                return {"All Tickers": old}
        except Exception:
            pass
    return {"All Tickers": DEFAULT_TICKERS.copy()}

def save_watchlists(wl):
    try:
        with open(WATCHLISTS_FILE, "w") as f:
            json.dump(wl, f, indent=2)
    except Exception:
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
    except Exception:
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

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr_pct(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return (atr / close * 100).iloc[-1]

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

# ── Data Fetching ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(tickers):
    results = []
    for t in tickers:
        try:
            df = yf.download(t, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 50:
                continue
            close = df["Close"].squeeze()
            high  = df["High"].squeeze()
            low   = df["Low"].squeeze()
            vol   = df["Volume"].squeeze()
            price  = close.iloc[-1]
            ma50   = close.rolling(50).mean().iloc[-1]
            ma200  = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
            rsi    = calc_rsi(close).iloc[-1]
            atr_pct = calc_atr_pct(high, low, close)
            w52h   = high.max()
            w52l   = low.min()
            pos52  = ((price - w52l) / (w52h - w52l) * 100) if w52h != w52l else np.nan
            vs50   = ((price / ma50) - 1) * 100
            vs200  = ((price / ma200) - 1) * 100 if not np.isnan(ma200) else np.nan
            avg_vol = vol.iloc[-30:].mean()
            rel_vol = round(vol.iloc[-1] / avg_vol, 2) if avg_vol > 0 else np.nan
            breakout = bool(pos52 >= 95)
            # Period returns for relative strength
            ret_1m  = round(((price / close.iloc[-22]) - 1) * 100, 1) if len(close) >= 22 else None
            ret_3m  = round(((price / close.iloc[-63]) - 1) * 100, 1) if len(close) >= 63 else None
            ret_6m  = round(((price / close.iloc[-126]) - 1) * 100, 1) if len(close) >= 126 else None
            # Last 30 closes for sparkline
            spark_30 = close.iloc[-30:].tolist() if len(close) >= 30 else close.tolist()
            results.append(dict(
                Ticker=t, Price=round(price,2),
                RSI=round(rsi,1), vsMA50=round(vs50,1),
                vsMA200=round(vs200,1) if not np.isnan(vs200) else None,
                ATR_pct=round(atr_pct,1), Pos52=round(pos52,1),
                Low52=round(w52l,2), High52=round(w52h,2),
                RelVol=rel_vol, Breakout=breakout,
                Ret1m=ret_1m, Ret3m=ret_3m, Ret6m=ret_6m,
                Spark30=spark_30,
            ))
        except Exception:
            continue
    return pd.DataFrame(results)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_spy_daily():
    """Fetch SPY 1y daily close & daily returns. Used for down-day RS analysis."""
    try:
        df = yf.download("SPY", period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        close = df["Close"].squeeze()
        daily_ret = close.pct_change().dropna()
        return close, daily_ret
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_spy_returns():
    """SPY benchmark returns for relative strength comparison."""
    try:
        df = yf.download("SPY", period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return {}
        close = df["Close"].squeeze()
        price = close.iloc[-1]
        ret = {}
        if len(close) >= 22:
            ret["1m"] = round(((price / close.iloc[-22]) - 1) * 100, 1)
        if len(close) >= 63:
            ret["3m"] = round(((price / close.iloc[-63]) - 1) * 100, 1)
        if len(close) >= 126:
            ret["6m"] = round(((price / close.iloc[-126]) - 1) * 100, 1)
        return ret
    except Exception:
        return {}

# ── Market Environment ────────────────────────────────────────────────────────

SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLC", "XLRE", "XLB"]

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_environment():
    """Fetch market-wide data for environment scoring.
    Returns dict with VIX, SPX, QQQ, sector, yield, and DXY data.
    """
    env = {}
    try:
        # VIX
        vix_df = yf.download("^VIX", period="1y", interval="1d", progress=False, auto_adjust=True)
        if not vix_df.empty:
            vix_close = vix_df["Close"].squeeze()
            env["vix_level"] = round(float(vix_close.iloc[-1]), 2)
            env["vix_ma20"] = round(float(vix_close.rolling(20).mean().iloc[-1]), 2)
            env["vix_pct_rank"] = round(float(
                (vix_close.iloc[-1] > vix_close.iloc[-252:]).mean() * 100
            ) if len(vix_close) >= 252 else 50, 0)
            env["vix_rising"] = bool(vix_close.iloc[-1] > vix_close.iloc[-5])

        # SPX
        spx_df = yf.download("^GSPC", period="1y", interval="1d", progress=False, auto_adjust=True)
        if not spx_df.empty:
            spx = spx_df["Close"].squeeze()
            env["spx_price"] = round(float(spx.iloc[-1]), 2)
            env["spx_vs_20d"] = round(float((spx.iloc[-1] / spx.rolling(20).mean().iloc[-1] - 1) * 100), 2)
            env["spx_vs_50d"] = round(float((spx.iloc[-1] / spx.rolling(50).mean().iloc[-1] - 1) * 100), 2)
            env["spx_vs_200d"] = round(float((spx.iloc[-1] / spx.rolling(200).mean().iloc[-1] - 1) * 100), 2) if len(spx) >= 200 else None

        # QQQ
        qqq_df = yf.download("QQQ", period="1y", interval="1d", progress=False, auto_adjust=True)
        if not qqq_df.empty:
            qqq = qqq_df["Close"].squeeze()
            env["qqq_price"] = round(float(qqq.iloc[-1]), 2)
            env["qqq_vs_50d"] = round(float((qqq.iloc[-1] / qqq.rolling(50).mean().iloc[-1] - 1) * 100), 2)
            env["qqq_ret_1m"] = round(float((qqq.iloc[-1] / qqq.iloc[-22] - 1) * 100), 1) if len(qqq) >= 22 else None

        # Sector ETFs — % above 50d MA, daily returns
        sector_data = {}
        for etf in SECTOR_ETFS:
            try:
                s_df = yf.download(etf, period="3mo", interval="1d", progress=False, auto_adjust=True)
                if not s_df.empty:
                    s_close = s_df["Close"].squeeze()
                    ma50 = s_close.rolling(50).mean().iloc[-1]
                    ret_1d = round(float((s_close.iloc[-1] / s_close.iloc[-2] - 1) * 100), 2) if len(s_close) >= 2 else 0
                    ret_5d = round(float((s_close.iloc[-1] / s_close.iloc[-5] - 1) * 100), 2) if len(s_close) >= 5 else 0
                    above_50 = bool(s_close.iloc[-1] > ma50)
                    sector_data[etf] = {
                        "ret_1d": ret_1d, "ret_5d": ret_5d,
                        "above_50d": above_50, "price": round(float(s_close.iloc[-1]), 2),
                    }
            except Exception:
                pass
        env["sectors"] = sector_data
        env["sectors_above_50d"] = sum(1 for s in sector_data.values() if s["above_50d"])
        env["sectors_positive_1d"] = sum(1 for s in sector_data.values() if s["ret_1d"] > 0)
        env["sector_leader"] = max(sector_data.items(), key=lambda x: x[1]["ret_5d"])[0] if sector_data else None
        env["sector_laggard"] = min(sector_data.items(), key=lambda x: x[1]["ret_5d"])[0] if sector_data else None

        # 10Y yield
        tnx_df = yf.download("^TNX", period="6mo", interval="1d", progress=False, auto_adjust=True)
        if not tnx_df.empty:
            tnx = tnx_df["Close"].squeeze()
            env["tnx_yield"] = round(float(tnx.iloc[-1]), 2)
            env["tnx_vs_20d"] = round(float((tnx.iloc[-1] / tnx.rolling(20).mean().iloc[-1] - 1) * 100), 2)
            env["tnx_rising"] = bool(tnx.iloc[-1] > tnx.iloc[-5])

        # DXY
        dxy_df = yf.download("DX-Y.NYB", period="6mo", interval="1d", progress=False, auto_adjust=True)
        if not dxy_df.empty:
            dxy = dxy_df["Close"].squeeze()
            env["dxy_level"] = round(float(dxy.iloc[-1]), 2)
            env["dxy_vs_20d"] = round(float((dxy.iloc[-1] / dxy.rolling(20).mean().iloc[-1] - 1) * 100), 2)
            env["dxy_strengthening"] = bool(dxy.iloc[-1] > dxy.iloc[-5])

    except Exception:
        pass
    return env


def calc_market_env_score(env):
    """Score market environment 0-100 across 5 pillars. Returns total, pillar dict, and decision."""
    pillars = {}

    # ── 1. Volatility (0-100, lower VIX = higher score) ──
    vix = env.get("vix_level", 20)
    vix_pct = env.get("vix_pct_rank", 50)
    vix_rising = env.get("vix_rising", False)
    if vix <= 15:
        vol_score = 90
    elif vix <= 20:
        vol_score = 70
    elif vix <= 25:
        vol_score = 45
    elif vix <= 30:
        vol_score = 25
    else:
        vol_score = 10
    # Penalize if VIX is rising and elevated
    if vix_rising and vix > 20:
        vol_score = max(vol_score - 15, 0)
    # Penalize high percentile rank (elevated vs history)
    if vix_pct > 75:
        vol_score = max(vol_score - 10, 0)
    pillars["Volatility"] = {"score": round(vol_score), "weight": 25}

    # ── 2. Trend (0-100, price vs MAs) ──
    spx_20 = env.get("spx_vs_20d", 0)
    spx_50 = env.get("spx_vs_50d", 0)
    spx_200 = env.get("spx_vs_200d", 0)
    trend_score = 0
    # Above 200d = base strength (30 pts)
    if spx_200 is not None:
        if spx_200 > 0:
            trend_score += min(spx_200, 5) / 5 * 30
        else:
            trend_score += max(0, 30 + spx_200 * 3)  # lose 3pts per % below
    else:
        trend_score += 15  # no data, neutral
    # Above 50d = intermediate trend (40 pts)
    if spx_50 > 0:
        trend_score += min(spx_50, 5) / 5 * 40
    else:
        trend_score += max(0, 40 + spx_50 * 4)
    # Above 20d = near-term trend (30 pts)
    if spx_20 > 0:
        trend_score += min(spx_20, 3) / 3 * 30
    else:
        trend_score += max(0, 30 + spx_20 * 5)
    trend_score = max(0, min(100, trend_score))
    # QQQ confirmation
    qqq_50 = env.get("qqq_vs_50d", 0)
    if qqq_50 < -3:
        trend_score = max(trend_score - 10, 0)
    pillars["Trend"] = {"score": round(trend_score), "weight": 25}

    # ── 3. Breadth (0-100, sector participation) ──
    above_50d = env.get("sectors_above_50d", 6)
    pos_1d = env.get("sectors_positive_1d", 6)
    total_sectors = len(SECTOR_ETFS)
    # % sectors above 50d MA (60 pts)
    breadth_score = (above_50d / total_sectors) * 60
    # % sectors positive today (40 pts)
    breadth_score += (pos_1d / total_sectors) * 40
    pillars["Breadth"] = {"score": round(breadth_score), "weight": 20}

    # ── 4. Momentum (0-100, sector strength & rotation) ──
    sectors = env.get("sectors", {})
    if sectors:
        ret_5d_vals = [s["ret_5d"] for s in sectors.values()]
        avg_5d = np.mean(ret_5d_vals) if ret_5d_vals else 0
        positive_5d = sum(1 for r in ret_5d_vals if r > 0)
        # Average sector 5d return scaled (50 pts)
        mom_score = min(max((avg_5d + 3) / 6, 0), 1) * 50
        # Participation — how many sectors positive over 5d (50 pts)
        mom_score += (positive_5d / len(ret_5d_vals)) * 50 if ret_5d_vals else 25
    else:
        mom_score = 50
    pillars["Momentum"] = {"score": round(mom_score), "weight": 15}

    # ── 5. Macro (0-100, yield + dollar headwinds) ──
    macro_score = 50  # neutral baseline
    tnx = env.get("tnx_yield", 4.0)
    tnx_rising = env.get("tnx_rising", False)
    dxy_str = env.get("dxy_strengthening", False)
    # Lower yields better for growth stocks
    if tnx < 3.5:
        macro_score += 25
    elif tnx < 4.0:
        macro_score += 15
    elif tnx > 4.5:
        macro_score -= 15
    elif tnx > 5.0:
        macro_score -= 25
    # Rising yields = headwind
    if tnx_rising:
        macro_score -= 10
    # Strengthening dollar = headwind for risk
    if dxy_str:
        macro_score -= 10
    macro_score = max(0, min(100, macro_score))
    pillars["Macro"] = {"score": round(macro_score), "weight": 15}

    # ── Weighted total ──
    total_weight = sum(p["weight"] for p in pillars.values())
    total = sum(p["score"] * p["weight"] for p in pillars.values()) / total_weight
    total = round(total)

    # Decision
    if total >= 65:
        decision = "OPPORTUNITY"
        decision_sub = "Conditions favor adding to positions"
    elif total >= 45:
        decision = "SELECTIVE"
        decision_sub = "Pick your spots — quality setups only"
    else:
        decision = "DEFENSIVE"
        decision_sub = "Preserve capital — wait for better entries"

    return total, pillars, decision, decision_sub


def calc_execution_window(df_all):
    """Assess execution quality from portfolio RS data.
    Returns dict with breakout/leader/pullback health signals.
    """
    result = {}
    n = len(df_all)
    if n == 0:
        return {"score": 0, "breakouts_working": "N/A", "leaders_holding": "N/A",
                "pullbacks_bought": "N/A", "follow_through": "N/A"}

    # Breakouts working? — stocks near 52w high that are holding
    breakout_tickers = df_all[df_all["Pos52"] >= 90]
    if len(breakout_tickers) > 0:
        held = breakout_tickers[breakout_tickers["vsMA50"] > 0]
        result["breakouts_working"] = "Yes" if len(held) / len(breakout_tickers) >= 0.6 else "No"
    else:
        result["breakouts_working"] = "None active"

    # Leaders holding? — RS Leaders still above 50d MA
    if "RS_Label" in df_all.columns:
        leaders = df_all[df_all["RS_Label"] == "Leader"]
        if len(leaders) > 0:
            holding = leaders[leaders["vsMA50"] > -5]
            result["leaders_holding"] = "Yes" if len(holding) / len(leaders) >= 0.6 else "Fading"
        else:
            result["leaders_holding"] = "None"
    else:
        result["leaders_holding"] = "N/A"

    # Pullbacks getting bought? — stocks 5-15% off high with RSI 40-55 (basing)
    pullback = df_all[(df_all["Pos52"] >= 50) & (df_all["Pos52"] <= 85) & (df_all["RSI"] >= 40) & (df_all["RSI"] <= 55)]
    if len(pullback) > 0:
        bought = pullback[pullback["vsMA50"] > -3]
        result["pullbacks_bought"] = "Yes" if len(bought) / len(pullback) >= 0.5 else "No"
    else:
        result["pullbacks_bought"] = "N/A"

    # Follow-through — trending stocks making progress
    trending = df_all[df_all["SetupStage"].isin(["Trending", "Emerging"])]
    if len(trending) >= 2:
        strong_ft = trending[trending["Ret1m"].apply(lambda x: _safe(x, 0)) > 0]
        ratio = len(strong_ft) / len(trending)
        if ratio >= 0.6:
            result["follow_through"] = "Strong"
        elif ratio >= 0.35:
            result["follow_through"] = "Weak"
        else:
            result["follow_through"] = "None"
    else:
        result["follow_through"] = "N/A"

    # Execution score
    score_map = {"Yes": 25, "Strong": 25, "Weak": 10, "No": 0, "Fading": 5,
                 "None active": 10, "None": 5, "N/A": 10}
    exec_score = (
        score_map.get(result["breakouts_working"], 10) +
        score_map.get(result["leaders_holding"], 10) +
        score_map.get(result["pullbacks_bought"], 10) +
        score_map.get(result["follow_through"], 10)
    )
    result["score"] = exec_score
    return result


@st.cache_data(ttl=300, show_spinner=False)
def calc_downday_rs(tickers, spy_close, spy_daily_ret):
    """Calculate relative strength on SPY down-days for each ticker.

    Core idea from ZA: "What isn't going lower?" — stocks that hold up on market
    down-days are being accumulated by institutions. This is one of the clearest
    signals the market can give you.

    Returns DataFrame with columns:
    - Ticker, DownDayRS (avg excess return on SPY down-days, higher = stronger),
    - DownDayWinRate (% of SPY down-days where stock outperformed SPY),
    - RS_Score (0-100 composite), RS_Label (Leader/Holding/Lagging),
    - RecentRS_1m, RecentRS_3m (period relative strength vs SPY),
    - EarlyBottom (bool — stock's 20d low came before SPY's 20d low),
    - DaysSinceLow (how many days since the stock's recent low)
    """
    if spy_close.empty or spy_daily_ret.empty:
        return pd.DataFrame()

    # Identify SPY down-days (bottom 40% of daily returns ≈ meaningfully negative days)
    threshold = spy_daily_ret.quantile(0.40)
    down_days = spy_daily_ret[spy_daily_ret <= threshold].index

    # Also get recent 1m down-days for recency weighting
    recent_cutoff_1m = spy_daily_ret.index[-22] if len(spy_daily_ret) >= 22 else spy_daily_ret.index[0]
    recent_down_days = [d for d in down_days if d >= recent_cutoff_1m]

    results = []
    for t in tickers:
        try:
            df = yf.download(t, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 50:
                continue
            close = df["Close"].squeeze()
            stock_ret = close.pct_change().dropna()

            # Align dates
            common = stock_ret.index.intersection(spy_daily_ret.index)
            if len(common) < 30:
                continue
            s_ret = stock_ret.reindex(common)
            spy_r = spy_daily_ret.reindex(common)

            # Down-day analysis (all down-days)
            dd_idx = [d for d in down_days if d in common]
            if len(dd_idx) < 5:
                continue
            stock_on_dd = s_ret.loc[dd_idx]
            spy_on_dd = spy_r.loc[dd_idx]
            excess_on_dd = stock_on_dd - spy_on_dd  # positive = outperformed on down-day

            dd_rs = round(excess_on_dd.mean() * 100, 3)  # avg daily excess on down-days (%)
            dd_win_rate = round((excess_on_dd > 0).sum() / len(excess_on_dd) * 100, 1)

            # Recent down-day RS (last month only) — more weight to current behaviour
            recent_dd_idx = [d for d in recent_down_days if d in common]
            if len(recent_dd_idx) >= 3:
                recent_excess = (s_ret.loc[recent_dd_idx] - spy_r.loc[recent_dd_idx]).mean() * 100
            else:
                recent_excess = dd_rs  # fall back to full-period

            # Period RS (simple returns vs SPY)
            price = close.iloc[-1]
            rs_1m = round(((price / close.iloc[-22]) - 1) * 100 - ((spy_close.iloc[-1] / spy_close.iloc[-22]) - 1) * 100, 1) if len(close) >= 22 and len(spy_close) >= 22 else 0
            rs_3m = round(((price / close.iloc[-63]) - 1) * 100 - ((spy_close.iloc[-1] / spy_close.iloc[-63]) - 1) * 100, 1) if len(close) >= 63 and len(spy_close) >= 63 else 0

            # Early bottoming: did the stock make its 20-day low before SPY?
            lookback = min(60, len(close) - 1, len(spy_close) - 1)
            stock_low_idx = close.iloc[-lookback:].idxmin()
            spy_low_idx = spy_close.iloc[-lookback:].idxmin()
            early_bottom = bool(stock_low_idx < spy_low_idx) if pd.notna(stock_low_idx) and pd.notna(spy_low_idx) else False
            days_since_low = (close.index[-1] - stock_low_idx).days if pd.notna(stock_low_idx) else 0

            # Composite RS Score (0-100)
            # 35% down-day win rate, 25% recent down-day excess, 25% 1m RS, 15% early bottom bonus
            score = 0
            score += min(max((dd_win_rate - 30) / 40, 0), 1.0) * 35  # 30-70% win rate maps to 0-35
            score += min(max((recent_excess + 1) / 2, 0), 1.0) * 25  # -1% to +1% maps to 0-25
            score += min(max((rs_1m + 15) / 30, 0), 1.0) * 25  # -15% to +15% maps to 0-25
            score += (15 if early_bottom else 0)
            score = round(min(score, 100), 1)

            results.append({
                "Ticker": t,
                "DownDayRS": round(dd_rs, 2),
                "DownDayWinRate": dd_win_rate,
                "RecentDDExcess": round(recent_excess, 2),
                "RS_1m": rs_1m,
                "RS_3m": rs_3m,
                "RS_Score": score,
                "EarlyBottom": early_bottom,
                "DaysSinceLow": days_since_low,
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df_rs = pd.DataFrame(results).sort_values("RS_Score", ascending=False)
    # Rank and label
    df_rs["RS_Rank"] = range(1, len(df_rs) + 1)
    n = len(df_rs)
    def _label(rank):
        if rank <= max(1, n * 0.25):
            return "Leader"
        elif rank <= max(2, n * 0.60):
            return "Holding"
        else:
            return "Lagging"
    df_rs["RS_Label"] = df_rs["RS_Rank"].apply(_label)
    return df_rs


@st.cache_data(ttl=300, show_spinner=False)
def fetch_etf_benchmark_data(etf_tickers):
    """Fetch 1-year daily price data for ETF benchmark tickers. Returns DataFrame with returns."""
    results = []
    for t in etf_tickers:
        try:
            df = yf.download(t, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 22:
                continue
            close = df["Close"].squeeze()
            price = close.iloc[-1]
            ret_1m = round(((price / close.iloc[-22]) - 1) * 100, 1) if len(close) >= 22 else None
            ret_3m = round(((price / close.iloc[-63]) - 1) * 100, 1) if len(close) >= 63 else None
            ret_6m = round(((price / close.iloc[-126]) - 1) * 100, 1) if len(close) >= 126 else None
            ret_1y = round(((price / close.iloc[0]) - 1) * 100, 1) if len(close) >= 200 else None
            # YTD return
            ytd_start = close[close.index >= f"{datetime.now().year}-01-01"]
            ret_ytd = round(((price / ytd_start.iloc[0]) - 1) * 100, 1) if len(ytd_start) > 1 else None
            results.append(dict(
                Ticker=t, Price=round(price, 2),
                Ret1m=ret_1m, Ret3m=ret_3m, Ret6m=ret_6m,
                Ret1y=ret_1y, RetYTD=ret_ytd,
            ))
        except Exception:
            continue
    return pd.DataFrame(results)

FUND_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fund_cache.json")

def _load_fund_cache():
    """Load cached fundamental data from disk. Uses any age as fallback — stale data > no data."""
    try:
        if os.path.exists(FUND_CACHE_FILE):
            with open(FUND_CACHE_FILE) as f:
                cache = json.load(f)
            return cache.get("data", {})
    except Exception:
        pass
    return {}

def _save_fund_cache(data_by_ticker):
    """Save fundamental data to disk cache."""
    try:
        cache = {"_ts": datetime.now().timestamp(), "data": data_by_ticker}
        with open(FUND_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(tickers):
    """Single call to yf.Ticker().info per ticker — covers both analyst/fundamental and valuation data.
    Includes rate-limit mitigation: delays between requests + disk cache fallback."""
    import time
    results = []
    _disk_cache = _load_fund_cache()
    _new_cache = {}
    _rate_limited = False
    for idx, t in enumerate(tickers):
        row = {"Ticker": t}
        # Rate limit mitigation: add delay between requests (skip first)
        if idx > 0:
            time.sleep(1.5)
        try:
            obj  = yf.Ticker(t)
            info = obj.info
            # Detect rate-limiting: info dict is empty or missing key financial fields
            _has_real_data = info and any(info.get(k) is not None for k in
                ["currentPrice", "revenueGrowth", "marketCap", "freeCashflow"])
            if not _has_real_data:
                _rate_limited = True
                # Fall back to disk cache for this ticker
                if t in _disk_cache:
                    row.update(_disk_cache[t])
                results.append(row)
                continue

            # Analyst targets & recommendations
            target_mean = info.get("targetMeanPrice")
            current     = info.get("currentPrice") or info.get("regularMarketPrice")
            upside      = round((target_mean / current - 1) * 100, 1) if target_mean and current else None
            short_pct   = info.get("shortPercentOfFloat")
            rev_growth  = info.get("revenueGrowth")
            earnings_ts = info.get("earningsDate") or info.get("earningsTimestamp")
            next_earnings = None
            if isinstance(earnings_ts, list) and earnings_ts:
                try:
                    next_earnings = datetime.fromtimestamp(earnings_ts[0]).strftime("%Y-%m-%d")
                except Exception:
                    pass
            elif isinstance(earnings_ts, (int, float)) and earnings_ts:
                try:
                    next_earnings = datetime.fromtimestamp(earnings_ts).strftime("%Y-%m-%d")
                except Exception:
                    pass

            beta_val = info.get("beta")
            beta_reliable = beta_val is not None and 0 < beta_val <= 5
            row.update({
                "Beta": beta_val,
                "BetaReliable": beta_reliable,
                "ShortPct": round(short_pct * 100, 1) if short_pct else None,
                "MarketCap": info.get("marketCap"),
                "TargetMean": round(target_mean, 2) if target_mean else None,
                "TargetLow": info.get("targetLowPrice"),
                "TargetHigh": info.get("targetHighPrice"),
                "NumAnalysts": info.get("numberOfAnalystOpinions", 0),
                "Recommendation": info.get("recommendationKey", ""),
                "RevenueGrowth": round(rev_growth * 100, 1) if rev_growth else None,
                "AnalystUpside": upside,
                "NextEarnings": next_earnings,
            })

            # Valuation multiples
            ps = info.get("priceToSalesTrailing12Months")
            ev = info.get("enterpriseToEbitda")
            gm = info.get("grossMargins")
            row["PS_Current"]  = round(ps, 2) if ps else None
            row["EV_EBITDA"]   = round(ev, 1) if ev and ev > 0 else None
            row["GrossMargin"] = round(gm * 100, 1) if gm else None

            # Rule of 40
            prof = info.get("profitMargins")
            row["RevGrowthPct"] = round(rev_growth * 100, 1) if rev_growth is not None else None
            row["Rule40"] = round(rev_growth * 100 + prof * 100, 1) if (rev_growth is not None and prof is not None) else None

            # FCF & cash runway
            fcf        = info.get("freeCashflow")
            total_cash = info.get("totalCash") or 0
            row["FCFPositive"]       = bool(fcf and fcf > 0)
            row["FCFValue"]          = fcf
            row["CashRunwayMonths"]  = (round((total_cash / abs(fcf)) * 12, 1)
                                        if fcf and fcf < 0 and total_cash else None)

            # Ownership
            row["InsiderPct"]  = round((info.get("heldPercentInsiders")  or 0) * 100, 1)
            row["InstitPct"]   = round((info.get("heldPercentInstitutions") or 0) * 100, 1)
            row["DaysToCover"] = info.get("shortRatio")

            # Extended hours
            post_chg = info.get("postMarketChangePercent")
            pre_chg = info.get("preMarketChangePercent")
            row["PostMktChg"] = round(post_chg, 2) if post_chg is not None else None
            row["PreMktChg"] = round(pre_chg, 2) if pre_chg is not None else None

            # Historical P/S range (3 years)
            shares = info.get("sharesOutstanding")
            if shares and ps:
                try:
                    hist = yf.download(t, period="3y", interval="1mo",
                                       progress=False, auto_adjust=True)
                    q_fin = getattr(obj, "quarterly_income_stmt", None)
                    if q_fin is None or (hasattr(q_fin, "empty") and q_fin.empty):
                        q_fin = getattr(obj, "quarterly_financials", None)

                    rev_row = None
                    if q_fin is not None and not q_fin.empty:
                        for name in ["Total Revenue", "Revenue", "Net Revenue"]:
                            if name in q_fin.index:
                                rev_row = name
                                break

                    if not hist.empty and rev_row:
                        monthly_close = hist["Close"].squeeze()
                        rev_q = q_fin.loc[rev_row].sort_index()
                        ttm   = rev_q.rolling(4).sum().dropna()

                        if not ttm.empty:
                            ttm.index          = ttm.index.tz_localize(None) if ttm.index.tz else ttm.index
                            monthly_close.index = monthly_close.index.tz_localize(None) if monthly_close.index.tz else monthly_close.index

                            ttm_monthly = (ttm.resample("MS").last()
                                              .reindex(monthly_close.index, method="ffill"))
                            ps_hist = (monthly_close * shares / ttm_monthly).dropna()
                            ps_hist = ps_hist[(ps_hist > 0) & (ps_hist < 500)]

                            if len(ps_hist) >= 6:
                                row["PS_3yr_Min"] = round(float(ps_hist.min()), 2)
                                row["PS_3yr_Max"] = round(float(ps_hist.max()), 2)
                                row["PS_3yr_Avg"] = round(float(ps_hist.mean()), 2)
                                rng = row["PS_3yr_Max"] - row["PS_3yr_Min"]
                                if rng > 0:
                                    row["PS_HistPos"] = round(
                                        (ps - row["PS_3yr_Min"]) / rng * 100, 1)
                except Exception:
                    pass
        except Exception:
            # Rate-limited or error — try disk cache
            if t in _disk_cache:
                row.update(_disk_cache[t])
        # Save successful data for cache
        if len(row) > 1:  # more than just Ticker
            _new_cache[t] = {k: v for k, v in row.items()
                            if k != "Ticker" and not isinstance(v, (list, dict))
                            and v is not None}
        results.append(row)
    # Save to disk cache if we got any real data
    if _new_cache and not _rate_limited:
        _save_fund_cache(_new_cache)
    elif _rate_limited and _disk_cache:
        # Merge: keep old cache entries, add any new ones we managed to get
        merged = {**_disk_cache, **_new_cache}
        _save_fund_cache(merged)
    return pd.DataFrame(results)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_extras(tickers):
    import time
    results = []
    for idx, t in enumerate(tickers):
        row = {"Ticker": t, "InsiderSignal": "N/A", "InsiderNet": 0,
               "EarningsBeats": None, "Headlines": [], "InsiderBuys": []}
        if idx > 0:
            time.sleep(1.0)
        try:
            obj = yf.Ticker(t)
            try:
                raw_news = obj.news or []
                headlines = []
                for n in raw_news[:5]:
                    title = (n.get("title") or (n.get("content") or {}).get("title", ""))
                    pub   = (n.get("publisher") or
                             (n.get("content") or {}).get("provider", {}).get("displayName", ""))
                    ts    = n.get("providerPublishTime") or n.get("pubDate", "")
                    if title:
                        try:
                            date_str = datetime.fromtimestamp(int(ts)).strftime("%b %d") if isinstance(ts, (int, float)) else str(ts)[:10]
                        except Exception:
                            date_str = ""
                        headlines.append({"date": date_str, "title": title, "pub": pub})
                row["Headlines"] = headlines
            except Exception:
                pass
            try:
                ins = obj.insider_transactions
                if ins is not None and not ins.empty:
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                    if isinstance(ins.index, pd.DatetimeIndex):
                        recent = ins[ins.index >= cutoff]
                    else:
                        date_col = next((c for c in ins.columns if "date" in c.lower()), None)
                        if date_col:
                            ins[date_col] = pd.to_datetime(ins[date_col], errors="coerce")
                            recent = ins[ins[date_col] >= cutoff]
                        else:
                            recent = ins
                    tx_col = next((c for c in recent.columns
                                   if "text" in c.lower()), None)
                    if tx_col is None:
                        tx_col = next((c for c in recent.columns
                                       if "transaction" in c.lower()), None)
                    if tx_col is not None:
                        tx = recent[tx_col].astype(str).str.lower()
                        buy_mask  = tx.str.contains("buy|purchase|acquisition", na=False)
                        sell_mask = tx.str.contains("sell|sale|disposition", na=False)
                        buys  = int(buy_mask.sum())
                        sells = int(sell_mask.sum())
                        # Capture buy details for display
                        buy_details = []
                        if buys > 0:
                            buy_rows = recent[buy_mask]
                            for _, br in buy_rows.iterrows():
                                detail = {}
                                detail["insider"] = br.get("Insider", "Unknown")
                                detail["position"] = br.get("Position", "")
                                date_val = br.get("Start Date", "")
                                if pd.notna(date_val):
                                    try:
                                        detail["date"] = pd.to_datetime(date_val).strftime("%b %d")
                                    except Exception:
                                        detail["date"] = str(date_val)[:10]
                                else:
                                    detail["date"] = ""
                                val = br.get("Value", 0)
                                detail["value"] = float(val) if pd.notna(val) else 0
                                shares = br.get("Shares", 0)
                                detail["shares"] = int(shares) if pd.notna(shares) else 0
                                buy_details.append(detail)
                        row["InsiderBuys"] = buy_details
                    else:
                        buys, sells = 0, 0
                    net = int(buys) - int(sells)
                    row["InsiderNet"] = net
                    row["InsiderSignal"] = ("Buying" if net > 0
                                            else "Selling" if net < 0 else "Neutral")
            except Exception:
                pass
            try:
                eh = obj.earnings_history
                if eh is not None and not eh.empty:
                    cols_lower = [c.lower() for c in eh.columns]
                    rep_col = next((eh.columns[i] for i, c in enumerate(cols_lower)
                                    if "report" in c or "actual" in c), None)
                    est_col = next((eh.columns[i] for i, c in enumerate(cols_lower)
                                    if "estimate" in c), None)
                    if rep_col and est_col:
                        recent4 = eh.tail(4)
                        beats = int((recent4[rep_col] > recent4[est_col]).sum())
                        row["EarningsBeats"] = f"{beats}/4"
            except Exception:
                pass
        except Exception:
            pass
        results.append(row)
    return pd.DataFrame(results)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stocktwits(tickers):
    out = {}
    for t in tickers:
        try:
            r = requests.get(
                f"https://api.stocktwits.com/api/2/streams/symbol/{t}.json",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=6,
            )
            if r.status_code != 200:
                continue
            msgs = r.json().get("messages", [])
            bulls = sum(1 for m in msgs if
                        (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bullish")
            bears = sum(1 for m in msgs if
                        (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bearish")
            total = bulls + bears
            out[t] = {
                "bull_pct":  round(bulls / total * 100) if total else None,
                "msg_count": len(msgs),
            }
        except Exception:
            pass
    return out

# ── Scanner: Universe Sourcing ────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def scan_yahoo_screener(min_mcap=200_000_000, max_mcap=10_000_000_000, min_rev_growth=0.10):
    """Use yfinance EquityQuery to find small/mid-cap growth stocks."""
    try:
        from yfinance import EquityQuery, screen
        q = EquityQuery("AND", [
            EquityQuery("GT", ["lastclosemarketcap.lasttwelvemonths", min_mcap]),
            EquityQuery("LT", ["lastclosemarketcap.lasttwelvemonths", max_mcap]),
            EquityQuery("GT", ["quarterlyrevenuegrowth.quarterly", min_rev_growth]),
        ])
        result = screen(q, sortField="lastclosemarketcap.lasttwelvemonths",
                        sortAsc=False, size=100)
        if result and "quotes" in result:
            return [q["symbol"] for q in result["quotes"] if "." not in q.get("symbol", ".")]
        return []
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def scan_yahoo_predefined(screener_key="most_actives"):
    """Fetch tickers from Yahoo predefined screeners (most_actives, small_cap_gainers, etc.)."""
    try:
        result = yf.screen(yf.PREDEFINED_SCREENER_QUERIES[screener_key])
        if result and "quotes" in result:
            return [q["symbol"] for q in result["quotes"] if "." not in q.get("symbol", ".")]
        return []
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def scan_finviz():
    """Scrape FINVIZ screener for small-cap stocks with technical setups."""
    try:
        url = ("https://finviz.com/screener.ashx?v=111"
               "&f=cap_smallover,fa_salesqoq_o10,sh_avgvol_o200"
               "&ft=4&o=-marketcap")
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        # Parse ticker symbols from FINVIZ HTML table
        import re
        # FINVIZ table links: <a href="quote.ashx?t=TICKER"...>TICKER</a>
        matches = re.findall(r'quote\.ashx\?t=([A-Z]{1,5})&', resp.text)
        # Deduplicate while preserving order
        seen = set()
        tickers = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                tickers.append(m)
        return tickers[:100]
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def scan_yahoo_trending():
    """Get trending tickers from Yahoo Finance."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US?count=50"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        return [q["symbol"] for q in quotes if "." not in q.get("symbol", "")]
    except Exception:
        return []


def score_scanner_candidates(tickers, spy_close, spy_daily_ret):
    """Lightweight scoring pipeline for scanner candidates.
    Fetches price + fundamental data and scores each ticker.
    Returns a DataFrame with scores and key metrics.
    """
    if not tickers:
        return pd.DataFrame()

    # Fetch price data (technical indicators)
    df_price = fetch_price_data(tickers)
    if df_price.empty:
        return pd.DataFrame()

    # Fetch fundamentals (lighter — just info dict)
    df_fund = fetch_fundamentals(tickers)

    # Merge
    df = df_price.merge(df_fund, on="Ticker", how="left")

    # Calculate scores
    scores = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        momentum = calc_momentum_score(rd)
        convexity = calc_convexity_score(rd)
        stage = calc_setup_stage(rd)
        scores.append({
            "Ticker": rd["Ticker"],
            "Price": rd.get("Price"),
            "MCap": rd.get("MarketCap"),
            "RSI": rd.get("RSI"),
            "Pos52": rd.get("Pos52"),
            "vsMA50": rd.get("vsMA50"),
            "RevGrowth": rd.get("RevenueGrowth"),
            "ShortPct": rd.get("ShortPct"),
            "Beta": rd.get("Beta"),
            "AnalystUpside": rd.get("AnalystUpside"),
            "Momentum": momentum,
            "Convexity": convexity,
            "SetupStage": stage,
        })
    df_scored = pd.DataFrame(scores)

    # Down-day RS if SPY data available
    if not spy_close.empty and not spy_daily_ret.empty:
        df_rs = calc_downday_rs(tickers, spy_close, spy_daily_ret)
        if not df_rs.empty:
            df_scored = df_scored.merge(
                df_rs[["Ticker", "RS_Score", "RS_Label"]],
                on="Ticker", how="left"
            )

    # Composite scanner score: 40% Convexity + 30% Momentum + 30% RS
    if "RS_Score" not in df_scored.columns:
        df_scored["RS_Score"] = None
        df_scored["RS_Label"] = None
    df_scored["ScanScore"] = df_scored.apply(
        lambda r: round(
            0.40 * _safe(r.get("Convexity"), 0) +
            0.30 * _safe(r.get("Momentum"), 0) +
            0.30 * _safe(r.get("RS_Score"), 0), 1
        ), axis=1
    )
    df_scored = df_scored.sort_values("ScanScore", ascending=False).reset_index(drop=True)
    return df_scored


# ── Scoring ──────────────────────────────────────────────────────────────────

def _safe(val, default=0):
    """Return default if val is None or NaN."""
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val

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
    except Exception:
        pass

def load_score_history():
    """Load score history from local JSON."""
    if os.path.exists(SCORES_FILE):
        try:
            with open(SCORES_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def calc_momentum_score(row):
    pos52 = _safe(row.get("Pos52"), 50)
    rsi   = _safe(row.get("RSI"), 50)
    vs50  = _safe(row.get("vsMA50"), 0)
    vs200 = _safe(row.get("vsMA200"), 0)
    s1 = (pos52 / 100) * 40
    s2 = min(max(rsi - 30, 0) / 40, 1.0) * 20
    s3 = min(max(vs50  + 20, 0) / 40, 1.0) * 20
    s4 = min(max(vs200 + 20, 0) / 40, 1.0) * 20
    return round(s1 + s2 + s3 + s4, 1)

def calc_asymmetry_score(row, weights=None):
    if weights is None:
        weights = {}
    w_upside    = weights.get("Analyst Upside", 15)
    w_beta      = weights.get("Beta", 25)
    w_atr       = weights.get("Volatility (ATR%)", 20)
    w_short     = weights.get("Short Interest", 20)
    w_rsi       = weights.get("RSI Positioning", 15)
    w_consensus = weights.get("Analyst Consensus", 5)
    components = {}
    upside = _safe(row.get("AnalystUpside"))
    # Scale analyst upside confidence by number of analysts covering the stock.
    # Note: short interest data from yfinance is ~2 weeks stale (FINRA bi-monthly reporting).
    num_analysts = int(_safe(row.get("NumAnalysts"), 0))
    analyst_conf = 0.0 if num_analysts == 0 else min(1.0, 0.4 + 0.2 * num_analysts)
    # 0 analysts → 0.0 (no signal), 1 → 0.6, 2 → 0.8, 3+ → 1.0
    components["Analyst Upside"] = (min(upside / 80, 1.0) * w_upside if upside > 0 else 0) * analyst_conf
    beta = _safe(row.get("Beta"))
    beta_reliable = row.get("BetaReliable", True)
    if not beta_reliable:
        beta = 1.5  # fallback for missing/unreliable beta (average for small-cap growth)
    if beta > 0:
        components["Beta"] = min(beta / 2.5, 1.0) * w_beta
    else:
        components["Beta"] = w_beta * 0.5
    atr = _safe(row.get("ATR_pct"))
    components["Volatility (ATR%)"] = min(atr / 12, 1.0) * w_atr
    short = _safe(row.get("ShortPct"))
    components["Short Interest"] = min(short / 20, 1.0) * w_short
    rsi = _safe(row.get("RSI"), 50)
    if   35 <= rsi <= 60:  rsi_score = 1.0
    elif 60 < rsi <= 75:   rsi_score = 1.0 - (rsi - 60) / 25
    elif rsi > 75:         rsi_score = 0.0
    else:                  rsi_score = max(rsi / 35, 0) * 0.5
    components["RSI Positioning"] = rsi_score * w_rsi
    rec_map = {"strong_buy": 1.0, "buy": 0.8, "hold": 0.5, "sell": 0.2, "strong_sell": 0.0}
    rec = str(row.get("Recommendation") or "").lower().replace(" ", "_")
    components["Analyst Consensus"] = rec_map.get(rec, 0.5) * w_consensus
    return round(sum(components.values()), 1), components

def calc_value_growth_score(row):
    """
    Score 0-100. Implements the Value-Growth Score (three-legged stool framework):
      Leg 1 — Undervalued (low P/S vs own history)   40 pts
      Leg 2 — Revenue acceleration (growth + Rule of 40) 35 pts
      Leg 3 — Sector/price momentum                  25 pts
    """
    components = {}

    # Leg 1: Undervalued
    ps_pos = _safe(row.get("PS_HistPos"))
    if ps_pos > 0:
        components["Undervalued (P/S)"] = (1 - ps_pos / 100) * 40
    else:
        ps = _safe(row.get("PS_Current"))
        components["Undervalued (P/S)"] = min(1 / ps * 4, 1.0) * 40 if ps > 0 else 20

    # Leg 2: Revenue acceleration
    rev_g = _safe(row.get("RevGrowthPct"))
    rule40 = _safe(row.get("Rule40"))
    components["Revenue Growth"] = min(max(rev_g, 0) / 60, 1.0) * 30 + min(max(rule40 - 20, 0) / 40, 1.0) * 5

    # Leg 3: Momentum (RSI + above MA50)
    rsi  = _safe(row.get("RSI"), 50)
    vs50 = _safe(row.get("vsMA50"))
    components["Sector Momentum"] = min(max(rsi - 40, 0) / 30, 1.0) * 15 + min(max(vs50 + 10, 0) / 20, 1.0) * 10

    return round(sum(components.values()), 1), components

def calc_convexity_score(row, weights=None):
    """Combined convexity = 50% asymmetry + 50% value-growth."""
    asym, _ = calc_asymmetry_score(row, weights=weights)
    stool, _ = calc_value_growth_score(row)
    return round((asym + stool) / 2, 1)

def calc_setup_stage(row):
    """Classify ticker into a swing-trading setup stage."""
    rsi   = _safe(row.get("RSI"), 50)
    pos52 = _safe(row.get("Pos52"), 50)
    vs50  = _safe(row.get("vsMA50"), 0)
    vs200 = _safe(row.get("vsMA200"), 0)
    # Breaking Down — below support, thesis at risk
    if rsi < 35 and vs50 < -10:
        return "Breaking Down"
    # Extended — far from support, risky entry
    if rsi > 70 or pos52 > 90 or vs50 > 25:
        return "Extended"
    # Basing — consolidating near support
    if 35 <= rsi <= 50 and pos52 < 40 and abs(vs50) <= 5:
        return "Basing"
    # Emerging — starting to move
    if 50 <= rsi <= 60 and 30 <= pos52 <= 60 and 0 < vs50 < 10:
        return "Emerging"
    # Trending — established uptrend
    if 55 <= rsi <= 70 and pos52 > 50 and vs50 > 5:
        return "Trending"
    return "Neutral"


def calc_four_pillars(row, themes, spy_ret=None, etf_data=None, st_data=None):
    """Score each of the 4 swing-trading pillars 0-100 and return alignment."""
    # ── Technical (0-100) ──
    tech = 0.0
    rsi = _safe(row.get("RSI"), 50)
    # RSI sweet spot 40-65: up to 30 pts, peak at 50
    if 40 <= rsi <= 65:
        tech += 30 * (1 - abs(rsi - 52.5) / 12.5)
    elif 30 <= rsi < 40:
        tech += 30 * (rsi - 30) / 10 * 0.5
    # Price above MA50: 25 pts proportional
    vs50 = _safe(row.get("vsMA50"), 0)
    if vs50 > 0:
        tech += min(vs50 / 20, 1.0) * 25
    # 52wk position 30-70%: 25 pts, peak at 50%
    pos52 = _safe(row.get("Pos52"), 50)
    if 30 <= pos52 <= 70:
        tech += 25 * (1 - abs(pos52 - 50) / 20)
    elif pos52 > 70:
        tech += 25 * max(0, 1 - (pos52 - 70) / 30) * 0.5
    # Trend consistency: vsMA50 > 0 AND vsMA200 > 0: 20 pts
    vs200 = _safe(row.get("vsMA200"), 0)
    if vs50 > 0 and vs200 > 0:
        tech += 20
    elif vs50 > 0:
        tech += 10

    # ── Fundamental (0-100) ──
    # Designed for small-cap/micro-cap growth companies where FCF is often negative
    # and Rule of 40 is rarely achieved. A decent growth stock should score ~45-55.
    fund = 0.0

    # Baseline: 10 pts — the company exists and has financials
    fund += 10

    # Revenue growth: up to 25 pts (the key metric for growth-stage companies)
    rev_g = _safe(row.get("RevGrowthPct"), 0)
    if rev_g > 30:
        fund += 25
    elif rev_g > 15:
        fund += 15 + (rev_g - 15) / 15 * 10
    elif rev_g > 0:
        fund += rev_g / 15 * 15
    else:
        fund += 3  # negative growth isn't great but isn't the only metric

    # Gross margin > 40%: up to 12 pts — shows business quality
    gm = _safe(row.get("GrossMargin"), 0)
    if gm > 60:
        fund += 12
    elif gm > 40:
        fund += 6 + (gm - 40) / 20 * 6
    elif gm > 20:
        fund += (gm - 20) / 20 * 6
    else:
        fund += 2  # hardware/manufacturing companies have low margins, not penalized harshly

    # FCF: 15 pts positive, 5 pts baseline if negative (most growth companies are pre-profit)
    if row.get("FCFPositive"):
        fund += 15
    else:
        fund += 5  # pre-profit growth is normal, not a failure

    # Analyst upside > 0: up to 10 pts (reduced — analysts often wrong on growth stocks)
    upside = _safe(row.get("AnalystUpside"), 0)
    num_analysts = int(_safe(row.get("NumAnalysts"), 0))
    analyst_conf = 0.0 if num_analysts == 0 else min(1.0, 0.4 + 0.2 * num_analysts)
    if upside > 0:
        fund += min(upside / 80, 1.0) * 10 * analyst_conf
    elif num_analysts == 0:
        fund += 3  # uncovered — not penalized

    # Rule of 40: up to 12 pts (bonus for companies that achieve it)
    rule40 = _safe(row.get("Rule40"), 0)
    if rule40 > 40:
        fund += 12
    elif rule40 > 20:
        fund += (rule40 - 20) / 20 * 8
    elif rule40 > 0:
        fund += 3

    # P/S valuation positioning: up to 8 pts — trading below historical avg is attractive
    ps_pos = _safe(row.get("PS_HistPos"))
    if ps_pos is not None:
        if ps_pos <= 25:
            fund += 8  # historically cheap
        elif ps_pos <= 50:
            fund += 5  # below average
        elif ps_pos <= 75:
            fund += 2  # above average
    else:
        fund += 3  # no data, neutral

    # ── Thematic (0-100) ──
    thematic = 0.0
    ticker = row.get("Ticker", "")
    # Find ALL themes this ticker belongs to (a ticker may appear in multiple themes)
    ticker_themes = get_ticker_themes(themes, ticker) if isinstance(themes, dict) and "themes" in themes else []
    ticker_theme = ticker_themes[0] if ticker_themes else "Other"
    # Belongs to at least one theme (not "Other"): 30 pts
    if ticker_themes:
        thematic += 30
    # ETF benchmark momentum boost: use the BEST (highest) ETF momentum score across all themes
    etf_boost = 0
    if etf_data is not None and not etf_data.empty and spy_ret and ticker_themes:
        theme_etfs_map = get_theme_etfs(themes) if isinstance(themes, dict) and "themes" in themes else {}
        spy_3m = spy_ret.get("3m", 0)
        best_etf_boost = 0
        for th in ticker_themes:
            theme_etf_list = theme_etfs_map.get(th, [])
            etf_sub = etf_data[etf_data["Ticker"].isin(theme_etf_list)]
            if not etf_sub.empty and etf_sub["Ret3m"].notna().any():
                etf_avg_3m = etf_sub["Ret3m"].dropna().mean()
                etf_rs = etf_avg_3m - spy_3m
                if etf_rs > 0:
                    candidate = min(etf_rs / 30, 1.0) * 25
                    if candidate > best_etf_boost:
                        best_etf_boost = candidate
        etf_boost = best_etf_boost
        thematic += etf_boost
    # Theme's own ticker 3m return vs SPY: up to 15 pts
    if spy_ret:
        ret3m = _safe(row.get("Ret3m"), 0)
        spy_3m = spy_ret.get("3m", 0) if spy_ret else 0
        theme_rs = ret3m - spy_3m
        if theme_rs > 0:
            thematic += min(theme_rs / 30, 1.0) * 15
    # Theme breadth: multiple tickers trending — use Pos52 as proxy
    # (scored at DataFrame level, add a base 15 if the ticker itself is trending)
    if _safe(row.get("Pos52"), 0) > 50 and _safe(row.get("vsMA50"), 0) > 0:
        thematic += 15

    # ── Narrative (0-100) ──
    # Captures the STORY: sentiment, attention, insider conviction, theme heat,
    # catalyst proximity, and track record. Distinct from Technical/Fundamental/Thematic.
    # Designed so a stock with no exceptional signals still scores ~40-50 (neutral narrative).
    narr = 0.0
    ticker = row.get("Ticker", "")

    # Baseline: 15 pts — you picked this stock for a reason, it has a thesis
    narr += 15

    # Social sentiment (StockTwits): up to 15 pts
    _st_sentiment = (st_data or {}).get(ticker, {})
    bull_pct = _st_sentiment.get("bull_pct")
    msg_count = _st_sentiment.get("msg_count", 0)
    if bull_pct is not None:
        if bull_pct >= 60:
            narr += 8 + min((bull_pct - 60) / 30, 1.0) * 7
        elif bull_pct < 40:
            narr += 5  # contrarian — extreme bearishness can mean opportunity
        else:
            narr += 7  # neutral sentiment, not a red flag
        if msg_count >= 10:
            narr += min(msg_count / 25, 1.0) * 3
    else:
        narr += 7  # no data = neutral, not penalized

    # Insider buying: 12 pts — strongest conviction signal
    insider_sig = str(row.get("InsiderSignal", "")).lower()
    insider_net = _safe(row.get("InsiderNet"), 0)
    if insider_sig == "buying" or insider_net > 0:
        narr += 12
    else:
        narr += 3  # absence of insider buying isn't bearish — most stocks don't have it

    # Insider ownership: up to 10 pts — skin in the game
    # Most small-caps have some insider ownership, scale generously
    insider_pct = _safe(row.get("InsiderPct"), 0)
    if insider_pct > 10:
        narr += 10
    elif insider_pct > 0:
        narr += max(3, min(insider_pct / 10 * 8, 10))
    else:
        narr += 2

    # Earnings beat track record: up to 10 pts
    beats = row.get("EarningsBeats")
    if beats is not None:
        try:
            beat_val = int(beats) if not isinstance(beats, str) else int(beats.split("/")[0])
            narr += min(beat_val / 3, 1.0) * 10  # 3 beats = full marks (was 4)
        except (ValueError, IndexError):
            narr += 5  # have data but can't parse
    else:
        narr += 5  # no data = neutral

    # Theme momentum: up to 15 pts — is the theme hot right now?
    ticker_themes = get_ticker_themes(themes, ticker)
    if ticker_themes and etf_data is not None and not etf_data.empty:
        theme_etf_rets = []
        _themes_inner = themes.get("themes", {}) if isinstance(themes, dict) else {}
        for th_name in ticker_themes:
            th_entry = _themes_inner.get(th_name, {})
            for etf_t in th_entry.get("etfs", []):
                etf_row = etf_data[etf_data["Ticker"] == etf_t]
                if not etf_row.empty:
                    r1m = etf_row.iloc[0].get("Ret1m")
                    if r1m is not None:
                        theme_etf_rets.append(r1m)
        if theme_etf_rets:
            avg_theme_ret = np.mean(theme_etf_rets)
            if avg_theme_ret > 5:
                narr += 15
            elif avg_theme_ret > 0:
                narr += 8 + min(avg_theme_ret / 5, 1.0) * 7
            elif avg_theme_ret > -5:
                narr += 6  # flat theme, not a penalty
            else:
                narr += 3  # theme is cold
        else:
            narr += 7  # no ETF data, neutral
    else:
        narr += 7  # untagged, neutral

    # Analyst coverage breadth: up to 10 pts
    num_analysts = int(_safe(row.get("NumAnalysts"), 0))
    if num_analysts >= 5:
        narr += 10
    elif num_analysts >= 3:
        narr += 8
    elif num_analysts >= 1:
        narr += 6
    else:
        narr += 4  # uncovered — could be undiscovered, not penalized heavily

    # Earnings catalyst proximity: up to 5 pts (bonus)
    next_earn = row.get("NextEarnings")
    if next_earn and str(next_earn) != "N/A":
        try:
            earn_dt = pd.to_datetime(next_earn)
            days_to = (earn_dt - pd.Timestamp.now()).days
            if 0 <= days_to <= 14:
                narr += 5
            elif 0 <= days_to <= 30:
                narr += 3
        except Exception:
            pass

    tech = min(round(tech, 1), 100)
    fund = min(round(fund, 1), 100)
    thematic = min(round(thematic, 1), 100)
    narr = min(round(narr, 1), 100)
    aligned = tech >= 50 and fund >= 50 and thematic >= 50 and narr >= 50

    return {
        "technical": tech, "fundamental": fund,
        "thematic": thematic, "narrative": narr,
        "aligned": aligned,
    }


def claude_summarize(ticker, price, rsi, atr_pct, pos52, vs50, vs200,
                     short_pct, beta, upside, recommendation, mcap,
                     rev_growth, insider_signal, earnings_beats,
                     next_earnings, st_bull_pct, st_msgs, headlines, api_key):
    """Synthesize all available data into a narrative using Claude API."""
    news_block = "\n".join(
        f"  - {h.get('date','')} {h.get('title','')} ({h.get('pub','')})"
        for h in (headlines or [])[:5]
    ) or "  None available"

    prompt = f"""You are a concise investment analyst. Synthesize the data below for ${ticker} into a clear investment summary for someone who favours asymmetric, high-beta re-rating opportunities.

TECHNICAL:
- Price: ${price} | RSI: {rsi} | 52wk position: {pos52:.0f}% | ATR%: {atr_pct}%
- vs MA50: {vs50:+.1f}% | vs MA200: {(str(vs200) + '%') if vs200 else 'N/A'}

FUNDAMENTAL:
- Market cap: {mcap} | Beta: {beta or 'N/A'} | Short interest: {short_pct or 'N/A'}%
- Revenue growth: {rev_growth or 'N/A'}% | Analyst consensus: {recommendation or 'N/A'}
- Analyst upside to mean target: {upside or 'N/A'}% | Next earnings: {next_earnings or 'N/A'}
- EPS beats last 4 quarters: {earnings_beats or 'N/A'} | Insider activity (90d): {insider_signal or 'N/A'}

SOCIAL SENTIMENT (StockTwits):
- Bull sentiment: {str(st_bull_pct) + '%' if st_bull_pct else 'N/A'} of {st_msgs} recent messages

RECENT NEWS:
{news_block}

Respond in exactly this format:

**SIGNAL** (Bullish / Bearish / Mixed / Neutral): [1 sentence overall read]

**THESIS**: [2-3 sentences. What is the re-rating case? What needs to happen for this to move?]

**CATALYSTS**:
- [specific catalyst 1]
- [specific catalyst 2]
- [specific catalyst 3 if relevant]

**KEY RISK**: [The main thing that kills the thesis]

**VERDICT**: [One of: Strong Setup / Developing Setup / Watch List / Avoid — with a one-line reason]

Be direct. No filler. Base everything on the data provided."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
        return f"API error {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return f"Request failed: {e}"

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Portfolio")
    grad_divider()

    # ── Watchlist selector ──
    wl_names = list(st.session_state.watchlists.keys())
    active_idx = wl_names.index(st.session_state.active_watchlist) if st.session_state.active_watchlist in wl_names else 0
    chosen_wl = st.selectbox("Watchlist", wl_names, index=active_idx, key="wl_select")
    if chosen_wl != st.session_state.active_watchlist:
        st.session_state.active_watchlist = chosen_wl
        st.session_state.tickers = st.session_state.watchlists[chosen_wl]
        st.cache_data.clear()
        st.rerun()

    wl_c1, wl_c2 = st.columns(2)
    with wl_c1:
        if st.button("Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with wl_c2:
        st.caption(f"{len(st.session_state.tickers)} tickers")
    st.caption(f"Cache TTL — Price: 5 min | Fund: 30 min | Sentiment: 15 min")
    st.caption(f"Last load: {datetime.now().strftime('%H:%M:%S')}")
    grad_divider()

    # ── Watchlist management ──
    with st.expander("Manage Watchlists"):
        # Create new watchlist
        new_wl_name = st.text_input("New watchlist name", placeholder="e.g. Holdings", key="new_wl")
        if st.button("Create Watchlist", use_container_width=True, key="btn_create_wl"):
            if new_wl_name and new_wl_name not in st.session_state.watchlists:
                st.session_state.watchlists[new_wl_name] = []
                save_watchlists(st.session_state.watchlists)
                st.session_state.active_watchlist = new_wl_name
                st.session_state.tickers = []
                st.cache_data.clear()
                st.rerun()

        grad_divider()

        # Import Yahoo Finance CSV
        st.markdown("**Import from Yahoo Finance**")
        st.caption("Export your watchlist from Yahoo Finance (CSV), then upload here.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload", label_visibility="collapsed")
        import_target = st.selectbox("Import into", wl_names, key="import_target")
        if st.button("Import", use_container_width=True, key="btn_import") and uploaded:
            imported = parse_yahoo_csv(uploaded)
            if imported:
                existing = st.session_state.watchlists.get(import_target, [])
                merged = list(dict.fromkeys(existing + imported))  # dedupe, preserve order
                st.session_state.watchlists[import_target] = merged
                save_watchlists(st.session_state.watchlists)
                if import_target == st.session_state.active_watchlist:
                    st.session_state.tickers = merged
                st.cache_data.clear()
                st.success(f"Imported {len(imported)} tickers into '{import_target}'")
                st.rerun()
            else:
                st.error("No tickers found in CSV. Expected a 'Symbol' column.")

        grad_divider()

        # Rename watchlist
        rename_val = st.text_input("Rename current watchlist", value=st.session_state.active_watchlist, key="rename_wl")
        if st.button("Rename", use_container_width=True, key="btn_rename_wl"):
            if rename_val and rename_val != st.session_state.active_watchlist and rename_val not in st.session_state.watchlists:
                tickers_copy = st.session_state.watchlists.pop(st.session_state.active_watchlist)
                st.session_state.watchlists[rename_val] = tickers_copy
                st.session_state.active_watchlist = rename_val
                st.session_state.tickers = tickers_copy
                save_watchlists(st.session_state.watchlists)
                st.rerun()

        grad_divider()

        # Duplicate watchlist
        dup_name = st.text_input("Duplicate as", placeholder="e.g. Watchlist Copy", key="dup_wl")
        if st.button("Duplicate Current", use_container_width=True, key="btn_dup_wl"):
            if dup_name and dup_name not in st.session_state.watchlists:
                st.session_state.watchlists[dup_name] = st.session_state.tickers.copy()
                save_watchlists(st.session_state.watchlists)
                st.rerun()

        # Delete watchlist (only if more than one)
        if len(wl_names) > 1:
            del_wl = st.selectbox("Delete watchlist", ["--"] + [w for w in wl_names if w != st.session_state.active_watchlist], key="del_wl")
            if st.button("Delete", use_container_width=True, key="btn_del_wl"):
                if del_wl != "--" and del_wl in st.session_state.watchlists:
                    del st.session_state.watchlists[del_wl]
                    save_watchlists(st.session_state.watchlists)
                    st.rerun()

    grad_divider()

    # ── Ticker management (within active watchlist) ──
    st.subheader("Manage Tickers")
    new_t = st.text_input("Add ticker", placeholder="e.g. NVDA").upper().strip()
    if st.button("Add", use_container_width=True):
        if new_t and new_t not in st.session_state.tickers:
            st.session_state.tickers.append(new_t)
            st.session_state.watchlists[st.session_state.active_watchlist] = st.session_state.tickers
            save_watchlists(st.session_state.watchlists)
            st.cache_data.clear()
            st.rerun()
        elif new_t in st.session_state.tickers:
            st.warning(f"{new_t} already in list.")
    remove_t = st.selectbox("Remove ticker", ["--"] + sorted(st.session_state.tickers))
    if st.button("Remove", use_container_width=True):
        if remove_t != "--":
            st.session_state.tickers.remove(remove_t)
            st.session_state.watchlists[st.session_state.active_watchlist] = st.session_state.tickers
            save_watchlists(st.session_state.watchlists)
            st.cache_data.clear()
            st.rerun()
    grad_divider()
    selected = st.multiselect("Filter view", st.session_state.tickers,
                               default=st.session_state.tickers)
    grad_divider()
    # Single-ticker deep-dive selector
    st.subheader("Ticker Deep Dive")
    deep_dive_ticker = st.selectbox("Select ticker", ["--"] + sorted(st.session_state.tickers), key="deep_dive")
    grad_divider()

    # Ideas & Roadmap (moved from tab)
    with st.expander("Ideas & Roadmap"):
        st.markdown("**From the S&J Framework — high priority, free to build**")
        st.markdown("""
| Feature | Status |
|---|---|
| ~~Sector / theme tagging~~ | DONE |
| Peer comparison table | Planned |
| ~~Custom score weight sliders~~ | DONE |
| ~~Priced for perfection flag~~ | DONE |
| EBITDA turn detector | Planned |
| Google Trends (pytrends) | Planned |
| ~~Relative strength vs SPY~~ | DONE |
| ~~Earnings calendar view~~ | DONE |
| Volume spike detector | Planned |
| ~~Export to CSV~~ | DONE |
        """)
        st.markdown("**Performance review**")
        st.markdown("""
| Area | Notes |
|---|---|
| StockTwits fetch | Now in main data load for Narrative pillar — adds ~10-15s. Review: batch API, async, or move to background refresh |
| Sector ETF fetch | 11 individual downloads for Market Conditions. Consider bulk download or pre-market cron |
| Scanner scoring | Caps at 80 tickers but still heavy. Consider caching scored results |
        """)
        st.markdown("**Needs API / data source**")
        st.markdown("""
| Feature | Source | Notes |
|---|---|---|
| **Options & dark pool flow** | [Unusual Whales API](https://unusualwhales.com/api) | Paid API — options flow, dark pool prints, whale alerts, Congress trades. MCP available? TBC |
| **X / social sentiment** | [xAI Grok API](https://x.ai/api) | X post sentiment per ticker. Grok API is live (free tier available). MCP available? TBC |
| **Options chain analysis** | TBD — research needed | Implied vol surface, put/call skew, gamma exposure. Candidates: Tradier, Polygon.io, Tastytrade API |
| ETF ownership & flows | ETF.com / Bloomberg | Which ETFs hold each ticker; inflow/outflow signal |
| SEC 8-K filing alerts | EDGAR RSS | Catalyst detection, dilution risk — free via EDGAR RSS, no API key needed |
| Forward P/S & EV/Rev | Refinitiv / Koyfin | Trailing multiples overstate valuation for fast-growers |
| Real-time short interest | S3 Analytics / Ortex | yfinance short data is ~2 weeks stale (FINRA bi-monthly) |
| Institutional-grade beta | Bloomberg / Refinitiv | yfinance uses 5yr monthly — unreliable for small caps & recent IPOs |
        """)
        st.markdown("**MCP servers to explore**")
        st.markdown("""
| MCP | What it unlocks |
|---|---|
| Unusual Whales MCP | Options flow + dark pool data directly in Claude |
| Browser/Firecrawl MCP | Scrape X, Stocktwits, Reddit for sentiment |
| Polygon.io MCP | Real-time options chain, news, financials |
| EDGAR MCP | SEC filings, 8-K catalyst alerts |
        """)

    grad_divider()
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# ── Load all data once ───────────────────────────────────────────────────────

with st.status("◣ Loading Convexity Terminal...", expanded=True) as _status:
    _status.update(label="◣ Scanning markets...", state="running")
    st.write("⚡ Fetching price data...")
    df_price = fetch_price_data(st.session_state.tickers)

    if df_price.empty:
        _status.update(label="◣ Failed", state="error")
        st.error("No data. Check internet connection.")
        st.stop()

    st.write("📊 Loading fundamentals & valuation...")
    df_fund = fetch_fundamentals(st.session_state.tickers)

    spy_ret = fetch_spy_returns()
    spy_close, spy_daily_ret = fetch_spy_daily()

    st.write("🌐 Scanning market environment...")
    market_env = fetch_market_environment()
    env_total, env_pillars, env_decision, env_decision_sub = calc_market_env_score(market_env)

    st.write("💪 Calculating relative strength...")
    df_rs = calc_downday_rs(tuple(st.session_state.tickers), spy_close, spy_daily_ret)

    st.write("📡 Fetching theme benchmarks...")
    _all_etf_tickers = get_all_etf_tickers(st.session_state.themes)
    df_etf = fetch_etf_benchmark_data(tuple(_all_etf_tickers)) if _all_etf_tickers else pd.DataFrame()

    df_price = df_price[df_price["Ticker"].isin(selected)].copy()

    st.write("🧮 Computing scores...")
    df_all = df_price.merge(df_fund, on="Ticker", how="left")
    _asym_weights = st.session_state.get("asym_weights")
    df_all["ConvexityScore"] = df_all.apply(lambda r: calc_convexity_score(r, weights=_asym_weights), axis=1)
    df_all["MomentumScore"]  = df_all.apply(calc_momentum_score, axis=1)
    df_all["AsymmetryScore"]  = df_all.apply(lambda r: calc_asymmetry_score(r, weights=_asym_weights)[0], axis=1)
    df_all["VGScore"]      = df_all.apply(lambda r: calc_value_growth_score(r)[0], axis=1)

    st.write("🔍 Loading news & insider signals...")
    df_extras = fetch_extras(tuple(st.session_state.tickers))
    df_all = df_all.merge(
        df_extras[["Ticker", "InsiderSignal", "InsiderNet", "InsiderBuys", "EarningsBeats"]],
        on="Ticker", how="left"
    )

    st.write("💬 Loading sentiment...")
    st_data = fetch_stocktwits(tuple(st.session_state.tickers))

    _status.update(label="◣ Terminal Ready", state="complete", expanded=False)
# Merge sentiment into df_all
df_all["ST_BullPct"] = df_all["Ticker"].map(lambda t: st_data.get(t, {}).get("bull_pct"))
df_all["ST_MsgCount"] = df_all["Ticker"].map(lambda t: st_data.get(t, {}).get("msg_count", 0))

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

# Setup Stage and Four-Pillar scores
df_all["SetupStage"] = df_all.apply(calc_setup_stage, axis=1)
_pillar_results = df_all.apply(
    lambda r: calc_four_pillars(r, st.session_state.themes, spy_ret, df_etf, st_data), axis=1
)
df_all["PillarTech"]    = _pillar_results.apply(lambda d: d["technical"])
df_all["PillarFund"]    = _pillar_results.apply(lambda d: d["fundamental"])
df_all["PillarTheme"]   = _pillar_results.apply(lambda d: d["thematic"])
df_all["PillarNarr"]    = _pillar_results.apply(lambda d: d["narrative"])
df_all["PillarAligned"] = _pillar_results.apply(lambda d: d["aligned"])

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
    "PostMktChg": None, "PreMktChg": None,
    "TargetLow": None, "TargetHigh": None, "TargetMean": None,
    "RelVol": None,
    # Extras
    "NextEarnings": None, "InsiderSignal": None, "InsiderNet": 0,
    "InsiderBuys": 0, "EarningsBeats": None, "Headlines": None,
    # Sentiment
    "ST_BullPct": None, "ST_MsgCount": 0,
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
st.caption(f"As of {_data_ts}  |  **{st.session_state.active_watchlist}**  |  {len(df_price)} tickers")
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

    # Score breakdown
    _, asym_comps = calc_asymmetry_score(t_row, weights=_asym_weights)
    _, stool_comps = calc_value_growth_score(t_row)
    with st.expander("Score Breakdown"):
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**Asymmetry Components**")
            for k, v in asym_comps.items():
                st.markdown(f"- {k}: **{v:.1f}** pts")
        with bc2:
            st.markdown("**Value-Growth Score**")
            for k, v in stool_comps.items():
                st.markdown(f"- {k}: **{v:.1f}** pts")

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
    except Exception:
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
tab_dash, tab_conv, tab_themes, tab_charts, tab_sig, tab_scan = st.tabs([
    "Dashboard", "Convexity", "Themes", "Charts", "Signals", "Scanner",
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

    # Expandable detail panels
    with st.expander("Environment Detail"):
        det_c1, det_c2, det_c3 = st.columns(3)

        with det_c1:
            st.markdown("**Volatility**")
            vix_trend = "Rising" if market_env.get("vix_rising") else "Falling"
            vix_trend_color = "red" if market_env.get("vix_rising") else "green"
            st.markdown(f"- VIX: **{market_env.get('vix_level', 'N/A')}** "
                        f"(vs 20d MA: {market_env.get('vix_ma20', 'N/A')})")
            st.markdown(f"- VIX Trend: **:{vix_trend_color}[{vix_trend}]**")
            st.markdown(f"- VIX 1Y Percentile: **{market_env.get('vix_pct_rank', 'N/A'):.0f}th**")

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

        with det_c3:
            # Sector performance mini-table
            st.markdown("**Sector Performance (5d)**")
            sectors = market_env.get("sectors", {})
            if sectors:
                sorted_sectors = sorted(sectors.items(), key=lambda x: x[1]["ret_5d"], reverse=True)
                for etf, data in sorted_sectors:
                    r5 = data["ret_5d"]
                    sc = "green" if r5 > 0 else "red"
                    st.markdown(f"- {etf}: **:{sc}[{r5:+.1f}%]**")

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

    overview_cols = ["Ticker", "Price", "PostMktChg", "Spark30", "SetupStage", "RS_Label",
                     "PillarAligned", "ConvexityScore", "MomentumScore", "RSI", "Pos52",
                     "ST_BullPct", "AnalystUpside", "RevGrowthPct", "PS_Current", "ShortPct", "Theme"]
    # Ensure required columns exist
    for _oc in ["RS_Label", "Spark30", "PostMktChg", "ST_BullPct"]:
        if _oc not in overview_df.columns:
            overview_df[_oc] = None
    overview_disp = overview_df[[c for c in overview_cols if c in overview_df.columns]].copy()
    overview_disp["PillarAligned"] = overview_disp["PillarAligned"].apply(lambda x: "Yes" if x else "No")
    overview_disp = overview_disp.rename(columns={
        "PostMktChg": "AH %", "Spark30": "30d", "SetupStage": "Stage", "RS_Label": "RS",
        "PillarAligned": "Aligned", "ConvexityScore": "Convexity", "MomentumScore": "Momentum",
        "Pos52": "52wk Pos", "ST_BullPct": "Sentiment",
        "AnalystUpside": "Upside %",
        "RevGrowthPct": "Rev Growth %", "PS_Current": "P/S", "ShortPct": "Short %",
    })
    overview_disp = overview_disp.sort_values("Convexity", ascending=False).reset_index(drop=True)

    _overview_col_config = {
        "30d": st.column_config.LineChartColumn("30d Trend", width="small"),
        "Convexity": st.column_config.ProgressColumn("Convexity", min_value=0, max_value=100, format="%d"),
        "Momentum": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%d"),
        "AH %": st.column_config.NumberColumn("AH %", format="%.2f%%"),
    }
    st.dataframe(
        overview_disp,
        column_config=_overview_col_config,
        use_container_width=True, hide_index=True,
        height=min(800, 38 * len(overview_disp) + 40),
    )
    st.download_button("Export Overview CSV",
                       overview_disp.drop(columns=["30d"], errors="ignore").to_csv(index=False),
                       "portfolio_overview.csv", "text/csv", key="dl_overview")

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
        pillar_disp = pillar_df[["Ticker", "Price", "SetupStage", "PillarTech", "PillarFund",
                                  "PillarTheme", "PillarNarr", "PillarAligned"]].copy()
        pillar_disp = pillar_disp.rename(columns={
            "SetupStage": "Stage", "PillarTech": "Technical", "PillarFund": "Fundamental",
            "PillarTheme": "Thematic", "PillarNarr": "Narrative", "PillarAligned": "Aligned",
        }).reset_index(drop=True)
        pillar_disp["Aligned"] = pillar_disp["Aligned"].apply(lambda x: "Yes" if x else "No")
        pillar_styled = (pillar_disp.style
            .background_gradient(subset=["Technical", "Fundamental", "Thematic", "Narrative"],
                                 cmap="RdYlGn", vmin=0, vmax=100)
            .map(_color_stage, subset=["Stage"])
            .map(lambda v: _color_aligned(v == "Yes"), subset=["Aligned"])
            .format({"Price": "${:.2f}", "Technical": "{:.0f}", "Fundamental": "{:.0f}",
                      "Thematic": "{:.0f}", "Narrative": "{:.0f}"}, na_rep="N/A")
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

    # Value-Growth Score ranking
    st.markdown("#### Value-Growth Score")
    st.caption("Green = all three legs present. Hover for component breakdown.")

    stool_sorted = df_conv.sort_values("VGScore", ascending=False)
    stool_comps  = stool_sorted.apply(lambda r: calc_value_growth_score(r)[1], axis=1)

    comp_names = ["Undervalued (P/S)", "Revenue Growth", "Sector Momentum"]
    comp_colors = ["#9b59b6", "#2ecc71", "#3498db"]
    fig_stool = go.Figure()
    for name, color in zip(comp_names, comp_colors):
        fig_stool.add_trace(go.Bar(
            name=name,
            x=stool_sorted["Ticker"],
            y=[c.get(name, 0) for c in stool_comps],
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}} pts<extra></extra>",
        ))
    fig_stool.update_layout(
        barmode="stack", height=380,
        xaxis_title="", yaxis_title="Score (0-100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=50, b=40), **DARK,
    )
    fig_stool.add_hline(y=60, line_dash="dash", line_color="#f39c12",
                        annotation_text="Strong setup", annotation_font_color="#f39c12")
    st.plotly_chart(fig_stool, use_container_width=True)

    grad_divider()

    # Asymmetry score breakdown
    st.markdown("#### Asymmetry Score Breakdown")
    _w = st.session_state.get("asym_weights", {})
    st.caption(f"Catalyst fuel: analyst upside ({_w.get('Analyst Upside', 30)}) + beta ({_w.get('Beta', 20)}) + volatility ({_w.get('Volatility (ATR%)', 15)}) + short interest ({_w.get('Short Interest', 15)}) + RSI positioning ({_w.get('RSI Positioning', 10)}) + consensus ({_w.get('Analyst Consensus', 10)}) — adjust in sidebar")

    asym_sorted = df_conv.sort_values("AsymmetryScore", ascending=False)
    asym_comps_list = asym_sorted.apply(lambda r: calc_asymmetry_score(r, weights=_asym_weights)[1], axis=1)

    comp_cols_a   = ["Analyst Upside", "Beta", "Volatility (ATR%)", "Short Interest", "RSI Positioning", "Analyst Consensus"]
    colors_comp_a = ["#9b59b6","#3498db","#e67e22","#e74c3c","#2ecc71","#f1c40f"]
    fig_asym = go.Figure()
    for col, color in zip(comp_cols_a, colors_comp_a):
        fig_asym.add_trace(go.Bar(
            name=col, x=asym_sorted["Ticker"],
            y=[c.get(col, 0) for c in asym_comps_list],
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.1f}} pts<extra></extra>",
        ))
    fig_asym.update_layout(
        barmode="stack", height=380,
        xaxis_title="", yaxis_title="Score (0-100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40,r=20,t=50,b=40), **DARK
    )
    st.plotly_chart(fig_asym, use_container_width=True)

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
                               margin=dict(l=60,r=20,t=20,b=40), showlegend=False, **DARK)
        st.plotly_chart(fig_conv, use_container_width=True)
        conv_tbl = conv[["Ticker","Price","ConvexityScore","AsymmetryScore","VGScore","AnalystUpside","NextEarnings"]].copy()
        conv_tbl["NextEarnings"] = conv_tbl["NextEarnings"].fillna("N/A")
        conv_tbl = conv_tbl.rename(columns={
            "ConvexityScore":"Convexity","AsymmetryScore":"Asym",
            "VGScore":"V-G Score","AnalystUpside":"Upside","NextEarnings":"Next Earnings",
        })
        st.dataframe(conv_tbl, width="stretch", hide_index=False,
                     column_config={
                         "Convexity": st.column_config.ProgressColumn("Convexity", min_value=0, max_value=100, format="%d"),
                         "Asym": st.column_config.ProgressColumn("Asym", min_value=0, max_value=100, format="%d"),
                         "V-G Score": st.column_config.ProgressColumn("V-G Score", min_value=0, max_value=100, format="%d"),
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

    # ── (b) Theme Momentum Trend (ETF-based) ──
    grad_divider()
    st.markdown("#### Theme Momentum Trend")
    st.caption(
        "ETF benchmark relative strength vs SPY. Compares 1-month vs 3-month to detect acceleration/deceleration. "
        "'Your Picks vs ETF' shows if your selections outperform the sector benchmark."
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
            if rs_1m > rs_3m:
                momentum_status = "Accelerating"
            elif rs_1m < rs_3m:
                momentum_status = "Decelerating"
            else:
                momentum_status = "Stable"
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
                "ETF 1m vs SPY": rs_1m,
                "ETF 3m vs SPY": rs_3m,
                "Trend": momentum_status,
                "Picks vs ETF": picks_vs_etf,
            })
        if trend_rows:
            df_trend = pd.DataFrame(trend_rows).sort_values("ETF 3m vs SPY", ascending=False)

            def _color_trend(val):
                if val == "Accelerating":
                    return "background-color: #1a3a2a; color: #2ecc71"
                if val == "Decelerating":
                    return "background-color: #5b2c2c; color: #ff6b6b"
                return ""

            def _color_vs_spy(val):
                if pd.isna(val): return ""
                if val >= 5: return "background-color: #1a3a2a; color: #2ecc71"
                if val <= -5: return "background-color: #5b2c2c; color: #ff6b6b"
                return ""

            def _color_picks(val):
                if pd.isna(val): return ""
                if val >= 3: return "background-color: #1a3a2a; color: #2ecc71"
                if val <= -3: return "background-color: #5b2c2c; color: #ff6b6b"
                return ""

            trend_style_cols = ["ETF 1m vs SPY", "ETF 3m vs SPY"]
            trend_styled = (df_trend.style
                .map(_color_trend, subset=["Trend"])
                .map(_color_vs_spy, subset=trend_style_cols)
                .map(_color_picks, subset=["Picks vs ETF"])
                .format({
                    "ETF 1m vs SPY": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                    "ETF 3m vs SPY": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                    "Picks vs ETF": lambda x: f"{x:+.1f}%" if pd.notna(x) else "--",
                }, na_rep="N/A")
            )
            st.dataframe(trend_styled, width="stretch", hide_index=True)
            st.caption(
                "Accelerating = 1m RS > 3m RS (capital flowing in)  |  "
                "Decelerating = 1m RS < 3m RS (capital flowing out)  |  "
                "Picks vs ETF = your tickers' 1m return minus the ETF's 1m return"
            )
        else:
            st.info("Not enough ETF return data to compute momentum trends.")
    else:
        st.info("SPY or ETF data not available for momentum trends.")

    # ── (c) Theme Scores (holdings only, with ETF context) ──
    grad_divider()
    st.markdown("#### Theme Scores (Your Holdings)")
    st.caption("Convexity and momentum scores for themes where you hold tickers. ETF 3m return shown for context.")

    theme_score_rows = []
    for theme_name in _all_theme_names:
        user_tks = _user_tickers_map.get(theme_name, [])
        active = [t for t in user_tks if t in df_theme["Ticker"].values]
        if not active:
            continue
        sub = df_theme[df_theme["Ticker"].isin(active)]
        avg_conv = round(sub["ConvexityScore"].mean(), 1)
        avg_mom  = round(sub["MomentumScore"].mean(), 1)
        # ETF 3m return for context
        etf_list = _etf_tickers_map.get(theme_name, [])
        etf_sub = df_etf[df_etf["Ticker"].isin(etf_list)] if not df_etf.empty else pd.DataFrame()
        etf_3m = round(etf_sub["Ret3m"].dropna().mean(), 1) if not etf_sub.empty and etf_sub["Ret3m"].notna().any() else None
        theme_score_rows.append({
            "Theme": theme_name,
            "Tickers": len(active),
            "Avg Convexity": avg_conv,
            "Avg Momentum": avg_mom,
            "ETF 3m Ret": etf_3m,
            "Members": ", ".join(active),
        })

    if theme_score_rows:
        df_scores = pd.DataFrame(theme_score_rows).sort_values("Avg Convexity", ascending=False)

        fig_th = go.Figure()
        fig_th.add_trace(go.Bar(
            name="Convexity", x=df_scores["Theme"], y=df_scores["Avg Convexity"],
            marker_color="#9b59b6",
            hovertemplate="<b>%{x}</b><br>Convexity: %{y:.1f}<extra></extra>",
        ))
        fig_th.add_trace(go.Bar(
            name="Momentum", x=df_scores["Theme"], y=df_scores["Avg Momentum"],
            marker_color="#27ae60",
            hovertemplate="<b>%{x}</b><br>Momentum: %{y:.1f}<extra></extra>",
        ))
        # Add ETF 3m return as line for context
        if df_scores["ETF 3m Ret"].notna().any():
            fig_th.add_trace(go.Scatter(
                name="ETF 3m Ret (%)", x=df_scores["Theme"], y=df_scores["ETF 3m Ret"],
                mode="markers+lines", marker=dict(size=8, color="#f39c12"),
                line=dict(dash="dot", color="#f39c12"),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>ETF 3m: %{y:+.1f}%<extra></extra>",
            ))
        fig_th.update_layout(
            barmode="group", height=400,
            yaxis_title="Avg Score",
            yaxis2=dict(title="ETF 3m Return (%)", overlaying="y", side="right",
                        showgrid=False, title_font=dict(color="#f39c12"),
                        tickfont=dict(color="#f39c12")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=60, t=50, b=40), **DARK,
        )
        fig_th.add_hline(y=50, line_dash="dot", line_color="#555")
        st.plotly_chart(fig_th, use_container_width=True)

    # ── (d) Theme Drilldown ──
    grad_divider()
    st.markdown("#### Theme Drilldown")
    for theme_name in _all_theme_names:
        user_tks = _user_tickers_map.get(theme_name, [])
        members = [t for t in user_tks if t in df_theme["Ticker"].values]
        if not members:
            continue
        sub = df_theme[df_theme["Ticker"].isin(members)].sort_values("ConvexityScore", ascending=False)
        etf_list = _etf_tickers_map.get(theme_name, [])
        etf_sub = df_etf[df_etf["Ticker"].isin(etf_list)] if not df_etf.empty else pd.DataFrame()

        # Build expander label
        avg_conv = round(sub["ConvexityScore"].mean(), 0)
        avg_mom = round(sub["MomentumScore"].mean(), 0)
        with st.expander(f"**{theme_name}** ({len(members)} tickers)  |  Convexity: {avg_conv:.0f}  |  Momentum: {avg_mom:.0f}  |  ETFs: {', '.join(etf_list)}"):
            # ETF benchmark performance at top
            if not etf_sub.empty:
                etf_disp = etf_sub[["Ticker", "Price", "Ret1m", "Ret3m", "RetYTD", "Ret1y"]].copy()
                etf_disp = etf_disp.rename(columns={
                    "Ret1m": "1m Ret", "Ret3m": "3m Ret", "RetYTD": "YTD Ret", "Ret1y": "1y Ret",
                })
                st.markdown("**ETF Benchmark:**")
                etf_styled = etf_disp.style.format({
                    "Price": "${:.2f}",
                    "1m Ret": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                    "3m Ret": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                    "YTD Ret": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                    "1y Ret": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                }, na_rep="N/A")
                st.dataframe(etf_styled, width="stretch", hide_index=True)
                st.markdown("**Your Picks:**")

            drill = sub[["Ticker", "Price", "RSI", "Pos52", "ConvexityScore", "MomentumScore", "Ret1m", "Ret3m"]].copy()
            drill = drill.rename(columns={
                "Pos52": "52wk Pos", "ConvexityScore": "Convexity",
                "MomentumScore": "Momentum", "Ret1m": "1m Ret", "Ret3m": "3m Ret",
            })
            dr_grad = [c for c in ["Convexity", "Momentum"] if drill[c].notna().any()]
            dr_styled = drill.style
            if dr_grad:
                dr_styled = dr_styled.background_gradient(subset=dr_grad, cmap="RdYlGn", vmin=20, vmax=80)
            dr_styled = dr_styled.format({
                "Price": "${:.2f}",
                "RSI": "{:.0f}",
                "52wk Pos": "{:.0f}%",
                "Convexity": "{:.0f}",
                "Momentum": "{:.0f}",
                "1m Ret": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
                "3m Ret": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
            }, na_rep="N/A")
            st.dataframe(dr_styled, width="stretch", hide_index=True)

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
        except Exception:
            st.caption(f"{chart_ticker}: chart unavailable")

    grad_divider()

    # Portfolio Map — RSI vs 52-Week Position (moved from Dashboard)
    section_header("Portfolio Map — RSI vs 52-Week Position", "#f39c12")
    st.caption("Bubble size = ATR volatility. Bottom-left = weak. Top-right = strong. Bottom-right = pullback opportunity.")
    fig5 = go.Figure()
    for _, row in df_price.iterrows():
        vs200_str = f"{row['vsMA200']:+.1f}%" if row["vsMA200"] is not None else "N/A"
        fig5.add_trace(go.Scatter(
            x=[row["Pos52"]], y=[row["RSI"]],
            mode="markers+text", text=[row["Ticker"]], textposition="top center",
            marker=dict(size=row["ATR_pct"]*4, color=pos_color(row["Pos52"]),
                        line=dict(width=1, color="#0d1117")),
            hovertemplate=(
                f"<b>{row['Ticker']}</b><br>Price: ${row['Price']}<br>"
                f"RSI: {row['RSI']}<br>52wk Pos: {row['Pos52']:.1f}%<br>"
                f"ATR%: {row['ATR_pct']}<br>vs MA50: {row['vsMA50']:+.1f}%<br>"
                f"vs MA200: {vs200_str}<extra></extra>"
            ),
            showlegend=False,
        ))
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

    # ── Relative Strength — Down-Day Analysis ──
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=60, r=20, t=50, b=40), **DARK,
        )
        st.plotly_chart(fig_rs, use_container_width=True)

        # ── Down-Day Win Rate scatter ──
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
                except Exception:
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
            "EarningsBeats": row.get("EarningsBeats"),
            "NextEarnings": row.get("NextEarnings"),
            "BullPct": sent.get("bull_pct"), "MsgCount": sent.get("msg_count", 0),
        })

    tbl = pd.DataFrame(sig_rows)
    tbl["Recommendation"] = tbl["Recommendation"].apply(fmt_rec)
    tbl["MarketCap"]      = tbl["MarketCap"].apply(fmt_mcap)
    tbl["NextEarnings"]   = tbl["NextEarnings"].fillna("N/A")
    tbl["EarningsBeats"]  = tbl["EarningsBeats"].fillna("N/A")
    tbl = tbl.rename(columns={
        "AnalystUpside":"Upside","TargetMean":"Mean Target",
        "NumAnalysts":"# Analysts","Recommendation":"Consensus",
        "RevenueGrowth":"Rev Growth","MarketCap":"Mkt Cap",
        "ATR_pct":"ATR%","RelVol":"Rel Vol","InsiderSignal":"Insider",
        "EarningsBeats":"EPS Beats","NextEarnings":"Next Earnings",
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

    api_key = st.session_state.get("claude_key", "")
    if not api_key:
        st.warning(
            "Add your Anthropic API key in the sidebar to use this feature.  \n"
            "Get one at **console.anthropic.com** — $5 free credit to start."
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
                earnings_beats=ex.get("EarningsBeats") if hasattr(ex, "get") else None,
                next_earnings=fd.get("NextEarnings"),
                st_bull_pct=sent.get("bull_pct"), st_msgs=sent.get("msg_count", 0),
                headlines=ex.get("Headlines") if hasattr(ex, "get") else [],
                api_key=api_key,
            )

        if st.button(f"Analyze {selected_ticker}", use_container_width=True):
            with st.spinner(f"Generating brief for ${selected_ticker}..."):
                result = build_summary(selected_ticker)
            st.markdown(result)
            st.caption(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · model: claude-haiku-4-5-20251001")

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
