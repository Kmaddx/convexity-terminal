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
    page_title="Portfolio Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_TICKERS = [
    "IREN", "CIFR", "RKLB", "ONDS", "NBIS", "CRCL", "ASTS", "HOOD",
    "AREC", "ASPI", "MSTR", "BITF", "BMNR", "LTRX", "OSS",
    "UAMY", "TSLA", "PLTR"
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

def get_theme_tickers(themes_data):
    """Extract {theme_name: [user_tickers]} from the new themes structure."""
    themes_dict = themes_data.get("themes", {})
    return {name: val.get("tickers", []) for name, val in themes_dict.items()}

def get_ticker_themes(themes_data, ticker):
    """Return a list of all theme names that contain this ticker (may be in multiple themes)."""
    themes_dict = themes_data.get("themes", {})
    return [name for name, val in themes_dict.items() if ticker in val.get("tickers", [])]

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
    font=dict(color="#e6edf3"),
    xaxis_gridcolor="#21262d", yaxis_gridcolor="#21262d",
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
            results.append(dict(
                Ticker=t, Price=round(price,2),
                RSI=round(rsi,1), vsMA50=round(vs50,1),
                vsMA200=round(vs200,1) if not np.isnan(vs200) else None,
                ATR_pct=round(atr_pct,1), Pos52=round(pos52,1),
                Low52=round(w52l,2), High52=round(w52h,2),
                RelVol=rel_vol, Breakout=breakout,
                Ret1m=ret_1m, Ret3m=ret_3m, Ret6m=ret_6m,
            ))
        except Exception:
            continue
    return pd.DataFrame(results)

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

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(tickers):
    """Single call to yf.Ticker().info per ticker — covers both analyst/fundamental and valuation data."""
    results = []
    for t in tickers:
        row = {"Ticker": t}
        try:
            obj  = yf.Ticker(t)
            info = obj.info

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
            pass
        results.append(row)
    return pd.DataFrame(results)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_extras(tickers):
    results = []
    for t in tickers:
        row = {"Ticker": t, "InsiderSignal": "N/A", "InsiderNet": 0,
               "EarningsBeats": None, "Headlines": []}
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
                                   if "transaction" in c.lower() or "text" in c.lower()), None)
                    if tx_col is not None:
                        tx = recent[tx_col].astype(str).str.lower()
                        buys  = tx.str.contains("buy|purchase|acquisition", na=False).sum()
                        sells = tx.str.contains("sell|sale|disposition", na=False).sum()
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

@st.cache_data(ttl=900, show_spinner=False)
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
            if history[t] and history[t][-1].get("date") == today:
                history[t][-1] = {
                    "date": today,
                    "convexity": round(float(_safe(row.get("ConvexityScore"))), 1),
                    "momentum": round(float(_safe(row.get("MomentumScore"))), 1),
                    "asymmetry": round(float(_safe(row.get("AsymmetryScore"))), 1),
                    "stool": round(float(_safe(row.get("StoolScore"))), 1),
                    "price": round(float(_safe(row.get("Price"))), 2),
                }
            else:
                history[t].append({
                    "date": today,
                    "convexity": round(float(_safe(row.get("ConvexityScore"))), 1),
                    "momentum": round(float(_safe(row.get("MomentumScore"))), 1),
                    "asymmetry": round(float(_safe(row.get("AsymmetryScore"))), 1),
                    "stool": round(float(_safe(row.get("StoolScore"))), 1),
                    "price": round(float(_safe(row.get("Price"))), 2),
                })
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
    w_upside    = weights.get("Analyst Upside", 30)
    w_beta      = weights.get("Beta", 20)
    w_atr       = weights.get("Volatility (ATR%)", 15)
    w_short     = weights.get("Short Interest", 15)
    w_rsi       = weights.get("RSI Positioning", 10)
    w_consensus = weights.get("Analyst Consensus", 10)
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

def calc_three_legged_stool(row):
    """
    Score 0-100. Implements the Three-Legged Stool:
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
    """Combined convexity = 50% asymmetry + 50% three-legged stool."""
    asym, _ = calc_asymmetry_score(row, weights=weights)
    stool, _ = calc_three_legged_stool(row)
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


def calc_four_pillars(row, themes, spy_ret=None, etf_data=None):
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
    fund = 0.0
    # Revenue growth > 0: up to 30 pts (scaled, cap at 60% growth)
    rev_g = _safe(row.get("RevGrowthPct"), 0)
    if rev_g > 0:
        fund += min(rev_g / 60, 1.0) * 30
    # Analyst upside > 0: up to 25 pts, scaled by analyst count confidence
    upside = _safe(row.get("AnalystUpside"), 0)
    num_analysts = int(_safe(row.get("NumAnalysts"), 0))
    analyst_conf = 0.0 if num_analysts == 0 else min(1.0, 0.4 + 0.2 * num_analysts)
    # 0 analysts → 0.0 (no signal), 1 → 0.6, 2 → 0.8, 3+ → 1.0
    if upside > 0:
        fund += min(upside / 100, 1.0) * 25 * analyst_conf
    # FCF positive: 20 pts
    if row.get("FCFPositive"):
        fund += 20
    # Rule of 40 > 40: up to 15 pts
    rule40 = _safe(row.get("Rule40"), 0)
    if rule40 > 40:
        fund += min((rule40 - 40) / 40, 1.0) * 15
    elif rule40 > 20:
        fund += (rule40 - 20) / 20 * 7.5
    # Insider buying: 10 pts (use InsiderSignal if available, else InsiderNet)
    insider_sig = str(row.get("InsiderSignal", "")).lower()
    insider_net = _safe(row.get("InsiderNet"), 0)
    if insider_sig == "buying" or insider_net > 0:
        fund += 10

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
    narr = 0.0
    # Analyst consensus buy/strong_buy: 30 pts
    rec = str(row.get("Recommendation", "")).lower().replace(" ", "_")
    if rec in ("strong_buy", "buy"):
        narr += 30
    elif rec == "hold":
        narr += 10
    # Short interest > 10% (squeeze potential): up to 20 pts
    short_pct = _safe(row.get("ShortPct"), 0)
    if short_pct > 10:
        narr += min((short_pct - 10) / 20, 1.0) * 20
    elif short_pct > 5:
        narr += (short_pct - 5) / 5 * 10
    # Volume spike (RelVol > 1.5): 20 pts
    rel_vol = _safe(row.get("RelVol"), 1.0)
    if rel_vol > 1.5:
        narr += min((rel_vol - 1.5) / 1.5, 1.0) * 20
    # Earnings coming within 30 days: 15 pts
    next_earn = row.get("NextEarnings")
    if next_earn and str(next_earn) != "N/A":
        try:
            earn_dt = pd.to_datetime(next_earn)
            days_to = (earn_dt - pd.Timestamp.now()).days
            if 0 <= days_to <= 30:
                narr += 15
        except Exception:
            pass
    # Skip headlines for now (not available in df_all)
    # narr += 15 if headlines exist

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
    st.divider()

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
    st.divider()

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

        st.divider()

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

        st.divider()

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

        st.divider()

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

    st.divider()

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
    st.divider()
    selected = st.multiselect("Filter view", st.session_state.tickers,
                               default=st.session_state.tickers)
    st.divider()
    st.subheader("AI Summary")
    claude_key_input = st.text_input("Anthropic API key", type="password",
                                      placeholder="sk-ant-...",
                                      help="Get one at console.anthropic.com")
    if claude_key_input:
        st.session_state["claude_key"] = claude_key_input
    st.caption("Used only in AI Summary tab. ~$0.001 per ticker.")
    st.divider()
    # Single-ticker deep-dive selector
    st.subheader("Ticker Deep Dive")
    deep_dive_ticker = st.selectbox("Select ticker", ["--"] + sorted(st.session_state.tickers), key="deep_dive")
    st.divider()

    # Score weight sliders
    with st.expander("Score Weights"):
        st.caption("Adjust asymmetry score component weights. Total does not need to equal 100.")
        w_upside = st.slider("Analyst Upside", 0, 50, 30, key="w_upside")
        w_beta = st.slider("Beta", 0, 30, 20, key="w_beta")
        w_atr = st.slider("Volatility (ATR%)", 0, 30, 15, key="w_atr")
        w_short = st.slider("Short Interest", 0, 30, 15, key="w_short")
        w_rsi = st.slider("RSI Positioning", 0, 20, 10, key="w_rsi")
        w_consensus = st.slider("Analyst Consensus", 0, 20, 10, key="w_consensus")
    st.session_state["asym_weights"] = {
        "Analyst Upside": w_upside,
        "Beta": w_beta,
        "Volatility (ATR%)": w_atr,
        "Short Interest": w_short,
        "RSI Positioning": w_rsi,
        "Analyst Consensus": w_consensus,
    }

    st.divider()

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

    st.divider()
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# ── Load all data once ───────────────────────────────────────────────────────

with st.spinner("Fetching price data..."):
    df_price = fetch_price_data(st.session_state.tickers)

if df_price.empty:
    st.error("No data. Check internet connection.")
    st.stop()

with st.spinner("Loading fundamentals & valuation..."):
    df_fund = fetch_fundamentals(st.session_state.tickers)

spy_ret = fetch_spy_returns()

# Fetch ETF benchmark data for all themes
_all_etf_tickers = get_all_etf_tickers(st.session_state.themes)
df_etf = fetch_etf_benchmark_data(tuple(_all_etf_tickers)) if _all_etf_tickers else pd.DataFrame()

df_price = df_price[df_price["Ticker"].isin(selected)].copy()

# Pre-compute merged dataframe and scores (used across all tabs)
df_all = df_price.merge(df_fund, on="Ticker", how="left")
_asym_weights = st.session_state.get("asym_weights")
df_all["ConvexityScore"] = df_all.apply(lambda r: calc_convexity_score(r, weights=_asym_weights), axis=1)
df_all["MomentumScore"]  = df_all.apply(calc_momentum_score, axis=1)
df_all["AsymmetryScore"]  = df_all.apply(lambda r: calc_asymmetry_score(r, weights=_asym_weights)[0], axis=1)
df_all["StoolScore"]      = df_all.apply(lambda r: calc_three_legged_stool(r)[0], axis=1)

# Setup Stage and Four-Pillar scores
df_all["SetupStage"] = df_all.apply(calc_setup_stage, axis=1)
_pillar_results = df_all.apply(
    lambda r: calc_four_pillars(r, st.session_state.themes, spy_ret, df_etf), axis=1
)
df_all["PillarTech"]    = _pillar_results.apply(lambda d: d["technical"])
df_all["PillarFund"]    = _pillar_results.apply(lambda d: d["fundamental"])
df_all["PillarTheme"]   = _pillar_results.apply(lambda d: d["thematic"])
df_all["PillarNarr"]    = _pillar_results.apply(lambda d: d["narrative"])
df_all["PillarAligned"] = _pillar_results.apply(lambda d: d["aligned"])

# Log scores for history tracking
log_scores(df_all)

# Failed tickers warning
_fetched = set(df_price["Ticker"].tolist())
_failed  = [t for t in st.session_state.tickers if t not in _fetched]
if _failed:
    st.warning(f"Failed to fetch data for: **{', '.join(_failed)}**. These tickers are excluded from analysis.")

st.title("Portfolio Analysis")
_data_ts = datetime.now().strftime('%Y-%m-%d %H:%M')
st.caption(f"As of {_data_ts}  |  **{st.session_state.active_watchlist}**  |  {len(df_price)} tickers")

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

    dd11, dd12, dd13, dd14, dd15 = st.columns(5)
    dd11.metric("ATR%", f"{t_row['ATR_pct']:.1f}%")
    dd12.metric("Short %", f"{_safe(t_row.get('ShortPct')):.1f}%" if _safe(t_row.get('ShortPct')) else "N/A")
    dd13.metric("FCF+", "Yes" if t_row.get("FCFPositive") else "No")
    dd14.metric("Rule of 40", f"{_safe(t_row.get('Rule40')):.0f}" if _safe(t_row.get('Rule40')) else "N/A")
    dd15.metric("Next Earnings", t_row.get("NextEarnings") or "N/A")

    # Relative strength
    if spy_ret:
        rs_1m = round(_safe(t_row.get("Ret1m")) - spy_ret.get("1m", 0), 1)
        rs_3m = round(_safe(t_row.get("Ret3m")) - spy_ret.get("3m", 0), 1)
        rs_6m = round(_safe(t_row.get("Ret6m")) - spy_ret.get("6m", 0), 1)
        st.caption(f"vs SPY — 1m: {rs_1m:+.1f}%  |  3m: {rs_3m:+.1f}%  |  6m: {rs_6m:+.1f}%")

    # Score breakdown
    _, asym_comps = calc_asymmetry_score(t_row, weights=_asym_weights)
    _, stool_comps = calc_three_legged_stool(t_row)
    with st.expander("Score Breakdown"):
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**Asymmetry Components**")
            for k, v in asym_comps.items():
                st.markdown(f"- {k}: **{v:.1f}** pts")
        with bc2:
            st.markdown("**Three-Legged Stool**")
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

    st.divider()

tab_dash, tab_conv, tab_themes, tab_charts, tab_sig, tab_ai = st.tabs([
    "Dashboard", "Convexity", "Themes", "Charts", "Signals", "AI Summary",
])

# ── TAB: Dashboard ───────────────────────────────────────────────────────────

with tab_dash:
    # ── a) Four-Pillar Alignment ─────────────────────────────────────────────
    st.subheader("Four-Pillar Alignment")
    st.caption("Technical + Fundamental + Thematic + Narrative — all four must score >= 50 for full alignment.")

    aligned_tickers = df_all[df_all["PillarAligned"] == True]
    aligned_count = len(aligned_tickers)
    if aligned_count > 0:
        st.success(f"**{aligned_count} fully aligned ticker{'s' if aligned_count != 1 else ''}** — all 4 pillars >= 50: "
                   f"**{', '.join(aligned_tickers['Ticker'].tolist())}**")
        # Show aligned tickers detail
        for _, arow in aligned_tickers.sort_values("PillarTech", ascending=False).iterrows():
            stage = arow.get("SetupStage", "Neutral")
            st.markdown(
                f"**{arow['Ticker']}**  |  Stage: `{stage}`  |  "
                f"Tech **{arow['PillarTech']:.0f}**  |  Fund **{arow['PillarFund']:.0f}**  |  "
                f"Theme **{arow['PillarTheme']:.0f}**  |  Narr **{arow['PillarNarr']:.0f}**"
            )
    else:
        st.info("No tickers currently have all four pillars aligned (>= 50).")

    # Full pillar table for all tickers
    _stage_colors = {
        "Basing": "#3498db", "Emerging": "#2ecc71", "Trending": "#27ae60",
        "Extended": "#e67e22", "Breaking Down": "#e74c3c", "Neutral": "#888",
    }

    def _color_stage(val):
        color = _stage_colors.get(val, "#888")
        return f"color: {color}"

    def _color_aligned(val):
        if val:
            return "background-color: #1a3a2a; color: #2ecc71"
        return "background-color: #5b2c2c; color: #ff6b6b"

    # Compute strong pillar count and average for sorting
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
        .format({
            "Price": "${:.2f}",
            "Technical": "{:.0f}",
            "Fundamental": "{:.0f}",
            "Thematic": "{:.0f}",
            "Narrative": "{:.0f}",
        }, na_rep="N/A")
    )
    st.dataframe(pillar_styled, use_container_width=True, hide_index=True)

    st.divider()

    # ── Top Convexity Picks (kept, moved below pillars) ──
    st.subheader("Top Convexity Picks")
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
        if not row.get("FCFPositive") and rev > 20:
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

    st.divider()

    # ── b) Active Alerts (kept) ──
    st.subheader("Active Alerts")
    overbought = df_price[df_price["RSI"] >= 70]["Ticker"].tolist()
    oversold   = df_price[df_price["RSI"] <= 30]["Ticker"].tolist()
    breakouts  = df_price[df_price["Breakout"] == True]["Ticker"].tolist()
    vol_spikes = df_price[df_price["RelVol"] >= 2.0]["Ticker"].tolist()

    alerts = []
    if oversold:
        alerts.append(f"**Oversold** (RSI <= 30): {', '.join(oversold)} -- potential entry")
    if overbought:
        alerts.append(f"**Overbought** (RSI >= 70): {', '.join(overbought)} -- consider trimming")
    if breakouts:
        alerts.append(f"**52wk Breakout** (>= 95%): {', '.join(breakouts)} -- momentum confirmation")
    if vol_spikes:
        alerts.append(f"**Volume Spike** (>= 2x avg): {', '.join(vol_spikes)} -- unusual activity")

    if alerts:
        for a in alerts:
            st.markdown(f"- {a}")
    else:
        st.info("No active alerts. Portfolio is in a neutral zone.")

    st.divider()

    # ── c) Setup Stage Summary ──
    st.subheader("Setup Stage Summary")
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

    st.divider()

    # ── d) Portfolio Overview table (kept, with Stage and Aligned columns added) ──
    st.subheader("Portfolio Overview")
    st.caption("All tickers at a glance. Convexity and Momentum are gradient-coloured. RSI and 52wk Pos are conditionally highlighted.")
    st.caption("⚠️ Short interest data is typically 2 weeks stale (FINRA bi-monthly reporting).")

    overview_df = df_all.copy()
    overview_df["Theme"] = overview_df["Ticker"].apply(
        lambda t: ", ".join(get_ticker_themes(st.session_state.themes, t)) or "")

    overview_cols = ["Ticker", "Price", "SetupStage", "PillarAligned", "ConvexityScore", "MomentumScore", "RSI", "Pos52",
                     "AnalystUpside", "RevGrowthPct", "PS_Current", "ShortPct", "Theme"]
    overview_disp = overview_df[overview_cols].copy()
    overview_disp["PillarAligned"] = overview_disp["PillarAligned"].apply(lambda x: "Yes" if x else "No")
    overview_disp = overview_disp.rename(columns={
        "SetupStage": "Stage", "PillarAligned": "Aligned",
        "ConvexityScore": "Convexity", "MomentumScore": "Momentum",
        "Pos52": "52wk Pos", "AnalystUpside": "Upside %",
        "RevGrowthPct": "Rev Growth %", "PS_Current": "P/S", "ShortPct": "Short %",
    })
    overview_disp = overview_disp.sort_values("Convexity", ascending=False).reset_index(drop=True)

    def _color_rsi(val):
        if pd.isna(val):
            return ""
        if val >= 70:
            return "background-color: #5b2c2c; color: #ff6b6b"
        if val <= 30:
            return "background-color: #1a3a2a; color: #2ecc71"
        return ""

    def _color_pos52(val):
        if pd.isna(val):
            return ""
        if val >= 70:
            return "background-color: #1a3a2a; color: #2ecc71"
        if val <= 20:
            return "background-color: #5b2c2c; color: #ff6b6b"
        return ""

    def _color_upside(val):
        if pd.isna(val):
            return ""
        if val >= 30:
            return "background-color: #1a3a2a; color: #2ecc71"
        if val <= -10:
            return "background-color: #5b2c2c; color: #ff6b6b"
        return ""

    def _color_short(val):
        if pd.isna(val):
            return ""
        if val >= 15:
            return "background-color: #3a2c1a; color: #f39c12"
        return ""

    styled = (overview_disp.style
        .background_gradient(subset=["Convexity"], cmap="RdYlGn", vmin=20, vmax=80)
        .background_gradient(subset=["Momentum"], cmap="RdYlGn", vmin=20, vmax=80)
        .map(_color_rsi, subset=["RSI"])
        .map(_color_pos52, subset=["52wk Pos"])
        .map(_color_upside, subset=["Upside %"])
        .map(_color_short, subset=["Short %"])
        .map(_color_stage, subset=["Stage"])
        .map(lambda v: _color_aligned(v == "Yes"), subset=["Aligned"])
        .format({
            "Price": "${:.2f}",
            "Convexity": "{:.0f}",
            "Momentum": "{:.0f}",
            "RSI": "{:.0f}",
            "52wk Pos": "{:.0f}%",
            "Upside %": lambda x: f"{x:+.0f}%" if pd.notna(x) else "N/A",
            "Rev Growth %": lambda x: f"{x:+.0f}%" if pd.notna(x) else "N/A",
            "P/S": lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A",
            "Short %": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        }, na_rep="N/A")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.download_button("Export Overview CSV", overview_disp.to_csv(index=False),
                       "portfolio_overview.csv", "text/csv", key="dl_overview")

    st.divider()

    # ── e) Upcoming Earnings (kept) ──
    st.subheader("Upcoming Earnings")
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
            st.info("No upcoming earnings dates found in the portfolio.")
    else:
        st.info("No earnings date data available.")

    st.divider()
    st.caption(f"Dashboard data refreshed: {_data_ts}")

# ── TAB: Convexity ───────────────────────────────────────────────────────────

with tab_conv:
    st.subheader("Convexity Analysis")
    st.caption(
        "Two complementary lenses on asymmetric upside. "
        "**Three-Legged Stool**: undervalued + growing + trending (the foundation). "
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

    st.divider()

    # Three-Legged Stool ranking
    st.markdown("#### Three-Legged Stool")
    st.caption("Green = all three legs present. Hover for component breakdown.")

    stool_sorted = df_conv.sort_values("StoolScore", ascending=False)
    stool_comps  = stool_sorted.apply(lambda r: calc_three_legged_stool(r)[1], axis=1)

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

    st.divider()

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

    st.divider()

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
        st.divider()
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
        st.dataframe(pfp_tbl, use_container_width=True, hide_index=True)

    st.divider()

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
    st.dataframe(val_styled, use_container_width=True, hide_index=True)
    st.download_button("Export Valuation CSV", val_tbl.to_csv(index=False),
                       "valuation_metrics.csv", "text/csv", key="dl_val")

    st.divider()

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
        st.dataframe(eb_styled, use_container_width=True, hide_index=True)
    else:
        st.info("No pre-flip candidates in current selection.")

    st.divider()

    # Rankings (merged from former Rankings tab)
    st.subheader("Convexity vs Momentum Rankings")
    st.caption(
        "**Convexity** (primary) = average of Asymmetry + Three-Legged Stool scores — "
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
        conv_tbl = conv[["Ticker","Price","ConvexityScore","AsymmetryScore","StoolScore","AnalystUpside","NextEarnings"]].copy()
        conv_tbl["NextEarnings"] = conv_tbl["NextEarnings"].fillna("N/A")
        conv_tbl = conv_tbl.rename(columns={
            "ConvexityScore":"Convexity","AsymmetryScore":"Asym",
            "StoolScore":"Stool","AnalystUpside":"Upside","NextEarnings":"Next Earnings",
        })
        st.dataframe(conv_tbl, use_container_width=True, hide_index=False,
                     column_config={
                         "Convexity": st.column_config.ProgressColumn("Convexity", min_value=0, max_value=100, format="%d"),
                         "Asym": st.column_config.ProgressColumn("Asym", min_value=0, max_value=100, format="%d"),
                         "Stool": st.column_config.ProgressColumn("Stool", min_value=0, max_value=100, format="%d"),
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
        st.dataframe(mom_tbl, use_container_width=True, hide_index=False,
                     column_config={
                         "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                         "52wk Pos": st.column_config.NumberColumn("52wk Pos", format="%.1f%%"),
                         "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
                         "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                     })
        st.download_button("Export Momentum CSV", mom_tbl.to_csv(index=False),
                           "momentum_ranking.csv", "text/csv", key="dl_mom")

    st.divider()
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
                     use_container_width=True, hide_index=True)
    else:
        st.info("No major divergences right now.")

# ── TAB: Themes ──────────────────────────────────────────────────────────────

with tab_themes:
    st.subheader("Theme / Sub-Sector Analysis")
    st.caption(
        "Custom sub-sector groupings benchmarked against representative ETFs. "
        "See where capital is flowing and how your picks compare to the sector."
    )

    themes_data = st.session_state.themes
    _user_tickers_map = get_theme_tickers(themes_data)
    _etf_tickers_map = get_theme_etfs(themes_data)
    _all_theme_names = list(themes_data.get("themes", {}).keys())

    # Theme management
    with st.expander("Manage Themes", expanded=False):
        mgmt_col1, mgmt_col2 = st.columns(2)
        with mgmt_col1:
            st.markdown("**Move ticker to theme**")
            move_ticker = st.selectbox("Ticker", sorted(st.session_state.tickers), key="theme_move_ticker")
            current_theme = next((th for th, tks in _user_tickers_map.items() if move_ticker in tks), "Untagged")
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
    st.divider()
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
            st.dataframe(trend_styled, use_container_width=True, hide_index=True)
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
    st.divider()
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
                        showgrid=False, titlefont=dict(color="#f39c12"),
                        tickfont=dict(color="#f39c12")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=60, t=50, b=40), **DARK,
        )
        fig_th.add_hline(y=50, line_dash="dot", line_color="#555")
        st.plotly_chart(fig_th, use_container_width=True)

    # ── (d) Theme Drilldown ──
    st.divider()
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
                st.dataframe(etf_styled, use_container_width=True, hide_index=True)
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
            st.dataframe(dr_styled, use_container_width=True, hide_index=True)

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
    st.subheader("Price Charts")

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

    chart_tickers = st.multiselect(
        "Select tickers to chart",
        df_price["Ticker"].tolist(),
        default=df_all.sort_values("ConvexityScore", ascending=False)["Ticker"].head(4).tolist()
        if len(df_all) >= 4 else df_all["Ticker"].tolist(),
        key="chart_tickers"
    )

    if chart_tickers:
        chart_cols = st.columns(min(len(chart_tickers), 2))
        for i, ct in enumerate(chart_tickers):
            with chart_cols[i % 2]:
                try:
                    ch_data = yf.download(ct, period=_ch_period, interval=_ch_interval,
                                          progress=False, auto_adjust=True)
                    if ch_data.empty:
                        st.caption(f"{ct}: no data")
                        continue
                    ch_close = ch_data["Close"].squeeze()
                    ch_vol = ch_data["Volume"].squeeze()
                    ch_ma_s = ch_close.rolling(_ch_ma_short).mean()
                    ch_ma_l = ch_close.rolling(_ch_ma_long).mean()

                    # Get current metrics for subtitle
                    cr = df_all[df_all["Ticker"] == ct]
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
                        height=380,
                        title=dict(text=f"{ct}  <span style='font-size:12px;color:#888'>{subtitle}</span>",
                                   font=dict(size=16)),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
                        margin=dict(l=50, r=15, t=50, b=30), **DARK,
                    )
                    fig_ch.update_yaxes(title_text="$", row=1, col=1, gridcolor="#21262d")
                    fig_ch.update_yaxes(row=2, col=1, gridcolor="#21262d")
                    st.plotly_chart(fig_ch, use_container_width=True)
                except Exception:
                    st.caption(f"{ct}: chart unavailable")
    else:
        st.info("Select at least one ticker to view charts.")

    st.divider()

    # Portfolio Map — RSI vs 52-Week Position (moved from Dashboard)
    st.subheader("Portfolio Map — RSI vs 52-Week Position")
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

    st.divider()

    # Relative Strength vs SPY (moved from Dashboard)
    st.subheader("Relative Strength vs SPY")
    if spy_ret:
        rs_rows = []
        for _, row in df_price.iterrows():
            rs_rows.append({
                "Ticker": row["Ticker"],
                "1m vs SPY": round(_safe(row.get("Ret1m")) - spy_ret.get("1m", 0), 1),
                "3m vs SPY": round(_safe(row.get("Ret3m")) - spy_ret.get("3m", 0), 1),
                "6m vs SPY": round(_safe(row.get("Ret6m")) - spy_ret.get("6m", 0), 1),
            })
        df_rs = pd.DataFrame(rs_rows).sort_values("3m vs SPY", ascending=True)

        fig_rs = go.Figure()
        for period, color in [("1m vs SPY", "#f1c40f"), ("3m vs SPY", "#3498db"), ("6m vs SPY", "#9b59b6")]:
            fig_rs.add_trace(go.Bar(
                name=period.replace(" vs SPY", ""),
                y=df_rs["Ticker"], x=df_rs[period], orientation="h",
                marker_color=color,
                hovertemplate=f"<b>%{{y}}</b><br>{period}: %{{x:+.1f}}%<extra></extra>",
            ))
        fig_rs.add_vline(x=0, line_color="#e6edf3", line_width=1)
        fig_rs.update_layout(
            barmode="group", height=max(400, len(df_rs) * 36),
            xaxis_title="Relative return vs SPY (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=60, r=20, t=50, b=40), **DARK,
        )
        st.plotly_chart(fig_rs, use_container_width=True)
        st.caption(f"SPY returns — 1m: {spy_ret.get('1m', 'N/A'):+.1f}%  |  3m: {spy_ret.get('3m', 'N/A'):+.1f}%  |  6m: {spy_ret.get('6m', 'N/A'):+.1f}%")
    else:
        st.info("SPY benchmark data unavailable.")

# ── TAB: Signals ─────────────────────────────────────────────────────────────

with tab_sig:
    st.subheader("Analyst Targets, News & Sentiment")

    with st.spinner("Loading news & sentiment..."):
        df_extras = fetch_extras(st.session_state.tickers)
        st_data   = fetch_stocktwits(st.session_state.tickers)

    df_sig = df_all.copy()
    df_sig = df_sig.merge(
        df_extras[["Ticker","InsiderSignal","InsiderNet","EarningsBeats"]],
        on="Ticker", how="left"
    )

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

    st.divider()

    # News & insider signals
    st.markdown("#### News & Insider Signals")
    news_cols = st.columns(2)
    tickers_list = df_sig.sort_values("AnalystUpside", ascending=False, na_position="last")["Ticker"].tolist()
    for i, t in enumerate(tickers_list):
        news_row = df_extras[df_extras["Ticker"] == t]
        if news_row.empty:
            continue
        nr = news_row.iloc[0]
        insider = nr.get("InsiderSignal","N/A")
        beats   = nr.get("EarningsBeats") or "N/A"
        headlines = nr.get("Headlines") or []
        with news_cols[i % 2]:
            with st.expander(f"{t}  |  Insider: {insider}  |  EPS beats: {beats}"):
                if headlines:
                    for h in headlines:
                        if isinstance(h, dict):
                            st.markdown(f"- **{h.get('date','')}** {h.get('title','')} _{h.get('pub','')}_")
                        else:
                            st.markdown(f"- {h}")
                else:
                    st.caption("No recent headlines found.")

    st.divider()

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

    sig_styled = (tbl.style
        .map(_sig_color_upside, subset=["Upside"])
        .map(_sig_color_short, subset=["Short%"])
        .map(_sig_color_rsi, subset=["RSI"])
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
    st.dataframe(sig_styled, use_container_width=True, hide_index=True)
    st.download_button("Export Signals CSV", tbl.to_csv(index=False),
                       "signals_data.csv", "text/csv", key="dl_sig")
    st.caption(
        "Insider = net buys vs sells last 90 days.  "
        "ST Bulls = StockTwits bull% from last 30 tagged messages."
    )
    st.caption("⚠️ Short interest data is typically 2 weeks stale (FINRA bi-monthly reporting).")

# ── TAB: AI Summary ─────────────────────────────────────────────────────────

with tab_ai:
    st.subheader("AI Summary — Claude-powered Investment Brief")
    st.caption(
        "Synthesizes technical data, analyst targets, insider activity, "
        "StockTwits sentiment, and recent news into a concise investment brief. "
        "Add your Anthropic API key in the sidebar (~$0.001 per ticker)."
    )

    api_key = st.session_state.get("claude_key", "")
    if not api_key:
        st.warning(
            "Add your Anthropic API key in the sidebar to use this tab.  \n"
            "Get one at **console.anthropic.com** — $5 free credit to start."
        )
    else:
        with st.spinner("Loading news & sentiment for AI analysis..."):
            df_extras_ai = fetch_extras(st.session_state.tickers)
            st_data_ai   = fetch_stocktwits(st.session_state.tickers)

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
            ex = df_extras_ai[df_extras_ai["Ticker"] == t].iloc[0] if not df_extras_ai[df_extras_ai["Ticker"] == t].empty else {}
            sent = st_data_ai.get(t, {})
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
            st.divider()
            for t in df_price["Ticker"].tolist():
                with st.spinner(f"Analyzing ${t}..."):
                    out = build_summary(t)
                with st.expander(f"**${t}**", expanded=False):
                    st.markdown(out)
            st.caption(f"All analyses generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
