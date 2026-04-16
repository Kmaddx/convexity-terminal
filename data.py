"""
Convexity Terminal — Data Fetching

All external data retrieval lives here: yfinance price data, fundamentals,
market environment, relative strength, ETF benchmarks, extras (news/insider),
StockTwits sentiment, scanner sources, and AI narrative synthesis.
"""

import os
import json
import time
import socket
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

# Force a 10s socket timeout on ALL network calls (including yfinance blocking HTTP)
# Without this, yf.Ticker().info can hang indefinitely when Yahoo is slow/returning 401s
socket.setdefaulttimeout(10)

import threading as _threading

from scoring import calc_rsi, calc_atr_pct, _safe, SECTOR_ETFS
from themes import _get_anthropic_key, _get_xai_key

# ── Finnhub client ────────────────────────────────────────────────────────────

def _get_finnhub_key():
    try:
        return st.secrets.get("FINNHUB_API_KEY", "") or os.getenv("FINNHUB_API_KEY", "")
    except Exception:
        return os.getenv("FINNHUB_API_KEY", "")

def _make_finnhub_client():
    key = _get_finnhub_key()
    if not key:
        return None
    try:
        import finnhub
        return finnhub.Client(api_key=key)
    except ImportError:
        return None

# Global rate limiter — Finnhub free tier: 60 calls/min, 30 calls/sec burst
# We target ~8 calls/sec globally to stay safely under burst limit
_fh_lock = _threading.Lock()
_fh_last_ts = 0.0
_FH_MIN_INTERVAL = 0.13  # seconds between calls globally (~7-8/sec)

def _fh_throttle():
    """Enforce minimum interval between Finnhub API calls."""
    global _fh_last_ts
    with _fh_lock:
        now = time.time()
        gap = _FH_MIN_INTERVAL - (now - _fh_last_ts)
        if gap > 0:
            time.sleep(gap)
        _fh_last_ts = time.time()

# ── xAI (Grok) client ────────────────────────────────────────────────────────

def _xai_chat(api_key, messages, model="grok-3-mini", max_tokens=400):
    """Call the xAI API (OpenAI-compatible). Returns response text or error string."""
    if not api_key:
        return None
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return f"[xAI {resp.status_code}] {resp.text[:300]}"
    except (requests.RequestException, requests.exceptions.Timeout, ValueError, KeyError) as e:
        return f"[xAI error] {e}"


def _xai_responses(api_key, messages, tools=None, model="grok-3", max_tokens=600):
    """Call xAI /v1/responses endpoint — required for Agent Tools (live search).

    Endpoint:  POST https://api.x.ai/v1/responses
    Tools:     [{"type": "x_search"}]  and/or  [{"type": "web_search"}]
    Response:  output[0]["content"][0]["text"]

    Ref: https://docs.x.ai/docs/guides/tools/overview
    """
    if not api_key:
        return None
    payload = {
        "model": model,
        "input": messages,          # same structure as chat "messages"
        "max_output_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    try:
        resp = requests.post(
            "https://api.x.ai/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
        if resp.status_code != 200:
            return f"[xAI {resp.status_code}] {resp.text[:300]}"
        data = resp.json()
        output = data.get("output", [])
        for block in output:
            for chunk in block.get("content", []):
                text = chunk.get("text", "")
                if text:
                    return text
        return None
    except (requests.RequestException, requests.exceptions.Timeout, ValueError, KeyError) as e:
        return f"[xAI error] {e}"


# ── File paths ───────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
FUND_CACHE_FILE = os.path.join(_DIR, "fund_cache.json")


# ── yfinance session helpers ──────────────────────────────────────────────────

def _clear_yf_cookies():
    """Delete yfinance cookie/crumb cache so next request gets a fresh session."""
    try:
        import glob as _glob, appdirs as _ad
        cache_dir = _ad.user_cache_dir("py-yfinance")
        for f in _glob.glob(os.path.join(cache_dir, "cookies.db*")):
            try:
                os.remove(f)
            except OSError:
                pass
    except (ModuleNotFoundError, ImportError, OSError):
        pass


_yf_refreshed = False   # only auto-refresh once per process lifetime


def _ensure_yf_session():
    """Clear stale cookies on first call each process, so we start with a fresh crumb."""
    global _yf_refreshed
    if not _yf_refreshed:
        _clear_yf_cookies()
        _yf_refreshed = True


# ── Fund cache helpers ───────────────────────────────────────────────────────

def _load_fund_cache():
    """Load cached fundamental data from disk. Stale data > no data."""
    try:
        if os.path.exists(FUND_CACHE_FILE):
            with open(FUND_CACHE_FILE) as f:
                cache = json.load(f)
            return cache.get("data", {})
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        pass
    return {}


def _save_fund_cache(data_by_ticker):
    """Save fundamental data to disk cache."""
    try:
        cache = {"_ts": datetime.now().timestamp(), "data": data_by_ticker}
        with open(FUND_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except (IOError, TypeError, OSError):
        pass


# ── Price data ───────────────────────────────────────────────────────────────

_price_fetch_error = ""  # Last error reason, readable by portfolio_app.py


def fetch_price_data(tickers):
    global _price_fetch_error
    import time
    _ensure_yf_session()
    _price_fetch_error = ""
    results = []
    ticker_list = list(tickers)
    # Batch download all tickers at once (much faster than sequential)
    # Retry up to 2 times with 2s backoff for transient failures
    raw = pd.DataFrame()
    for attempt in range(2):
        try:
            raw = yf.download(ticker_list, period="1y", interval="1d",
                              progress=False, auto_adjust=True, group_by="ticker", threads=True)
            if not raw.empty:
                break
        except (requests.RequestException, requests.exceptions.Timeout, Exception) as e:
            _price_fetch_error = str(e)
            if attempt == 0:
                time.sleep(2)  # Wait before retry
    for t in ticker_list:
        try:
            if len(ticker_list) == 1:
                df = raw
            else:
                df = raw[t].dropna(how="all") if t in raw.columns.get_level_values(0) else pd.DataFrame()
            # Fallback: batch download silently drops some tickers — fetch individually
            if df.empty or len(df) < 50:
                try:
                    df = yf.Ticker(t).history(period="1y", auto_adjust=True)
                except (requests.RequestException, requests.exceptions.Timeout, KeyError, ValueError):
                    pass
            if df.empty or len(df) < 50:
                continue
            close = df["Close"].squeeze()
            high  = df["High"].squeeze()
            low   = df["Low"].squeeze()
            vol   = df["Volume"].squeeze()
            price  = close.iloc[-1]
            ma20   = close.rolling(20).mean()
            ma50_s = close.rolling(50).mean()
            ma200_s = close.rolling(200).mean() if len(close) >= 200 else None
            ma20_last = ma20.iloc[-1]
            ma50   = ma50_s.iloc[-1]
            ma200  = ma200_s.iloc[-1] if ma200_s is not None else np.nan
            rsi    = calc_rsi(close).iloc[-1]
            atr_pct = calc_atr_pct(high, low, close)
            w52h   = high.max()
            w52l   = low.min()
            pos52  = ((price - w52l) / (w52h - w52l) * 100) if w52h != w52l else np.nan
            vs50   = ((price / ma50) - 1) * 100
            vs200  = ((price / ma200) - 1) * 100 if not np.isnan(ma200) else np.nan
            avg_vol = vol.iloc[-30:].mean()
            rel_vol = round(vol.iloc[-1] / avg_vol, 2) if avg_vol > 0 else np.nan
            ret_1m  = round(((price / close.iloc[-22]) - 1) * 100, 1) if len(close) >= 22 else None
            ret_3m  = round(((price / close.iloc[-63]) - 1) * 100, 1) if len(close) >= 63 else None
            ret_6m  = round(((price / close.iloc[-126]) - 1) * 100, 1) if len(close) >= 126 else None
            spark_30 = close.iloc[-30:].tolist() if len(close) >= 30 else close.tolist()

            # ── Coiled Base components ──────────────────────────────────────
            # Four measurable conditions; CoiledScore = how many fire (0-4).
            # 1) Basing: 20-day price range tight vs current price
            # 2) Tightening: 14d ATR well below 60d average ATR
            # 3) Compressing: Bollinger band width in bottom quintile of 126d
            # 4) MA Stack: Price > MA20 > MA50 > MA200 (and MA20/50 rising)
            base_tight = np.nan
            atr_contract = np.nan
            bb_width_pct = np.nan
            ma_stack = False
            coiled_score = None
            try:
                if len(close) >= 20 and price > 0:
                    hi20 = float(high.iloc[-20:].max())
                    lo20 = float(low.iloc[-20:].min())
                    base_tight = (hi20 - lo20) / price * 100
                # ATR contraction: current 14d ATR vs 60d average ATR
                if len(close) >= 60:
                    tr = pd.concat([high - low,
                                    (high - close.shift()).abs(),
                                    (low - close.shift()).abs()], axis=1).max(axis=1)
                    atr14 = tr.ewm(com=13, min_periods=14).mean()
                    atr_now = float(atr14.iloc[-1])
                    atr60_avg = float(atr14.iloc[-60:].mean())
                    if atr60_avg > 0:
                        atr_contract = atr_now / atr60_avg
                # BB width percentile rank (bottom 20% = compressing)
                if len(close) >= 126:
                    std20 = close.rolling(20).std()
                    bb_width = (std20 * 4) / ma20  # (upper-lower)/mid ≈ 4*std/mid
                    recent_window = bb_width.iloc[-126:].dropna()
                    if len(recent_window) >= 30:
                        cur_w = float(recent_window.iloc[-1])
                        bb_width_pct = float((recent_window <= cur_w).mean() * 100)
                # MA stack: Price > MA20 > MA50 > MA200, with MA20 and MA50 rising
                if (not np.isnan(ma200)
                        and not np.isnan(ma20_last)
                        and not np.isnan(ma50)
                        and len(ma20) >= 10 and len(ma50_s) >= 10):
                    rising_20 = float(ma20.iloc[-1]) > float(ma20.iloc[-10])
                    rising_50 = float(ma50_s.iloc[-1]) > float(ma50_s.iloc[-10])
                    ma_stack = bool(
                        price > ma20_last > ma50 > ma200
                        and rising_20 and rising_50
                    )
                coiled_score = int(
                    (1 if (not np.isnan(base_tight) and base_tight <= 12) else 0) +
                    (1 if (not np.isnan(atr_contract) and atr_contract <= 0.8) else 0) +
                    (1 if (not np.isnan(bb_width_pct) and bb_width_pct <= 20) else 0) +
                    (1 if ma_stack else 0)
                )
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
                coiled_score = None

            results.append(dict(
                Ticker=t, Price=round(price, 2),
                RSI=round(rsi, 1), vsMA50=round(vs50, 1),
                vsMA200=round(vs200, 1) if not np.isnan(vs200) else None,
                ATR_pct=round(atr_pct, 1), Pos52=round(pos52, 1),
                Low52=round(w52l, 2), High52=round(w52h, 2),
                RelVol=rel_vol, Breakout=bool(pos52 >= 95),
                Ret1m=ret_1m, Ret3m=ret_3m, Ret6m=ret_6m,
                Spark30=spark_30,
                BaseTightPct=round(base_tight, 1) if not np.isnan(base_tight) else None,
                ATRContract=round(atr_contract, 2) if not np.isnan(atr_contract) else None,
                BBWidthPct=round(bb_width_pct, 1) if not np.isnan(bb_width_pct) else None,
                MAStack=bool(ma_stack),
                CoiledScore=coiled_score,
            ))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return pd.DataFrame(results)


_spy_daily_fallback = (pd.Series(dtype=float), pd.Series(dtype=float))

@st.cache_data(ttl=300, show_spinner=False)
def fetch_spy_daily():
    """Fetch SPY 1y daily close & daily returns for down-day RS analysis."""
    global _spy_daily_fallback
    try:
        df = yf.download("SPY", period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return _spy_daily_fallback  # return last good data if available
        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        daily_ret = close.pct_change().dropna()
        _spy_daily_fallback = (close, daily_ret)  # remember last good fetch
        return close, daily_ret
    except (requests.RequestException, requests.exceptions.Timeout, KeyError, RuntimeError):
        # RuntimeError catches yfinance threading bugs ("dictionary changed size during iteration")
        return _spy_daily_fallback


@st.cache_data(ttl=300, show_spinner=False)
def fetch_spy_returns():
    """SPY benchmark returns for relative strength comparison.

    Only returns a key if its value is finite — NaN values silently propagate
    through every downstream comparison and show as 'None' in the UI.
    """
    try:
        df = yf.download("SPY", period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return {}
        close = df["Close"].squeeze().dropna()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0].dropna()
        if len(close) < 2:
            return {}
        price = float(close.iloc[-1])
        if not np.isfinite(price):
            return {}
        ret = {}
        for label, lookback in (("1m", 22), ("3m", 63), ("6m", 126)):
            if len(close) < lookback:
                continue
            prev = float(close.iloc[-lookback])
            if not np.isfinite(prev) or prev == 0:
                continue
            val = round((price / prev - 1) * 100, 1)
            if np.isfinite(val):
                ret[label] = val
        return ret
    except (requests.RequestException, requests.exceptions.Timeout, KeyError, RuntimeError):
        # RuntimeError catches yfinance threading bugs ("dictionary changed size during iteration")
        return {}


# ── Market Environment ───────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_environment():
    """Fetch market-wide data for environment scoring."""
    env = {}
    try:
        # Batch download indices (1 call instead of 5)
        idx_syms = ["^VIX", "^GSPC", "QQQ", "^TNX", "DX-Y.NYB"]
        idx_raw = yf.download(idx_syms, period="1y", interval="1d",
                              progress=False, auto_adjust=True, group_by="ticker", threads=True)

        def _get_idx(sym):
            try:
                d = idx_raw[sym].dropna(how="all")
                return d if not d.empty else pd.DataFrame()
            except (KeyError, AttributeError, IndexError):
                return pd.DataFrame()

        # VIX
        vix_df = _get_idx("^VIX")
        if not vix_df.empty:
            vix_close = vix_df["Close"].squeeze()
            env["vix_level"] = round(float(vix_close.iloc[-1]), 2)
            env["vix_ma20"] = round(float(vix_close.rolling(20).mean().iloc[-1]), 2)
            env["vix_pct_rank"] = round(float(
                (vix_close.iloc[-1] > vix_close.iloc[-252:]).mean() * 100
            ) if len(vix_close) >= 252 else 50, 0)
            env["vix_rising"] = bool(vix_close.iloc[-1] > vix_close.iloc[-5])

        # SPX
        spx_df = _get_idx("^GSPC")
        if not spx_df.empty:
            spx = spx_df["Close"].squeeze()
            env["spx_price"] = round(float(spx.iloc[-1]), 2)
            env["spx_vs_20d"] = round(float((spx.iloc[-1] / spx.rolling(20).mean().iloc[-1] - 1) * 100), 2)
            env["spx_vs_50d"] = round(float((spx.iloc[-1] / spx.rolling(50).mean().iloc[-1] - 1) * 100), 2)
            env["spx_vs_200d"] = round(float((spx.iloc[-1] / spx.rolling(200).mean().iloc[-1] - 1) * 100), 2) if len(spx) >= 200 else None

        # QQQ
        qqq_df = _get_idx("QQQ")
        if not qqq_df.empty:
            qqq = qqq_df["Close"].squeeze()
            env["qqq_price"] = round(float(qqq.iloc[-1]), 2)
            env["qqq_vs_50d"] = round(float((qqq.iloc[-1] / qqq.rolling(50).mean().iloc[-1] - 1) * 100), 2)
            env["qqq_ret_1m"] = round(float((qqq.iloc[-1] / qqq.iloc[-22] - 1) * 100), 1) if len(qqq) >= 22 else None

        # Batch download sector ETFs (1 call instead of 11)
        sector_data = {}
        try:
            sec_raw = yf.download(SECTOR_ETFS, period="3mo", interval="1d",
                                  progress=False, auto_adjust=True, group_by="ticker", threads=True)
            for etf in SECTOR_ETFS:
                try:
                    s_df = sec_raw[etf].dropna(how="all") if etf in sec_raw.columns.get_level_values(0) else pd.DataFrame()
                    if s_df.empty:
                        continue
                    s_close = s_df["Close"].squeeze()
                    ma50 = s_close.rolling(50).mean().iloc[-1]
                    ret_1d = round(float((s_close.iloc[-1] / s_close.iloc[-2] - 1) * 100), 2) if len(s_close) >= 2 else 0
                    ret_5d = round(float((s_close.iloc[-1] / s_close.iloc[-5] - 1) * 100), 2) if len(s_close) >= 5 else 0
                    above_50 = bool(s_close.iloc[-1] > ma50)
                    sector_data[etf] = {
                        "ret_1d": ret_1d, "ret_5d": ret_5d,
                        "above_50d": above_50, "price": round(float(s_close.iloc[-1]), 2),
                    }
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
        except (requests.RequestException, requests.exceptions.Timeout, KeyError, IndexError, ValueError):
            pass
        env["sectors"] = sector_data
        if sector_data:
            env["sectors_above_50d"] = sum(1 for s in sector_data.values() if s["above_50d"])
            env["sectors_positive_1d"] = sum(1 for s in sector_data.values() if s["ret_1d"] > 0)
        else:
            env["sectors_above_50d"] = None
            env["sectors_positive_1d"] = None
        env["sector_leader"] = max(sector_data.items(), key=lambda x: x[1]["ret_5d"])[0] if sector_data else None
        env["sector_laggard"] = min(sector_data.items(), key=lambda x: x[1]["ret_5d"])[0] if sector_data else None

        # ── Warning signals (signs of a terrible market) ──
        if not spx_df.empty:
            spx_close = spx_df["Close"].squeeze()
            spx_high = spx_df["High"].squeeze()
            spx_low = spx_df["Low"].squeeze()
            spx_open = spx_df["Open"].squeeze()
            spx_vol = spx_df["Volume"].squeeze()

            # Lower highs and lower lows (compare 5d rolling peaks/troughs)
            if len(spx_high) >= 40:
                recent_high = spx_high.iloc[-20:].max()
                prior_high = spx_high.iloc[-40:-20].max()
                recent_low = spx_low.iloc[-20:].min()
                prior_low = spx_low.iloc[-40:-20].min()
                env["lower_highs"] = bool(recent_high < prior_high)
                env["lower_lows"] = bool(recent_low < prior_low)

            # Rallies on low volume (up-day vol vs down-day vol, last 20 days)
            if len(spx_close) >= 21:
                last20_ret = spx_close.iloc[-20:].pct_change().dropna()
                last20_vol = spx_vol.iloc[-20:].iloc[1:]  # align with returns
                up_mask = last20_ret > 0
                down_mask = last20_ret < 0
                avg_up_vol = last20_vol[up_mask].mean() if up_mask.any() else 0
                avg_down_vol = last20_vol[down_mask].mean() if down_mask.any() else 0
                env["up_down_vol_ratio"] = round(float(avg_up_vol / avg_down_vol), 2) if avg_down_vol > 0 else None
                env["low_vol_rallies"] = bool(avg_up_vol < avg_down_vol) if avg_down_vol > 0 else False

            # Good opens, weak closes (intraday fade: open above prior close, close below open)
            if len(spx_close) >= 11:
                last10_open = spx_open.iloc[-10:]
                last10_close = spx_close.iloc[-10:]
                prev_close = spx_close.iloc[-11:-1].values
                fade_days = sum(
                    1 for o, c, pc in zip(last10_open, last10_close, prev_close)
                    if float(o) > float(pc) and float(c) < float(o)
                )
                env["fade_days_10d"] = int(fade_days)
                env["weak_closes"] = bool(fade_days >= 5)

            # Distribution days (down >0.2% on above-average volume, last 25 sessions)
            if len(spx_close) >= 50:
                avg_vol_50 = spx_vol.iloc[-50:].mean()
                last25_ret = spx_close.iloc[-25:].pct_change().dropna()
                last25_vol = spx_vol.iloc[-25:].iloc[1:]
                dist_days = sum(
                    1 for r, v in zip(last25_ret, last25_vol)
                    if float(r) < -0.002 and float(v) > float(avg_vol_50)
                )
                env["distribution_days"] = int(dist_days)
                env["high_distribution"] = bool(dist_days >= 5)

        # 10Y yield (already batch-downloaded)
        tnx_df = _get_idx("^TNX")
        if not tnx_df.empty:
            tnx = tnx_df["Close"].squeeze()
            env["tnx_yield"] = round(float(tnx.iloc[-1]), 2)
            env["tnx_vs_20d"] = round(float((tnx.iloc[-1] / tnx.rolling(20).mean().iloc[-1] - 1) * 100), 2)
            env["tnx_rising"] = bool(tnx.iloc[-1] > tnx.iloc[-5])

        # DXY (already batch-downloaded)
        dxy_df = _get_idx("DX-Y.NYB")
        if not dxy_df.empty:
            dxy = dxy_df["Close"].squeeze()
            env["dxy_level"] = round(float(dxy.iloc[-1]), 2)
            env["dxy_vs_20d"] = round(float((dxy.iloc[-1] / dxy.rolling(20).mean().iloc[-1] - 1) * 100), 2)
            env["dxy_strengthening"] = bool(dxy.iloc[-1] > dxy.iloc[-5])

    except (requests.RequestException, requests.exceptions.Timeout, KeyError, IndexError, ValueError):
        pass
    return env


# ── Relative Strength (ZA framework) ────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def calc_downday_rs(tickers, spy_close, spy_daily_ret):
    """Calculate relative strength on SPY down-days for each ticker.

    Core idea from ZA: "What isn't going lower?" — stocks that hold up on market
    down-days are being accumulated by institutions.
    """
    if spy_close.empty or spy_daily_ret.empty:
        return pd.DataFrame()

    threshold = spy_daily_ret.quantile(0.40)
    down_days = spy_daily_ret[spy_daily_ret <= threshold].index
    recent_cutoff_1m = spy_daily_ret.index[-22] if len(spy_daily_ret) >= 22 else spy_daily_ret.index[0]
    recent_down_days = [d for d in down_days if d >= recent_cutoff_1m]

    # Batch download all tickers at once
    ticker_list = list(tickers)
    try:
        raw = yf.download(ticker_list, period="1y", interval="1d",
                          progress=False, auto_adjust=True, group_by="ticker", threads=True)
    except (requests.RequestException, requests.exceptions.Timeout):
        raw = pd.DataFrame()

    results = []
    for t in ticker_list:
        try:
            if len(ticker_list) == 1:
                df = raw
            else:
                df = raw[t].dropna(how="all") if t in raw.columns.get_level_values(0) else pd.DataFrame()
            if df.empty or len(df) < 50:
                continue
            close = df["Close"].squeeze()
            stock_ret = close.pct_change().dropna()
            common = stock_ret.index.intersection(spy_daily_ret.index)
            if len(common) < 30:
                continue
            s_ret = stock_ret.reindex(common)
            spy_r = spy_daily_ret.reindex(common)

            dd_idx = [d for d in down_days if d in common]
            if len(dd_idx) < 5:
                continue
            stock_on_dd = s_ret.loc[dd_idx]
            spy_on_dd = spy_r.loc[dd_idx]
            excess_on_dd = stock_on_dd - spy_on_dd

            dd_rs = round(excess_on_dd.mean() * 100, 3)
            dd_win_rate = round((excess_on_dd > 0).sum() / len(excess_on_dd) * 100, 1)

            recent_dd_idx = [d for d in recent_down_days if d in common]
            if len(recent_dd_idx) >= 3:
                recent_excess = (s_ret.loc[recent_dd_idx] - spy_r.loc[recent_dd_idx]).mean() * 100
            else:
                recent_excess = dd_rs

            price = close.iloc[-1]
            rs_1m = round(((price / close.iloc[-22]) - 1) * 100 - ((spy_close.iloc[-1] / spy_close.iloc[-22]) - 1) * 100, 1) if len(close) >= 22 and len(spy_close) >= 22 else 0
            rs_3m = round(((price / close.iloc[-63]) - 1) * 100 - ((spy_close.iloc[-1] / spy_close.iloc[-63]) - 1) * 100, 1) if len(close) >= 63 and len(spy_close) >= 63 else 0

            lookback = min(60, len(close) - 1, len(spy_close) - 1)
            stock_low_idx = close.iloc[-lookback:].idxmin()
            spy_low_idx = spy_close.iloc[-lookback:].idxmin()
            early_bottom = bool(stock_low_idx < spy_low_idx) if pd.notna(stock_low_idx) and pd.notna(spy_low_idx) else False
            days_since_low = (close.index[-1] - stock_low_idx).days if pd.notna(stock_low_idx) else 0

            score = 0
            score += min(max((dd_win_rate - 30) / 40, 0), 1.0) * 35
            score += min(max((recent_excess + 1) / 2, 0), 1.0) * 25
            score += min(max((rs_1m + 15) / 30, 0), 1.0) * 25
            score += (15 if early_bottom else 0)
            score = round(min(score, 100), 1)

            results.append({
                "Ticker": t,
                "DownDayRS": round(dd_rs, 2),
                "DownDayWinRate": dd_win_rate,
                "RecentDDExcess": round(recent_excess, 2),
                "RS_1m": rs_1m, "RS_3m": rs_3m,
                "RS_Score": score,
                "EarlyBottom": early_bottom,
                "DaysSinceLow": days_since_low,
            })
        except (KeyError, IndexError, TypeError, ValueError):
            continue

    if not results:
        return pd.DataFrame()

    df_rs = pd.DataFrame(results).sort_values("RS_Score", ascending=False)
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


# ── ETF Benchmark Data ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_etf_benchmark_data(etf_tickers):
    """Fetch 1-year price data for ETF benchmarks."""
    results = []
    ticker_list = list(etf_tickers)
    if not ticker_list:
        return pd.DataFrame()
    # Batch download all ETFs at once
    try:
        raw = yf.download(ticker_list, period="1y", interval="1d",
                          progress=False, auto_adjust=True, group_by="ticker", threads=True)
    except (requests.RequestException, requests.exceptions.Timeout):
        raw = pd.DataFrame()
    for t in ticker_list:
        try:
            if len(ticker_list) == 1:
                df = raw
            else:
                df = raw[t].dropna(how="all") if t in raw.columns.get_level_values(0) else pd.DataFrame()
            if df.empty or len(df) < 22:
                try:
                    df = yf.Ticker(t).history(period="1y", auto_adjust=True)
                except (requests.RequestException, requests.exceptions.Timeout, KeyError):
                    pass
            if df.empty or len(df) < 22:
                continue
            close = df["Close"].squeeze()
            price = close.iloc[-1]
            ret_1m = round(((price / close.iloc[-22]) - 1) * 100, 1) if len(close) >= 22 else None
            ret_3m = round(((price / close.iloc[-63]) - 1) * 100, 1) if len(close) >= 63 else None
            ret_6m = round(((price / close.iloc[-126]) - 1) * 100, 1) if len(close) >= 126 else None
            ret_1y = round(((price / close.iloc[0]) - 1) * 100, 1) if len(close) >= 200 else None
            ytd_start = close[close.index >= f"{datetime.now().year}-01-01"]
            ret_ytd = round(((price / ytd_start.iloc[0]) - 1) * 100, 1) if len(ytd_start) > 1 else None
            results.append(dict(
                Ticker=t, Price=round(price, 2),
                Ret1m=ret_1m, Ret3m=ret_3m, Ret6m=ret_6m,
                Ret1y=ret_1y, RetYTD=ret_ytd,
            ))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return pd.DataFrame(results)


# ── Fundamentals ─────────────────────────────────────────────────────────────

def _fetch_single_fundamental(t, _disk_cache, fh_client=None):
    """Fetch fundamental data. Uses Finnhub if key available, else yfinance."""
    if fh_client is not None:
        return _fetch_fundamental_finnhub(t, _disk_cache, fh_client)
    return _fetch_fundamental_yfinance(t, _disk_cache)


def _fetch_fundamental_finnhub(t, _disk_cache, client):
    """Fetch fundamentals via Finnhub — no threading deadlocks, reliable."""
    row = {"Ticker": t}
    _rate_limited = False
    try:
        # ── 1. Basic financial metrics ────────────────────────────────────
        _fh_throttle()
        resp = client.company_basic_financials(t, 'all')
        m = resp.get('metric', {}) or {}

        rev_g = m.get('revenueGrowthTTMYoy')       # already in %
        gm    = m.get('grossMarginTTM')             # already in %
        npm   = m.get('netProfitMarginTTM')         # already in %
        beta  = m.get('beta')

        row['RevGrowthPct']  = round(rev_g, 1) if rev_g is not None else None
        row['GrossMargin']   = round(gm, 1)    if gm  is not None else None
        row['EV_Sales']      = round(m['evRevenueTTM'], 2) if m.get('evRevenueTTM') else None
        row['EV_EBITDA']     = round(m['evEbitdaTTM'], 1)  if m.get('evEbitdaTTM') and m['evEbitdaTTM'] > 0 else None
        row['Rule40']        = round(rev_g + npm, 1) if (rev_g is not None and npm is not None) else None
        row['MarketCap']     = int(m['marketCapitalization'] * 1e6) if m.get('marketCapitalization') else None

        # Beta
        if beta is not None and 0 < beta <= 5:
            row['Beta'] = round(beta, 2); row['BetaReliable'] = True
        else:
            row['Beta'] = beta; row['BetaReliable'] = False

        # FCF: positive if EV/FCF ratio is positive (EV is always positive for listed cos)
        ev_fcf = m.get('currentEv/freeCashFlowTTM')
        row['FCFPositive']      = bool(ev_fcf and ev_fcf > 0)
        row['FCFValue']         = None   # exact value not derivable from free-tier metrics
        row['CashRunwayMonths'] = None

        # Short interest — may be None on free tier
        short_pct = m.get('shortPercent')
        row['ShortPct']    = round(float(short_pct) * 100, 1) if short_pct else None
        row['DaysToCover'] = m.get('shortInterestRatio')

        # Insider ownership — often None on free tier, default 0
        ins_own = m.get('insiderOwnershipPercent')
        row['InsiderPct'] = round(float(ins_own), 1) if ins_own else 0.0
        row['InstitPct']  = None   # not available on free tier

        # ── 2. Quote — day change, overnight ─────────────────────────────
        _fh_throttle()
        quote = client.quote(t)
        row['DayChgPct']   = round(quote['dp'], 2) if quote.get('dp') is not None else None
        row['PostMktChg']  = None   # not on free tier
        row['PreMktChg']   = None
        row['OvernightChg'] = None

        # ── 3. Recommendation trends — analyst consensus ──────────────────
        _fh_throttle()
        try:
            rec = client.recommendation_trends(t)
            if rec:
                latest = rec[0]
                n = (latest.get('buy', 0) + latest.get('strongBuy', 0) +
                     latest.get('sell', 0) + latest.get('strongSell', 0) +
                     latest.get('hold', 0))
                row['NumAnalysts'] = n
                buy_n  = latest.get('buy', 0) + latest.get('strongBuy', 0)
                sell_n = latest.get('sell', 0) + latest.get('strongSell', 0)
                if buy_n > sell_n * 1.5:    row['Recommendation'] = 'buy'
                elif sell_n > buy_n * 1.5:  row['Recommendation'] = 'sell'
                else:                        row['Recommendation'] = 'hold'
            else:
                row['NumAnalysts'] = 0
        except Exception:
            row['NumAnalysts'] = 0
        # Price target requires paid tier — skip analyst upside
        row['AnalystUpside'] = None
        row['TargetMean']    = None
        row['TargetLow']     = None
        row['TargetHigh']    = None

        # ── 4. Earnings calendar ──────────────────────────────────────────
        _fh_throttle()
        try:
            today_str = datetime.now().strftime('%Y-%m-%d')
            fut_str   = (datetime.now().replace(year=datetime.now().year + 1)).strftime('%Y-%m-%d')
            ec = client.earnings_calendar(symbol=t, _from=today_str, to=fut_str)
            cal = ec.get('earningsCalendar', [])
            if cal:
                row['NextEarnings'] = cal[0].get('date')
            else:
                row['NextEarnings'] = None
        except Exception:
            row['NextEarnings'] = None

        # Alias so existing call sites using either name work
        row['RevenueGrowth'] = row.get('RevGrowthPct')

        # Sector / Industry — not in Finnhub basic metrics; will be missing
        # (sector percentile ranking uses watchlist data, not this field)
        row['Sector']   = ''
        row['Industry'] = ''

    except Exception as e:
        _rate_limited = '429' in str(e) or 'rate' in str(e).lower()
        if t in _disk_cache:
            row.update(_disk_cache[t])

    cache_entry = None
    if len(row) > 1:
        cache_entry = {k: v for k, v in row.items()
                       if k != 'Ticker' and not isinstance(v, (list, dict))
                       and v is not None}
    return row, cache_entry, _rate_limited


def _fetch_fundamental_yfinance(t, _disk_cache):
    """Fallback: fetch fundamentals via yfinance (legacy, prone to hangs)."""
    import random
    row = {"Ticker": t}
    _rate_limited = False
    try:
        time.sleep(random.uniform(0.1, 0.5))   # stagger parallel requests
        obj = yf.Ticker(t)
        info = obj.info
        _has_real_data = info and any(info.get(k) is not None for k in
            ["currentPrice", "revenueGrowth", "marketCap", "freeCashflow"])
        if not _has_real_data:
            _rate_limited = True
            if t in _disk_cache:
                row.update(_disk_cache[t])
            cache_entry = None
            if len(row) > 1:
                cache_entry = {k: v for k, v in row.items()
                               if k != "Ticker" and not isinstance(v, (list, dict))
                               and v is not None}
            return row, cache_entry, _rate_limited

        target_mean = info.get("targetMeanPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        upside = round((target_mean / current - 1) * 100, 1) if target_mean and current else None
        short_pct = info.get("shortPercentOfFloat")
        rev_growth = info.get("revenueGrowth")
        earnings_ts = info.get("earningsDate") or info.get("earningsTimestamp")
        next_earnings = None
        if isinstance(earnings_ts, list) and earnings_ts:
            try:
                next_earnings = datetime.fromtimestamp(earnings_ts[0]).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        elif isinstance(earnings_ts, (int, float)) and earnings_ts:
            try:
                next_earnings = datetime.fromtimestamp(earnings_ts).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
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

        ps = info.get("priceToSalesTrailing12Months")
        ev = info.get("enterpriseToEbitda")
        gm = info.get("grossMargins")
        row["PS_Current"] = round(ps, 2) if ps else None
        row["EV_EBITDA"] = round(ev, 1) if ev and ev > 0 else None
        row["GrossMargin"] = round(gm * 100, 1) if gm else None

        # EV/Sales — preferred over P/S as it accounts for debt
        ev_raw = info.get("enterpriseValue")
        rev_raw = info.get("totalRevenue")
        row["EV_Sales"] = round(ev_raw / rev_raw, 2) if ev_raw and rev_raw and rev_raw > 0 else None
        row["Sector"] = info.get("sector", "")
        row["Industry"] = info.get("industry", "")

        prof = info.get("profitMargins")
        row["RevGrowthPct"] = round(rev_growth * 100, 1) if rev_growth is not None else None
        row["Rule40"] = round(rev_growth * 100 + prof * 100, 1) if (rev_growth is not None and prof is not None) else None

        fcf = info.get("freeCashflow")
        total_cash = info.get("totalCash") or 0
        row["FCFPositive"] = bool(fcf and fcf > 0)
        row["FCFValue"] = fcf
        row["CashRunwayMonths"] = (round((total_cash / abs(fcf)) * 12, 1)
                                   if fcf and fcf < 0 and total_cash else None)

        row["InsiderPct"] = round((info.get("heldPercentInsiders") or 0) * 100, 1)
        row["InstitPct"] = round((info.get("heldPercentInstitutions") or 0) * 100, 1)
        row["DaysToCover"] = info.get("shortRatio")

        day_chg = info.get("regularMarketChangePercent")
        row["DayChgPct"] = round(day_chg, 2) if day_chg is not None else None

        post_chg = info.get("postMarketChangePercent")
        pre_chg = info.get("preMarketChangePercent")
        row["PostMktChg"] = round(post_chg, 2) if post_chg is not None else None
        row["PreMktChg"] = round(pre_chg, 2) if pre_chg is not None else None

        # Overnight: total move from regular close to latest extended-hours price
        reg_price = info.get("regularMarketPrice")
        post_price = info.get("postMarketPrice")
        pre_price = info.get("preMarketPrice")
        ext_price = pre_price or post_price
        if ext_price and reg_price and reg_price > 0:
            row["OvernightChg"] = round((ext_price / reg_price - 1) * 100, 2)
        else:
            row["OvernightChg"] = None

        # Historical EV/Sales range (3 years) — DISABLED IN CONCURRENT CONTEXT
        # yfinance has severe deadlock/race condition bugs when called from ThreadPoolExecutor
        # The 3yr historical range is nice-to-have but not critical for scoring
        # Current EV/Sales + sector percentile are sufficient and come from .info
        # Streamlit caches this anyway, so users will see historical ranges on cache hits
        pass  # Skipping historical fetch entirely to prevent hangs
    except (requests.RequestException, requests.exceptions.Timeout, KeyError, ValueError, TypeError):
        if t in _disk_cache:
            row.update(_disk_cache[t])
    cache_entry = None
    if len(row) > 1:
        cache_entry = {k: v for k, v in row.items()
                       if k != "Ticker" and not isinstance(v, (list, dict))
                       and v is not None}
    return row, cache_entry, _rate_limited


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(tickers):
    """Fetch analyst/fundamental data per ticker using parallel threads."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _disk_cache = _load_fund_cache()
    _new_cache = {}
    _rate_limited = False

    fh_client = _make_finnhub_client()   # None if key missing or package absent
    pool = ThreadPoolExecutor(max_workers=4)
    futures = {pool.submit(_fetch_single_fundamental, t, _disk_cache, fh_client): t for t in tickers}
    results_map = {}
    try:
        for future in as_completed(futures, timeout=45):
            t = futures[future]
            try:
                row, cache_entry, rl = future.result(timeout=12)
                results_map[t] = row
                if cache_entry:
                    _new_cache[t] = cache_entry
                if rl:
                    _rate_limited = True
            except Exception:
                results_map[t] = _disk_cache.get(t, {"Ticker": t})
                if t in _disk_cache:
                    results_map[t]["Ticker"] = t
    except TimeoutError:
        # Some tickers timed out — use disk cache for missing ones
        for t in tickers:
            if t not in results_map:
                results_map[t] = _disk_cache.get(t, {"Ticker": t})
                if t in _disk_cache:
                    results_map[t]["Ticker"] = t
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # Preserve original ticker order
    results = [results_map.get(t, {"Ticker": t}) for t in tickers]

    if _new_cache and not _rate_limited:
        _save_fund_cache(_new_cache)
    elif _rate_limited and _disk_cache:
        merged = {**_disk_cache, **_new_cache}
        _save_fund_cache(merged)
    return pd.DataFrame(results)


# ── Extras (news, insider, earnings) ─────────────────────────────────────────

def _fetch_single_extra(t):
    """Fetch news, insider, earnings for a single ticker. Used by ThreadPoolExecutor."""
    row = {"Ticker": t, "InsiderSignal": "N/A", "InsiderNet": 0,
           "InsiderBuyScore": 0.0, "InsiderSellScore": 0.0, "InsiderCluster": False,
           "Headlines": [], "InsiderBuys": []}
    try:
        obj = yf.Ticker(t)
        try:
            raw_news = obj.news or []
            headlines = []
            for n in raw_news[:5]:
                title = (n.get("title") or (n.get("content") or {}).get("title", ""))
                pub = (n.get("publisher") or
                       (n.get("content") or {}).get("provider", {}).get("displayName", ""))
                ts = n.get("providerPublishTime") or n.get("pubDate", "")
                if title:
                    try:
                        date_str = datetime.fromtimestamp(int(ts)).strftime("%b %d") if isinstance(ts, (int, float)) else str(ts)[:10]
                    except (ValueError, TypeError):
                        date_str = ""
                    headlines.append({"date": date_str, "title": title, "pub": pub})
            row["Headlines"] = headlines
        except (requests.RequestException, requests.exceptions.Timeout, KeyError, ValueError):
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

                tx_col = next((c for c in recent.columns if "text" in c.lower()), None)
                if tx_col is None:
                    tx_col = next((c for c in recent.columns if "transaction" in c.lower()), None)

                if tx_col is not None:
                    tx = recent[tx_col].astype(str).str.lower()

                    # ── Filter noise transactions ──────────────────────────────
                    # Exclude: derivative exercises, grants/awards, $0 price
                    noise_mask = (
                        tx.str.contains("exercise|conversion|derivative|convert|grant|award|gift", na=False) |
                        (recent.get("Value", pd.Series(dtype=float)).fillna(0).astype(float) == 0)
                    )
                    # Exclude 10%+ beneficial owners (institutional, not management signal)
                    pos_col = next((c for c in recent.columns if "position" in c.lower()), None)
                    if pos_col:
                        pos = recent[pos_col].astype(str).str.lower()
                        noise_mask = noise_mask | pos.str.contains("beneficial owner|10%|10 percent", na=False)

                    discretionary = recent[~noise_mask]
                    tx_disc = tx[~noise_mask]

                    buy_mask  = tx_disc.str.contains("buy|purchase|acquisition", na=False)
                    sell_mask = tx_disc.str.contains("sell|sale|disposition", na=False)

                    # ── Role weights ───────────────────────────────────────────
                    def _role_weight(position):
                        p = str(position).lower()
                        if any(x in p for x in ["chief executive", "ceo", "president"]):
                            return 3.0
                        if any(x in p for x in ["chief financial", "cfo", "chief operating", "coo"]):
                            return 2.5
                        if any(x in p for x in ["director"]):
                            return 1.5
                        if any(x in p for x in ["officer", "vp", "vice president"]):
                            return 1.5
                        return 1.0

                    # ── Dollar value weight ────────────────────────────────────
                    def _value_weight(val):
                        v = float(val) if pd.notna(val) else 0
                        if v >= 500_000: return 3.0
                        if v >= 100_000: return 2.0
                        if v >= 10_000:  return 1.0
                        return 0.5

                    buy_details = []
                    buy_score = 0.0
                    sell_score = 0.0
                    distinct_buyers = set()

                    buy_rows = discretionary[buy_mask]
                    for _, br in buy_rows.iterrows():
                        pos  = br.get("Position", "") or br.get(pos_col, "") if pos_col else br.get("Position", "")
                        val  = br.get("Value", 0)
                        w    = _role_weight(pos) * _value_weight(val)
                        buy_score += w
                        insider_name = br.get("Insider", "Unknown")
                        distinct_buyers.add(str(insider_name))
                        date_val = br.get("Start Date", "")
                        try:
                            date_str = pd.to_datetime(date_val).strftime("%b %d") if pd.notna(date_val) else ""
                        except (ValueError, TypeError):
                            date_str = str(date_val)[:10]
                        shares_val = br.get("Shares", 0)
                        buy_details.append({
                            "insider":   insider_name,
                            "position":  str(pos),
                            "date":      date_str,
                            "value":     float(val) if pd.notna(val) else 0,
                            "shares":    int(shares_val) if pd.notna(shares_val) else 0,
                            "weight":    round(w, 1),
                        })

                    sell_rows = discretionary[sell_mask]
                    for _, sr in sell_rows.iterrows():
                        pos = sr.get("Position", "") or sr.get(pos_col, "") if pos_col else sr.get("Position", "")
                        val = sr.get("Value", 0)
                        sell_score += _role_weight(pos) * _value_weight(val)

                    row["InsiderBuys"] = buy_details
                    row["InsiderBuyScore"]  = round(buy_score, 1)
                    row["InsiderSellScore"] = round(sell_score, 1)
                    row["InsiderCluster"]   = len(distinct_buyers) >= 3

                    # ── Signal classification ──────────────────────────────────
                    if buy_score == 0 and sell_score == 0:
                        signal = "Neutral"
                    elif buy_score >= sell_score * 1.5:
                        if len(distinct_buyers) >= 3:
                            signal = "Cluster Buy"
                        elif buy_score >= 4.0:
                            signal = "Strong Buy"
                        else:
                            signal = "Buying"
                    elif sell_score > buy_score * 1.5:
                        signal = "Selling"
                    else:
                        signal = "Neutral"

                    row["InsiderNet"]    = round(buy_score - sell_score, 1)
                    row["InsiderSignal"] = signal
                else:
                    row["InsiderBuyScore"]  = 0.0
                    row["InsiderSellScore"] = 0.0
                    row["InsiderCluster"]   = False
        except (KeyError, AttributeError, TypeError, ValueError):
            pass
        # EarningsBeats removed — minor signal, heavy API call per ticker
    except (requests.RequestException, requests.exceptions.Timeout, KeyError, AttributeError, ValueError):
        pass
    return row


def _fetch_single_extra_finnhub(t, client):
    """Fetch news and insider transactions via Finnhub. No yfinance, no hangs."""
    row = {"Ticker": t, "InsiderSignal": "N/A", "InsiderNet": 0,
           "InsiderBuyScore": 0.0, "InsiderSellScore": 0.0, "InsiderCluster": False,
           "Headlines": [], "InsiderBuys": []}
    try:
        # ── News ──────────────────────────────────────────────────────────────
        _fh_throttle()
        try:
            today   = datetime.now()
            from_dt = (today - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            to_dt   = today.strftime('%Y-%m-%d')
            news = client.company_news(t, _from=from_dt, to=to_dt)
            headlines = []
            for n in (news or [])[:5]:
                title = n.get('headline', '')
                pub   = n.get('source', '')
                ts    = n.get('datetime')
                if title:
                    try:
                        date_str = datetime.fromtimestamp(ts).strftime('%b %d') if ts else ''
                    except (ValueError, TypeError, OSError):
                        date_str = ''
                    headlines.append({'date': date_str, 'title': title, 'pub': pub})
            row['Headlines'] = headlines
        except Exception:
            pass

        # ── Insider transactions ──────────────────────────────────────────────
        _fh_throttle()
        try:
            today   = datetime.now()
            from_dt = (today - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
            to_dt   = today.strftime('%Y-%m-%d')
            ins_resp = client.stock_insider_transactions(t, from_dt, to_dt)
            txns     = (ins_resp or {}).get('data', []) or []

            def _value_weight(price, shares):
                v = abs(float(price or 0) * float(shares or 0))
                if v >= 500_000: return 3.0
                if v >= 100_000: return 2.0
                if v >= 10_000:  return 1.0
                return 0.5

            buy_details     = []
            buy_score       = 0.0
            sell_score      = 0.0
            distinct_buyers = set()

            for tx in txns:
                code   = (tx.get('transactionCode') or '').upper()
                name   = tx.get('name', 'Unknown')
                price  = tx.get('transactionPrice') or 0
                shares = abs(tx.get('change') or tx.get('share') or 0)
                date_v = tx.get('transactionDate') or tx.get('filingDate') or ''

                # Only open-market buys (P) and sales (S); skip grants, exercises, gifts
                if code not in ('P', 'S'):
                    continue
                if price == 0 or shares == 0:
                    continue

                try:
                    date_str = datetime.strptime(date_v[:10], '%Y-%m-%d').strftime('%b %d')
                except (ValueError, TypeError):
                    date_str = str(date_v)[:10]

                w = _value_weight(price, shares)
                if code == 'P':
                    buy_score += w
                    distinct_buyers.add(name)
                    buy_details.append({
                        'insider':  name,
                        'position': '',
                        'date':     date_str,
                        'value':    round(float(price) * float(shares), 0),
                        'shares':   int(shares),
                        'weight':   round(w, 1),
                    })
                else:
                    sell_score += w

            row['InsiderBuys']      = buy_details
            row['InsiderBuyScore']  = round(buy_score, 1)
            row['InsiderSellScore'] = round(sell_score, 1)
            row['InsiderCluster']   = len(distinct_buyers) >= 3

            if buy_score == 0 and sell_score == 0:
                signal = 'Neutral'
            elif buy_score >= sell_score * 1.5:
                signal = 'Cluster Buy' if len(distinct_buyers) >= 3 else ('Strong Buy' if buy_score >= 4.0 else 'Buying')
            elif sell_score > buy_score * 1.5:
                signal = 'Selling'
            else:
                signal = 'Neutral'

            row['InsiderNet']    = round(buy_score - sell_score, 1)
            row['InsiderSignal'] = signal
        except Exception:
            pass

    except Exception:
        pass
    return row


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_extras(tickers):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _empty = lambda t: {"Ticker": t, "InsiderSignal": "N/A", "InsiderNet": 0,
                        "InsiderBuyScore": 0.0, "InsiderSellScore": 0.0,
                        "InsiderCluster": False, "Headlines": [], "InsiderBuys": []}
    results_map = {}
    fh_client = _make_finnhub_client()   # None if key missing or package absent

    def _dispatch(t):
        if fh_client is not None:
            return _fetch_single_extra_finnhub(t, fh_client)
        return _fetch_single_extra(t)

    pool = ThreadPoolExecutor(max_workers=4)
    futures = {pool.submit(_dispatch, t): t for t in tickers}
    try:
        for future in as_completed(futures, timeout=45):
            t = futures[future]
            try:
                results_map[t] = future.result(timeout=12)
            except Exception:
                results_map[t] = _empty(t)
    except TimeoutError:
        for t in tickers:
            if t not in results_map:
                results_map[t] = _empty(t)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return pd.DataFrame([results_map.get(t, _empty(t)) for t in tickers])


# ── StockTwits Sentiment ─────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stocktwits(tickers):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(t):
        try:
            r = requests.get(
                f"https://api.stocktwits.com/api/2/streams/symbol/{t}.json",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=4,
            )
            if r.status_code != 200:
                return t, None
            msgs = r.json().get("messages", [])
            bulls = sum(1 for m in msgs if
                        (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bullish")
            bears = sum(1 for m in msgs if
                        (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bearish")
            total = bulls + bears
            return t, {
                "bull_pct": round(bulls / total * 100) if total else None,
                "msg_count": len(msgs),
            }
        except (requests.RequestException, requests.exceptions.Timeout, ValueError, KeyError):
            return t, None

    out = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for f in as_completed(futures):
            t, result = f.result()
            if result is not None:
                out[t] = result
    return out


# ── Scanner sources ──────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def scan_yahoo_screener(min_mcap=200_000_000, max_mcap=10_000_000_000, min_rev_growth=0.10):
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
    except (requests.RequestException, requests.exceptions.Timeout, ValueError, KeyError):
        return []


@st.cache_data(ttl=600, show_spinner=False)
def scan_yahoo_predefined(screener_key="most_actives"):
    try:
        result = yf.screen(yf.PREDEFINED_SCREENER_QUERIES[screener_key])
        if result and "quotes" in result:
            return [q["symbol"] for q in result["quotes"] if "." not in q.get("symbol", ".")]
        return []
    except (requests.RequestException, requests.exceptions.Timeout, ValueError, KeyError):
        return []


@st.cache_data(ttl=600, show_spinner=False)
def scan_finviz():
    try:
        import re
        url = ("https://finviz.com/screener.ashx?v=111"
               "&f=cap_smallover,fa_salesqoq_o10,sh_avgvol_o200"
               "&ft=4&o=-marketcap")
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        matches = re.findall(r'quote\.ashx\?t=([A-Z]{1,5})&', resp.text)
        seen = set()
        tickers = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                tickers.append(m)
        return tickers[:100]
    except (requests.RequestException, requests.exceptions.Timeout, ValueError, KeyError):
        return []


@st.cache_data(ttl=600, show_spinner=False)
def scan_yahoo_trending():
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US?count=50"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        return [q["symbol"] for q in quotes if "." not in q.get("symbol", "")]
    except (requests.RequestException, requests.exceptions.Timeout, ValueError, KeyError):
        return []


def score_scanner_candidates(tickers, spy_close, spy_daily_ret):
    """Lightweight scoring pipeline for scanner candidates."""
    from scoring import calc_setup_stage, calc_convexity_score, calc_four_pillars

    if not tickers:
        return pd.DataFrame()

    df_price = fetch_price_data(tickers)
    if df_price.empty:
        return pd.DataFrame()

    df_fund = fetch_fundamentals(tickers)
    df = df_price.merge(df_fund, on="Ticker", how="left")

    scores = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        stage = calc_setup_stage(rd)
        # Use pillars for scanner scoring too
        pillars = calc_four_pillars(rd, {"themes": {}})
        convexity = calc_convexity_score(
            pillars["technical"], pillars["fundamental"],
            pillars["thematic"], pillars["narrative"]
        )
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
            "Convexity": convexity,
            "SetupStage": stage,
        })
    df_scored = pd.DataFrame(scores)

    if not spy_close.empty and not spy_daily_ret.empty:
        df_rs = calc_downday_rs(tickers, spy_close, spy_daily_ret)
        if not df_rs.empty:
            df_scored = df_scored.merge(
                df_rs[["Ticker", "RS_Score", "RS_Label"]],
                on="Ticker", how="left"
            )

    if "RS_Score" not in df_scored.columns:
        df_scored["RS_Score"] = None
        df_scored["RS_Label"] = None
    df_scored["ScanScore"] = df_scored.apply(
        lambda r: round(
            0.50 * _safe(r.get("Convexity"), 0) +
            0.50 * _safe(r.get("RS_Score"), 0), 1
        ), axis=1
    )
    df_scored = df_scored.sort_values("ScanScore", ascending=False).reset_index(drop=True)
    return df_scored


# ── Market Headlines (Google News RSS) ────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_headlines():
    """Fetch top macro/market headlines from Google News RSS. Free, no API key."""
    from bs4 import BeautifulSoup
    headlines = []
    queries = ["stock+market", "S%26P+500", "Federal+Reserve", "economy"]
    seen_titles = set()
    for q in queries:
        try:
            r = requests.get(
                f"https://news.google.com/rss/search?q={q}+when:1d&hl=en-US&gl=US&ceid=US:en",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
            )
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, "xml")
            for item in soup.find_all("item")[:5]:
                title = item.find("title")
                pub_date = item.find("pubDate")
                source = item.find("source")
                t = title.text.strip() if title else ""
                if not t or t in seen_titles:
                    continue
                seen_titles.add(t)
                headlines.append({
                    "title": t,
                    "source": source.text.strip() if source else "",
                    "date": pub_date.text.strip() if pub_date else "",
                })
        except (AttributeError, TypeError, ValueError):
            continue
    return headlines[:15]


@st.cache_data(ttl=900, show_spinner=False)
def ai_market_summary(headlines, api_key):
    """Generate a concise AI market conditions summary from headlines.
    Accepts list of dicts or tuple of (title, source) pairs for cache hashability.
    """
    if not api_key or not headlines:
        return None

    lines = []
    for h in headlines[:15]:
        if isinstance(h, dict):
            lines.append(f"- {h['title']} ({h['source']})")
        else:
            lines.append(f"- {h[0]} ({h[1]})")
    news_block = "\n".join(lines)

    prompt = f"""You are a market analyst. Based on these current headlines, provide:

1. **MARKET READ** (2-3 sentences): What is the dominant narrative driving markets right now? What's the tone — risk-on, risk-off, mixed? Be direct and specific.

2. **KEY CATALYSTS THIS WEEK**: List 3-5 specific events, data releases, or developments to watch in the next few days. Include dates if known. Focus on what could move markets.

3. **WATCH FOR**: One sentence on the biggest risk or surprise scenario that isn't priced in.

Headlines:
{news_block}

Be concise and actionable. No filler. Write for someone managing a small-cap growth portfolio."""

    return _xai_chat(api_key, [{"role": "user", "content": prompt}],
                     model="grok-3-mini", max_tokens=400)


# ── AI Headline Sentiment (lightweight, feeds Narrative pillar) ───────────────

@st.cache_data(ttl=1800, show_spinner=False)
def score_headlines_ai(tickers_and_headlines, api_key):
    """Score headline sentiment for multiple tickers in a single Claude call.
    Accepts dict or tuple of (ticker, headlines) pairs (tuple for Streamlit cache).
    Returns {ticker: {"score": -1 to 1, "summary": "one line"}} .
    Score: -1 (very bearish) to +1 (very bullish), 0 = neutral/mixed.
    """
    if not api_key or not tickers_and_headlines:
        return {}

    # Convert tuple to iterable of (ticker, headlines) pairs
    items = tickers_and_headlines.items() if isinstance(tickers_and_headlines, dict) else tickers_and_headlines

    # Build a compact prompt with all tickers
    blocks = []
    tickers_with_news = []
    for ticker, headlines in items:
        if not headlines:
            continue
        titles = [h.get("title", "") for h in headlines[:5] if h.get("title")]
        if not titles:
            continue
        tickers_with_news.append(ticker)
        block = f"{ticker}:\n" + "\n".join(f"  - {t}" for t in titles)
        blocks.append(block)

    if not blocks:
        return {}

    prompt = f"""Score the headline sentiment for each ticker below. For each ticker, respond with ONLY a JSON line:
{{"ticker": "XXX", "score": 0.0, "summary": "one line"}}

Score range: -1.0 (very bearish) to +1.0 (very bullish). 0.0 = neutral/mixed.
Consider: Is the news positive for the stock price? Are there catalysts, risks, or neutral noise?
Be calibrated: most routine news is near 0. Only strong positive/negative warrants >0.5 or <-0.5.

Headlines:
{chr(10).join(blocks)}

Respond with one JSON line per ticker, nothing else."""

    text = _xai_chat(api_key, [{"role": "user", "content": prompt}],
                     model="grok-3-mini", max_tokens=500)
    if not text:
        return {}
    import json as _json
    results = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = _json.loads(line)
            t = obj.get("ticker", "")
            if t in tickers_with_news:
                results[t] = {
                    "score": max(-1.0, min(1.0, float(obj.get("score", 0)))),
                    "summary": obj.get("summary", ""),
                }
        except (ValueError, KeyError, TypeError):
            continue
    return results


# ── X Insights — per-ticker deep dive (Grok live search) ────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_x_insights(ticker, api_key):
    """Pull the most important insights from X posts about a ticker in the last 24h.
    Uses Grok live search. Returns formatted markdown string or None.
    Cached 30 min — called on-demand in deep dive, not at page load.
    """
    if not api_key:
        return None
    prompt = (
        f"Search X (Twitter) for posts about ${ticker} stock from the last 24 hours.\n\n"
        f"What are the most important things traders and investors are actually discussing? "
        f"Focus on: catalysts, earnings/guidance commentary, unusual activity, notable accounts, "
        f"any news being reacted to, and overall crowd read.\n\n"
        f"Respond with 3-5 concise bullet points. Be specific — cite what people are saying, "
        f"not generic observations. If volume is low or nothing notable, say so briefly. "
        f"Write for a sophisticated trader who wants signal, not noise."
    )
    result = _xai_responses(
        api_key, [{"role": "user", "content": prompt}],
        tools=[{"type": "x_search"}], model="grok-3", max_tokens=500,
    )
    if not result or result.startswith("[xAI"):
        result = _xai_chat(api_key, [{"role": "user", "content": prompt}],
                           model="grok-3", max_tokens=500)
    return result


# ── X Sentiment (Grok live search) ───────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_x_sentiment(tickers_tuple, api_key):
    """Get X/Twitter sentiment for each ticker via Grok live search.
    Returns {ticker: {"bull_pct": int, "summary": str}}.
    Runs one call per ticker in parallel; cached 30 min.
    """
    if not api_key or not tickers_tuple:
        return {}
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(t):
        prompt = (
            f"What is the current sentiment among traders and investors about ${t} stock "
            f"on X (Twitter) and social media? "
            f"Is opinion mostly bullish, bearish, or mixed? "
            f'Respond with ONLY this JSON (no other text): {{"ticker": "{t}", "bull_pct": <integer 0-100>, "summary": "<one sentence>"}}\n'
            f"bull_pct: 50=neutral, above 60=bullish, below 40=bearish."
        )
        text = _xai_chat(api_key, [{"role": "user", "content": prompt}],
                         model="grok-3-mini", max_tokens=80)
        if not text:
            return t, None
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = _json.loads(line)
                return t, {
                    "bull_pct": max(0, min(100, int(obj.get("bull_pct", 50)))),
                    "summary":  obj.get("summary", ""),
                }
            except (ValueError, KeyError, TypeError):
                continue
        return t, None

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers_tuple}
        try:
            for future in as_completed(futures, timeout=60):
                t, val = future.result(timeout=15)
                if val is not None:
                    results[t] = val
        except Exception:
            pass
    return results


# ── AI Narrative Synthesis ───────────────────────────────────────────────────

def grok_summarize(ticker, price, rsi, atr_pct, pos52, vs50, vs200,
                   short_pct, beta, upside, recommendation, mcap,
                   rev_growth, insider_signal, earnings_beats,
                   next_earnings, x_sentiment, headlines, api_key):
    """Synthesize all available data into a narrative using Grok (xAI).
    Grok searches X and the web for live context on top of the data provided.
    """
    news_block = "\n".join(
        f"  - {h.get('date','')} {h.get('title','')} ({h.get('pub','')})"
        for h in (headlines or [])[:5]
    ) or "  None available"

    _x = x_sentiment or {}
    x_bull = _x.get("bull_pct")
    x_summary = _x.get("summary", "")
    sentiment_line = (
        f"{x_bull}% bullish — {x_summary}" if x_bull is not None
        else "N/A"
    )

    prompt = f"""You are a concise investment analyst. Synthesize the data below for ${ticker} into a clear investment summary for someone who favours asymmetric, high-beta re-rating opportunities.

TECHNICAL:
- Price: ${price} | RSI: {rsi} | 52wk position: {pos52:.0f}% | ATR%: {atr_pct}%
- vs MA50: {vs50:+.1f}% | vs MA200: {(str(vs200) + '%') if vs200 else 'N/A'}

FUNDAMENTAL:
- Market cap: {mcap} | Beta: {beta or 'N/A'} | Short interest: {short_pct or 'N/A'}%
- Revenue growth: {rev_growth or 'N/A'}% | Analyst consensus: {recommendation or 'N/A'}
- Analyst upside to mean target: {upside or 'N/A'}% | Next earnings: {next_earnings or 'N/A'}
- EPS beats last 4 quarters: {earnings_beats or 'N/A'} | Insider activity (90d): {insider_signal or 'N/A'}

X (TWITTER) SENTIMENT:
- {sentiment_line}

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

Be direct. No filler."""

    result = _xai_responses(
        api_key, [{"role": "user", "content": prompt}],
        tools=[{"type": "web_search"}, {"type": "x_search"}], model="grok-3", max_tokens=700,
    )
    if not result or result.startswith("[xAI"):
        result = _xai_chat(api_key, [{"role": "user", "content": prompt}],
                           model="grok-3", max_tokens=700)
    return result or "[xAI error] Empty response."


# ── ETF Peer Comparison ───────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)  # Cache 24h — peer baskets are stable
def fetch_etf_peer_ev_sales(etf_ticker, fmp_key, max_holdings=20):
    """
    Fetch top holdings of an ETF and their EV/Sales ratios.
    Uses FMP /etf-holder endpoint + yfinance for EV/Sales.
    Returns list of (ticker, ev_sales) tuples, sorted by ev_sales.
    """
    holdings = []
    try:
        url = f"https://financialmodelingprep.com/stable/etf-holder?symbol={etf_ticker}&apikey={fmp_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # FMP returns list sorted by weight — take top N
            for h in data[:max_holdings]:
                t = h.get("asset") or h.get("symbol") or h.get("ticker")
                if t and isinstance(t, str) and len(t) <= 5:
                    holdings.append(t.upper())
    except (requests.RequestException, requests.exceptions.Timeout, ValueError, KeyError):
        pass

    if not holdings:
        return []

    # Batch-fetch EV/Sales for all holdings
    results = []
    try:
        objs = {t: yf.Ticker(t) for t in holdings}
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _get_ev_sales(t):
            try:
                info = objs[t].info
                ev = info.get("enterpriseValue")
                rev = info.get("totalRevenue")
                if ev and rev and rev > 0:
                    return (t, round(ev / rev, 2))
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            return (t, None)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_get_ev_sales, t): t for t in holdings}
            for future in as_completed(futures, timeout=30):
                try:
                    ticker, ev_sales = future.result(timeout=10)
                    if ev_sales is not None:
                        results.append((ticker, ev_sales))
                except (TimeoutError, Exception):
                    pass
    except Exception:
        pass

    return sorted(results, key=lambda x: x[1])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_peer_comparison(ticker_etf_pairs, fmp_key):
    """
    For each ticker, find its primary theme ETF, fetch peer EV/Sales,
    and return percentile rank of each ticker within its peer group.

    ticker_etf_pairs: tuple of (ticker, etf) pairs — hashable for st.cache_data
    Returns: dict {ticker: {"peer_rank_pct": float, "etf": str,
                             "peer_count": int, "peer_median": float}}
    """
    ticker_to_etf = dict(ticker_etf_pairs)
    # Get unique ETFs needed
    etf_set = set(v for v in ticker_to_etf.values() if v)
    if not etf_set or not fmp_key:
        return {}

    # Fetch peer data per ETF (cached per ETF)
    etf_peers = {}
    for etf in etf_set:
        peers = fetch_etf_peer_ev_sales(etf, fmp_key)
        if peers:
            etf_peers[etf] = peers  # [(ticker, ev_sales), ...]

    results = {}
    for ticker, etf in ticker_to_etf.items():
        if not etf or etf not in etf_peers:
            continue
        peers = etf_peers[etf]
        if len(peers) < 3:
            continue
        ev_sales_values = [ev for _, ev in peers]
        median_val = float(pd.Series(ev_sales_values).median())

        # Find where this ticker's EV/Sales sits in peer distribution
        # First check if ticker is already in the peer list
        ticker_ev = next((ev for t, ev in peers if t == ticker), None)

        if ticker_ev is not None:
            rank_pct = round(
                sum(1 for ev in ev_sales_values if ev <= ticker_ev) / len(ev_sales_values) * 100, 1
            )
            results[ticker] = {
                "peer_rank_pct": rank_pct,
                "etf": etf,
                "peer_count": len(peers),
                "peer_median": median_val,
                "peer_ev_sales": ticker_ev,
            }
        # If ticker not in ETF holdings, we'll use EV/Sales from df_all
        # and rank against the peer distribution — handled in portfolio_app.py

    return results
