"""
Convexity Terminal — Data Fetching

All external data retrieval lives here: yfinance price data, fundamentals,
market environment, relative strength, ETF benchmarks, extras (news/insider),
StockTwits sentiment, scanner sources, and AI narrative synthesis.
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

from scoring import calc_rsi, calc_atr_pct, _safe, SECTOR_ETFS
from themes import _get_anthropic_key

# ── File paths ───────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
FUND_CACHE_FILE = os.path.join(_DIR, "fund_cache.json")


# ── Fund cache helpers ───────────────────────────────────────────────────────

def _load_fund_cache():
    """Load cached fundamental data from disk. Stale data > no data."""
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


# ── Price data ───────────────────────────────────────────────────────────────

_price_fetch_error = ""  # Last error reason, readable by portfolio_app.py


def fetch_price_data(tickers):
    global _price_fetch_error
    import time
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
        except Exception as e:
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
                except Exception:
                    pass
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
            ret_1m  = round(((price / close.iloc[-22]) - 1) * 100, 1) if len(close) >= 22 else None
            ret_3m  = round(((price / close.iloc[-63]) - 1) * 100, 1) if len(close) >= 63 else None
            ret_6m  = round(((price / close.iloc[-126]) - 1) * 100, 1) if len(close) >= 126 else None
            spark_30 = close.iloc[-30:].tolist() if len(close) >= 30 else close.tolist()
            results.append(dict(
                Ticker=t, Price=round(price, 2),
                RSI=round(rsi, 1), vsMA50=round(vs50, 1),
                vsMA200=round(vs200, 1) if not np.isnan(vs200) else None,
                ATR_pct=round(atr_pct, 1), Pos52=round(pos52, 1),
                Low52=round(w52l, 2), High52=round(w52h, 2),
                RelVol=rel_vol, Breakout=bool(pos52 >= 95),
                Ret1m=ret_1m, Ret3m=ret_3m, Ret6m=ret_6m,
                Spark30=spark_30,
            ))
        except Exception:
            continue
    return pd.DataFrame(results)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_spy_daily():
    """Fetch SPY 1y daily close & daily returns for down-day RS analysis."""
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
            except Exception:
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
                except Exception:
                    pass
        except Exception:
            pass
        env["sectors"] = sector_data
        env["sectors_above_50d"] = sum(1 for s in sector_data.values() if s["above_50d"])
        env["sectors_positive_1d"] = sum(1 for s in sector_data.values() if s["ret_1d"] > 0)
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

    except Exception:
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
    except Exception:
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
        except Exception:
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
    except Exception:
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
                except Exception:
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
        except Exception:
            continue
    return pd.DataFrame(results)


# ── Fundamentals ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_single_fundamental(t, _disk_cache):
    """Fetch fundamental data for a single ticker. Used by ThreadPoolExecutor."""
    row = {"Ticker": t}
    _rate_limited = False
    try:
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

        ps = info.get("priceToSalesTrailing12Months")
        ev = info.get("enterpriseToEbitda")
        gm = info.get("grossMargins")
        row["PS_Current"] = round(ps, 2) if ps else None
        row["EV_EBITDA"] = round(ev, 1) if ev and ev > 0 else None
        row["GrossMargin"] = round(gm * 100, 1) if gm else None

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
                    ttm = rev_q.rolling(4).sum().dropna()
                    if not ttm.empty:
                        ttm.index = ttm.index.tz_localize(None) if ttm.index.tz else ttm.index
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

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_single_fundamental, t, _disk_cache): t for t in tickers}
        results_map = {}
        for future in as_completed(futures, timeout=30):
            t = futures[future]
            try:
                row, cache_entry, rl = future.result(timeout=15)
                results_map[t] = row
                if cache_entry:
                    _new_cache[t] = cache_entry
                if rl:
                    _rate_limited = True
            except Exception:
                results_map[t] = {"Ticker": t}

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
                        except Exception:
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
        except Exception:
            pass
        # EarningsBeats removed — minor signal, heavy API call per ticker
    except Exception:
        pass
    return row


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_extras(tickers):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results_map = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_single_extra, t): t for t in tickers}
        for future in as_completed(futures, timeout=30):
            t = futures[future]
            try:
                results_map[t] = future.result(timeout=15)
            except Exception:
                results_map[t] = {"Ticker": t, "InsiderSignal": "N/A", "InsiderNet": 0,
                                  "InsiderBuyScore": 0.0, "InsiderSellScore": 0.0,
                                  "InsiderCluster": False, "Headlines": [], "InsiderBuys": []}
    return pd.DataFrame([results_map.get(t, {"Ticker": t}) for t in tickers])


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
        except Exception:
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
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def scan_yahoo_predefined(screener_key="most_actives"):
    try:
        result = yf.screen(yf.PREDEFINED_SCREENER_QUERIES[screener_key])
        if result and "quotes" in result:
            return [q["symbol"] for q in result["quotes"] if "." not in q.get("symbol", ".")]
        return []
    except Exception:
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
    except Exception:
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
    except Exception:
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
        except Exception:
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

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
        return None
    except Exception:
        return None


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

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return {}
        text = resp.json()["content"][0]["text"]
        import re
        results = {}
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                import json as _json
                obj = _json.loads(line)
                t = obj.get("ticker", "")
                if t in tickers_with_news:
                    results[t] = {
                        "score": max(-1.0, min(1.0, float(obj.get("score", 0)))),
                        "summary": obj.get("summary", ""),
                    }
            except Exception:
                continue
        return results
    except Exception:
        return {}


# ── AI Narrative Synthesis ───────────────────────────────────────────────────

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
                "model": "claude-haiku-4-5",
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
