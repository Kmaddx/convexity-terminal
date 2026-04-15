"""
Convexity Terminal — Scoring Engine

All scoring logic lives here. The four pillars are the single spine:
  Technical  — price action, trend, relative strength
  Fundamental — quality, growth, valuation, ownership
  Thematic   — theme alignment, ETF momentum, sector rotation
  Narrative  — sentiment, insider conviction, catalyst proximity

Convexity Score is derived FROM the pillars, not computed in parallel.
Setup Stage classifies swing-trade timing from technical data.
Market Environment scores the macro backdrop.
"""

import numpy as np
import pandas as pd

# ── Scoring Thresholds (tunable config) ───────────────────────────────────────
#    All magic numbers extracted here for easy experimentation and visibility.

THRESHOLDS = {
    # Technical pillar — price action & momentum
    "rsi_oversold": 35,
    "rsi_overbought": 70,
    "rsi_strong_sell": 30,
    "rsi_strong_buy": 55,
    "rsi_mid_low": 40,
    "rsi_mid_high": 60,
    "pos52_pullback": 40,  # < this = pullback stage
    "pos52_strength": 50,   # > this = showing strength
    "pos52_very_strong": 70,
    "pos52_extremely_strong": 90,
    "vs50_strong_down": -10,
    "vs50_near_ma": 5,
    "vs50_breakout": 10,
    "vs50_strong_breakout": 25,
    "vs200_breakout": 0,
    "dd_win_strong": 60,
    "dd_win_moderate": 45,
    "dd_excess_positive": 0,
    "dd_excess_weak": -0.3,

    # Fundamental pillar — quality & growth
    "rev_growth_high": 30,   # %
    "rev_growth_medium": 15,
    "gross_margin_high": 60,  # %
    "gross_margin_good": 40,
    "gross_margin_ok": 20,
    "rule40_strong": 40,
    "rule40_medium": 20,
    "insider_ownership_high": 15,  # %
    "insider_ownership_medium": 8,
    "insider_ownership_low": 3,
    "analyst_base_conf": 0.4,
    "analyst_conf_per_analyst": 0.2,
    "ps_historical_25th": 25,
    "ps_historical_50th": 50,
    "ps_historical_75th": 75,

    # Thematic pillar — sector rotation
    "etf_rs_positive": 0,
    "short_interest_high": 10,  # %
    "short_interest_medium": 5,

    # Narrative pillar — sentiment & catalysts
    # (no thresholds here; mostly continuous scoring)

    # Four-pillar alignment
    "alignment_threshold": 50,

    # Convexity weighting
    "convexity_tech_weight": 0.30,
    "convexity_fundamental_weight": 0.30,
    "convexity_thematic_weight": 0.20,
    "convexity_narrative_weight": 0.20,

    # Market environment
    "vix_high": 25,
    "vix_low": 12,
    "spx_above_200ma": True,  # qualitative
}

# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Technical helpers ────────────────────────────────────────────────────────

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


# ── Setup Stage ──────────────────────────────────────────────────────────────

def calc_setup_stage(row):
    """Classify ticker into a swing-trading setup stage."""
    rsi   = _safe(row.get("RSI"), 50)
    pos52 = _safe(row.get("Pos52"), 50)
    vs50  = _safe(row.get("vsMA50"), 0)
    # Breaking Down — below support, thesis at risk
    if rsi < THRESHOLDS["rsi_oversold"] and vs50 < THRESHOLDS["vs50_strong_down"]:
        return "Breaking Down"
    # Extended — far from support, risky entry
    if rsi > THRESHOLDS["rsi_overbought"] or pos52 > THRESHOLDS["pos52_extremely_strong"] or vs50 > THRESHOLDS["vs50_strong_breakout"]:
        return "Extended"
    # Basing — consolidating near support
    if THRESHOLDS["rsi_oversold"] <= rsi <= 50 and pos52 < THRESHOLDS["pos52_pullback"] and abs(vs50) <= THRESHOLDS["vs50_near_ma"]:
        return "Basing"
    # Emerging — starting to move
    if 50 <= rsi <= THRESHOLDS["rsi_mid_high"] and THRESHOLDS["pos52_pullback"] <= pos52 <= THRESHOLDS["pos52_strength"] and 0 < vs50 < THRESHOLDS["vs50_breakout"]:
        return "Emerging"
    # Trending — established uptrend
    if THRESHOLDS["rsi_strong_buy"] <= rsi <= THRESHOLDS["rsi_overbought"] and pos52 > THRESHOLDS["pos52_strength"] and vs50 > THRESHOLDS["vs50_breakout"]:
        return "Trending"
    return "Neutral"


# ── Four Pillars (the single scoring spine) ──────────────────────────────────

def calc_four_pillars(row, themes, spy_ret=None, etf_data=None, st_data=None, ai_sentiment=None):
    """Score each of the 4 pillars 0-100. Returns dict with scores + alignment.

    The four pillars absorb ALL sub-scores:
    - Technical absorbs: RSI, trend (MA50/MA200), position, relative strength, momentum
    - Fundamental absorbs: growth, margins, FCF, valuation (P/S), ownership, analyst targets
    - Thematic absorbs: theme alignment, ETF benchmark momentum, sector rotation
    - Narrative absorbs: sentiment, insider buying, earnings beats, catalyst proximity
    """
    from themes import get_ticker_themes, get_theme_etfs

    # ── Technical (0-100) ──
    # Absorbs old momentum score components + RS signals
    tech = 0.0
    rsi = _safe(row.get("RSI"), 50)

    # RSI positioning: 25 pts — sweet spot 40-65, peak at 52
    if THRESHOLDS["rsi_mid_low"] <= rsi <= 65:
        tech += 25 * (1 - abs(rsi - 52.5) / 12.5)
    elif THRESHOLDS["rsi_strong_sell"] <= rsi < THRESHOLDS["rsi_mid_low"]:
        tech += 25 * (rsi - THRESHOLDS["rsi_strong_sell"]) / 10 * 0.5
    elif rsi < THRESHOLDS["rsi_strong_sell"]:
        tech += 5  # oversold — potential but risky

    # Price vs MA50: 20 pts — above = trending
    vs50 = _safe(row.get("vsMA50"), 0)
    if vs50 > 0:
        tech += min(vs50 / 20, 1.0) * 20

    # 52wk position: 20 pts — sweet spot 30-70%, peak at 50
    pos52 = _safe(row.get("Pos52"), 50)
    if THRESHOLDS["pos52_pullback"] <= pos52 <= THRESHOLDS["pos52_very_strong"]:
        tech += 20 * (1 - abs(pos52 - 50) / 20)
    elif pos52 > THRESHOLDS["pos52_very_strong"]:
        tech += 20 * max(0, 1 - (pos52 - THRESHOLDS["pos52_very_strong"]) / 30) * 0.5

    # Trend consistency: vs MA50 AND MA200: 15 pts
    vs200 = _safe(row.get("vsMA200"), 0)
    if vs50 > THRESHOLDS["vs200_breakout"] and vs200 > THRESHOLDS["vs200_breakout"]:
        tech += 15
    elif vs50 > THRESHOLDS["vs200_breakout"]:
        tech += 8

    # Down-day relative strength (from ZA framework): 20 pts
    # "What isn't going lower?" — institutional accumulation signal
    dd_win = _safe(row.get("DownDayWinRate"), 50)
    recent_dd = _safe(row.get("RecentDDExcess"), 0)
    if dd_win > THRESHOLDS["dd_win_strong"]:
        tech += min((dd_win - 40) / 30, 1.0) * 12
    elif dd_win > THRESHOLDS["dd_win_moderate"]:
        tech += 6
    if recent_dd > THRESHOLDS["dd_excess_positive"]:
        tech += min(recent_dd / 0.5, 1.0) * 8
    elif recent_dd > THRESHOLDS["dd_excess_weak"]:
        tech += 3

    # ── Fundamental (0-100) ──
    # Designed for small-cap/micro-cap growth companies where FCF is often negative
    # and Rule of 40 is rarely achieved. A decent growth stock should score ~45-55.
    fund = 0.0

    # Baseline: 8 pts — the company exists and has financials
    fund += 8

    # Revenue growth: up to 22 pts (the key metric for growth-stage companies)
    rev_g = _safe(row.get("RevGrowthPct"), 0)
    if rev_g > THRESHOLDS["rev_growth_high"]:
        fund += 22
    elif rev_g > THRESHOLDS["rev_growth_medium"]:
        fund += 12 + (rev_g - THRESHOLDS["rev_growth_medium"]) / 15 * 10
    elif rev_g > THRESHOLDS["rev_growth_medium"] - THRESHOLDS["rev_growth_medium"]:
        fund += rev_g / THRESHOLDS["rev_growth_medium"] * 12
    else:
        fund += 3  # negative growth isn't great but isn't the only metric

    # Gross margin > 40%: up to 10 pts — shows business quality
    gm = _safe(row.get("GrossMargin"), 0)
    if gm > THRESHOLDS["gross_margin_high"]:
        fund += 10
    elif gm > THRESHOLDS["gross_margin_good"]:
        fund += 5 + (gm - THRESHOLDS["gross_margin_good"]) / 20 * 5
    elif gm > THRESHOLDS["gross_margin_ok"]:
        fund += (gm - THRESHOLDS["gross_margin_ok"]) / 20 * 5
    else:
        fund += 2  # hardware/manufacturing, not penalized harshly

    # FCF: 12 pts positive, 4 pts baseline if negative (growth companies pre-profit)
    if row.get("FCFPositive"):
        fund += 12
    else:
        fund += 4  # pre-profit growth is normal, not a failure

    # Insider ownership: up to 12 pts — skin in the game, founder alignment
    insider_pct = _safe(row.get("InsiderPct"), 0)
    if insider_pct >= THRESHOLDS["insider_ownership_high"]:
        fund += 12  # founder-level ownership
    elif insider_pct >= THRESHOLDS["insider_ownership_medium"]:
        fund += 9   # strong insider alignment
    elif insider_pct >= THRESHOLDS["insider_ownership_low"]:
        fund += 6   # moderate
    elif insider_pct > 0:
        fund += 3   # minimal
    else:
        fund += 1   # no data

    # Analyst upside: up to 10 pts (confidence-weighted by coverage breadth)
    upside = _safe(row.get("AnalystUpside"), 0)
    num_analysts = int(_safe(row.get("NumAnalysts"), 0))
    analyst_conf = 0.0 if num_analysts == 0 else min(1.0, THRESHOLDS["analyst_base_conf"] + THRESHOLDS["analyst_conf_per_analyst"] * num_analysts)
    if upside > 0:
        fund += min(upside / 80, 1.0) * 10 * analyst_conf
    elif num_analysts == 0:
        fund += 3  # uncovered — not penalized

    # Rule of 40: up to 10 pts (bonus for companies that achieve it)
    rule40 = _safe(row.get("Rule40"), 0)
    if rule40 > THRESHOLDS["rule40_strong"]:
        fund += 10
    elif rule40 > THRESHOLDS["rule40_medium"]:
        fund += (rule40 - THRESHOLDS["rule40_medium"]) / 20 * 7
    elif rule40 > 0:
        fund += 3

    # EV/Sales valuation positioning: up to 8 pts
    # Primary: EV/Sales vs own 3yr history (accounts for debt, better than P/S)
    # Secondary: vs sector peers (passed in as evs_sector_pct — percentile within sector)
    evs_pos = _safe(row.get("EVS_HistPos"))
    evs_sector_pct = _safe(row.get("EVS_SectorPct"))  # computed in portfolio_app after data load
    if evs_pos is not None and evs_pos >= 0:
        if evs_pos <= THRESHOLDS["ps_historical_25th"]:
            fund += 8  # historically cheap on EV/Sales
        elif evs_pos <= THRESHOLDS["ps_historical_50th"]:
            fund += 5
        elif evs_pos <= THRESHOLDS["ps_historical_75th"]:
            fund += 2
    elif evs_sector_pct is not None:
        # Fallback: use sector-relative percentile (lower = cheaper vs peers)
        if evs_sector_pct <= THRESHOLDS["ps_historical_25th"]:
            fund += 7  # cheapest quartile vs sector peers
        elif evs_sector_pct <= THRESHOLDS["ps_historical_50th"]:
            fund += 4
        elif evs_sector_pct <= THRESHOLDS["ps_historical_75th"]:
            fund += 1
    else:
        fund += 3  # no data, neutral

    # Beta / volatility bonus: up to 8 pts — higher beta = more convexity potential
    beta = _safe(row.get("Beta"))
    beta_reliable = row.get("BetaReliable", True)
    if not beta_reliable:
        beta = 1.5  # fallback for missing/unreliable
    if beta and beta > 0:
        fund += min(beta / 2.5, 1.0) * 8
    else:
        fund += 4

    # ── Thematic (0-100) ──
    thematic = 0.0
    ticker = row.get("Ticker", "")
    ticker_themes = get_ticker_themes(themes, ticker) if isinstance(themes, dict) and "themes" in themes else []

    # Belongs to at least one theme (not "Other"): 30 pts
    if ticker_themes:
        thematic += 30

    # ETF benchmark momentum boost: up to 25 pts (best across all themes)
    if etf_data is not None and not etf_data.empty and spy_ret and ticker_themes:
        theme_etfs_map = get_theme_etfs(themes) if isinstance(themes, dict) and "themes" in themes else {}
        _spy_3m_raw = spy_ret.get("3m", 0) if isinstance(spy_ret, dict) else 0
        spy_3m = float(_spy_3m_raw) if not isinstance(_spy_3m_raw, pd.Series) else 0
        best_etf_boost = 0
        for th in ticker_themes:
            theme_etf_list = theme_etfs_map.get(th, [])
            etf_sub = etf_data[etf_data["Ticker"].isin(theme_etf_list)]
            if not etf_sub.empty and etf_sub["Ret3m"].notna().any():
                _ret3m_vals = pd.to_numeric(etf_sub["Ret3m"], errors="coerce").dropna()
                etf_avg_3m = float(_ret3m_vals.mean()) if len(_ret3m_vals) > 0 else 0
                etf_rs = float(etf_avg_3m - spy_3m)
                if etf_rs > 0:
                    candidate = min(etf_rs / 30, 1.0) * 25
                    if candidate > best_etf_boost:
                        best_etf_boost = candidate
        thematic += best_etf_boost

    # Ticker 3m return vs SPY: up to 15 pts
    if spy_ret:
        ret3m = float(_safe(row.get("Ret3m"), 0))
        _spy_3m_raw = spy_ret.get("3m", 0) if isinstance(spy_ret, dict) else 0
        spy_3m = float(_spy_3m_raw) if not isinstance(_spy_3m_raw, pd.Series) else 0
        theme_rs = ret3m - spy_3m
        if theme_rs > 0:
            thematic += min(theme_rs / 30, 1.0) * 15

    # Theme breadth signal: 15 pts if ticker itself is trending
    if _safe(row.get("Pos52"), 0) > 50 and _safe(row.get("vsMA50"), 0) > 0:
        thematic += 15

    # Short interest squeeze potential: up to 15 pts
    short = _safe(row.get("ShortPct"))
    if short and short > THRESHOLDS["short_interest_high"]:
        thematic += min((short - 5) / 15, 1.0) * 15
    elif short and short > THRESHOLDS["short_interest_medium"]:
        thematic += 5

    # ── Narrative (0-100) ──
    # Captures the STORY: sentiment, attention, insider conviction, theme heat,
    # catalyst proximity, and track record. Distinct from Technical/Fundamental/Thematic.
    # Designed so a stock with no exceptional signals still scores ~40-50.
    narr = 0.0

    # Baseline: 15 pts — you picked this stock for a reason, it has a thesis
    narr += 15

    # Social sentiment (StockTwits): up to 10 pts
    _st_sentiment = (st_data or {}).get(ticker, {})
    bull_pct = _st_sentiment.get("bull_pct")
    msg_count = _st_sentiment.get("msg_count", 0)
    if bull_pct is not None:
        if bull_pct >= 60:
            narr += 5 + min((bull_pct - 60) / 30, 1.0) * 5
        elif bull_pct < 40:
            narr += 3  # contrarian — extreme bearishness can mean opportunity
        else:
            narr += 5  # neutral sentiment, not a red flag
        if msg_count >= 10:
            narr += min(msg_count / 25, 1.0) * 2
    else:
        narr += 5  # no data = neutral, not penalized

    # AI headline sentiment: up to 15 pts (from Claude analysis of recent news)
    _ai_score = _safe(row.get("AI_HeadlineScore"))
    if _ai_score is not None and _ai_score != 0:
        # Score range: -1 to +1 → mapped to 0-15 pts (7.5 = neutral)
        narr += round(7.5 + _ai_score * 7.5, 1)
    else:
        narr += 7.5  # no AI data or neutral = baseline

    # Insider buying: up to 20 pts — weighted by role, value, and cluster
    insider_sig   = str(row.get("InsiderSignal", "")).lower()
    buy_score     = _safe(row.get("InsiderBuyScore"), 0)
    sell_score    = _safe(row.get("InsiderSellScore"), 0)
    cluster       = bool(row.get("InsiderCluster", False))
    if insider_sig == "cluster buy" or cluster:
        narr += 20          # 3+ distinct insiders buying — very strong signal
    elif insider_sig == "strong buy" or buy_score >= 4.0:
        narr += 17          # C-suite or large-value open market purchase
    elif insider_sig == "buying" or buy_score > 0:
        narr += 12          # Discretionary buying present
    elif insider_sig == "selling" and sell_score > buy_score * 1.5:
        narr += 2           # Net discretionary selling — mild negative
    else:
        narr += 5           # Neutral / no signal

    # Theme momentum: up to 15 pts — is the theme hot right now?
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
                narr += 6  # flat theme
            else:
                narr += 3  # theme is cold
        else:
            narr += 7  # no ETF data, neutral
    else:
        narr += 7  # untagged, neutral

    # Analyst coverage breadth: up to 10 pts
    if num_analysts >= 5:
        narr += 10
    elif num_analysts >= 3:
        narr += 8
    elif num_analysts >= 1:
        narr += 6
    else:
        narr += 4  # uncovered — could be undiscovered

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
        except (ValueError, TypeError):
            pass

    # ── Cap and return ──
    tech = min(round(tech, 1), 100)
    fund = min(round(fund, 1), 100)
    thematic = min(round(thematic, 1), 100)
    narr = min(round(narr, 1), 100)
    aligned = round(tech) >= THRESHOLDS["alignment_threshold"] and round(fund) >= THRESHOLDS["alignment_threshold"] and round(thematic) >= THRESHOLDS["alignment_threshold"] and round(narr) >= THRESHOLDS["alignment_threshold"]

    return {
        "technical": tech, "fundamental": fund,
        "thematic": thematic, "narrative": narr,
        "aligned": aligned,
    }


# ── Convexity Score (derived FROM pillars) ───────────────────────────────────

def calc_convexity_score(pillar_tech, pillar_fund, pillar_theme, pillar_narr):
    """Single output number derived from the four pillars.

    Convexity = asymmetric upside potential. Weighted to emphasise the pillars
    that most directly indicate payoff asymmetry:
    - Technical (30%) — timing and trend confirmation
    - Fundamental (30%) — quality and valuation support
    - Thematic (20%) — tailwind strength
    - Narrative (20%) — catalyst and conviction
    """
    score = (
        pillar_tech * THRESHOLDS["convexity_tech_weight"] +
        pillar_fund * THRESHOLDS["convexity_fundamental_weight"] +
        pillar_theme * THRESHOLDS["convexity_thematic_weight"] +
        pillar_narr * THRESHOLDS["convexity_narrative_weight"]
    )
    return round(score, 1)


# ── Market Environment ───────────────────────────────────────────────────────

SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLC", "XLRE", "XLB"]


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
    if vix_rising and vix > 20:
        vol_score = max(vol_score - 15, 0)
    if vix_pct > 75:
        vol_score = max(vol_score - 10, 0)
    pillars["Volatility"] = {"score": round(vol_score), "weight": 25}

    # ── 2. Trend (0-100, price vs MAs) ──
    spx_20 = env.get("spx_vs_20d", 0)
    spx_50 = env.get("spx_vs_50d", 0)
    spx_200 = env.get("spx_vs_200d", 0)
    trend_score = 0
    if spx_200 is not None:
        if spx_200 > 0:
            trend_score += min(spx_200, 5) / 5 * 30
        else:
            trend_score += max(0, 30 + spx_200 * 3)
    else:
        trend_score += 15
    if spx_50 > 0:
        trend_score += min(spx_50, 5) / 5 * 40
    else:
        trend_score += max(0, 40 + spx_50 * 4)
    if spx_20 > 0:
        trend_score += min(spx_20, 3) / 3 * 30
    else:
        trend_score += max(0, 30 + spx_20 * 5)
    trend_score = max(0, min(100, trend_score))
    qqq_50 = env.get("qqq_vs_50d", 0)
    if qqq_50 < -3:
        trend_score = max(trend_score - 10, 0)
    pillars["Trend"] = {"score": round(trend_score), "weight": 25}

    # ── 3. Breadth (0-100, sector participation) ──
    above_50d = env.get("sectors_above_50d")
    pos_1d = env.get("sectors_positive_1d")
    total_sectors = len(SECTOR_ETFS)
    if above_50d is None or pos_1d is None:
        breadth_score = 50  # neutral when sector data unavailable
    else:
        breadth_score = (above_50d / total_sectors) * 60
        breadth_score += (pos_1d / total_sectors) * 40
    pillars["Breadth"] = {"score": round(breadth_score), "weight": 20}

    # ── 4. Momentum (0-100, sector strength) ──
    sectors = env.get("sectors", {})
    if sectors:
        ret_5d_vals = [s["ret_5d"] for s in sectors.values()]
        avg_5d = np.mean(ret_5d_vals) if ret_5d_vals else 0
        positive_5d = sum(1 for r in ret_5d_vals if r > 0)
        mom_score = min(max((avg_5d + 3) / 6, 0), 1) * 50
        mom_score += (positive_5d / len(ret_5d_vals)) * 50 if ret_5d_vals else 25
    else:
        mom_score = 50
    pillars["Momentum"] = {"score": round(mom_score), "weight": 15}

    # ── 5. Macro (0-100, yield + dollar headwinds) ──
    macro_score = 50
    tnx = env.get("tnx_yield", 4.0)
    tnx_rising = env.get("tnx_rising", False)
    dxy_str = env.get("dxy_strengthening", False)
    if tnx < 3.5:
        macro_score += 25
    elif tnx < 4.0:
        macro_score += 15
    elif tnx > 4.5:
        macro_score -= 15
    elif tnx > 5.0:
        macro_score -= 25
    if tnx_rising:
        macro_score -= 10
    if dxy_str:
        macro_score -= 10
    macro_score = max(0, min(100, macro_score))
    pillars["Macro"] = {"score": round(macro_score), "weight": 15}

    # ── Warning flags (signs of a terrible market) ──
    warnings = []

    # Lower highs + lower lows = downtrend structure
    if env.get("lower_highs") and env.get("lower_lows"):
        warnings.append("Lower highs AND lower lows — downtrend structure")
        pillars["Trend"]["score"] = max(pillars["Trend"]["score"] - 10, 0)
    elif env.get("lower_highs"):
        warnings.append("Lower highs — resistance dropping")

    # Distribution days (institutional selling)
    dist_days = env.get("distribution_days", 0)
    if dist_days >= 6:
        warnings.append(f"Heavy distribution — {dist_days} sell-offs on high volume (25d)")
        pillars["Momentum"]["score"] = max(pillars["Momentum"]["score"] - 15, 0)
    elif dist_days >= 5:
        warnings.append(f"Increasing selling pressure — {dist_days} distribution days (25d)")
        pillars["Momentum"]["score"] = max(pillars["Momentum"]["score"] - 8, 0)

    # Rallies on low volume
    if env.get("low_vol_rallies"):
        ratio = env.get("up_down_vol_ratio", 0)
        warnings.append(f"Rallies on low volume — up/down vol ratio: {ratio:.2f}x")
        pillars["Momentum"]["score"] = max(pillars["Momentum"]["score"] - 8, 0)

    # Good opens, weak closes (fading)
    fade_days = env.get("fade_days_10d", 0)
    if fade_days >= 5:
        warnings.append(f"Weak closes — {fade_days}/10 sessions opened up but closed below open")
        pillars["Trend"]["score"] = max(pillars["Trend"]["score"] - 8, 0)

    # Extreme volatility
    if vix > 30:
        warnings.append(f"Extreme volatility — VIX at {vix:.0f}")
    elif vix > 25 and vix_rising:
        warnings.append(f"Volatility spiking — VIX at {vix:.0f} and rising")

    # Narrow breadth
    if above_50d <= 3:
        warnings.append(f"Narrow breadth — only {above_50d}/11 sectors above 50d MA")
    elif above_50d <= 5 and pos_1d <= 3:
        warnings.append(f"Breadth deteriorating — {above_50d}/11 above 50d, only {pos_1d}/11 positive today")

    # Below key MAs
    below_mas = []
    if spx_200 is not None and spx_200 < 0:
        below_mas.append("200d")
    if spx_50 < 0:
        below_mas.append("50d")
    if spx_20 < 0:
        below_mas.append("20d")
    if len(below_mas) >= 2:
        warnings.append(f"SPX below key MAs: {', '.join(below_mas)}")

    # ── Tailwinds (positive signals) ──
    tailwinds = []

    if env.get("lower_highs") is False and env.get("lower_lows") is False:
        tailwinds.append("Higher highs AND higher lows — uptrend structure intact")

    if dist_days <= 1:
        tailwinds.append("Minimal distribution — institutions not selling")

    if env.get("up_down_vol_ratio") and env.get("up_down_vol_ratio", 0) >= 1.3:
        ratio = env["up_down_vol_ratio"]
        tailwinds.append(f"Strong volume on up days — up/down vol ratio: {ratio:.2f}x")

    fade_days = env.get("fade_days_10d", 0)
    if fade_days <= 2:
        tailwinds.append("Strong closes — buyers holding into the bell")

    if vix <= 15:
        tailwinds.append(f"Low volatility — VIX at {vix:.0f}, risk appetite healthy")
    elif vix <= 20 and not vix_rising:
        tailwinds.append(f"Contained volatility — VIX at {vix:.0f} and stable")

    if above_50d >= 9:
        tailwinds.append(f"Broad participation — {above_50d}/11 sectors above 50d MA")

    above_mas = []
    if spx_200 is not None and spx_200 > 0:
        above_mas.append("200d")
    if spx_50 > 0:
        above_mas.append("50d")
    if spx_20 > 0:
        above_mas.append("20d")
    if len(above_mas) >= 3:
        tailwinds.append(f"SPX above all key MAs: {', '.join(above_mas)}")

    # ── Weighted total ──
    total_weight = sum(p["weight"] for p in pillars.values())
    total = sum(p["score"] * p["weight"] for p in pillars.values()) / total_weight
    total = round(total)

    if total >= 65:
        decision = "OPPORTUNITY"
        decision_sub = "Conditions favor adding to positions"
    elif total >= 45:
        decision = "SELECTIVE"
        decision_sub = "Pick your spots — quality setups only"
    else:
        decision = "DEFENSIVE"
        decision_sub = "Preserve capital — wait for better entries"

    return total, pillars, decision, decision_sub, warnings, tailwinds


# ── Execution Window ─────────────────────────────────────────────────────────

def calc_execution_window(df_all):
    """Assess execution quality from portfolio RS data."""
    result = {}
    n = len(df_all)
    if n == 0:
        return {"score": 0, "breakouts_working": "N/A", "leaders_holding": "N/A",
                "pullbacks_bought": "N/A", "follow_through": "N/A"}

    breakout_tickers = df_all[df_all["Pos52"] >= 90]
    if len(breakout_tickers) > 0:
        held = breakout_tickers[breakout_tickers["vsMA50"] > 0]
        result["breakouts_working"] = "Yes" if len(held) / len(breakout_tickers) >= 0.6 else "No"
    else:
        result["breakouts_working"] = "None active"

    if "RS_Label" in df_all.columns:
        leaders = df_all[df_all["RS_Label"] == "Leader"]
        if len(leaders) > 0:
            holding = leaders[leaders["vsMA50"] > -5]
            result["leaders_holding"] = "Yes" if len(holding) / len(leaders) >= 0.6 else "Fading"
        else:
            result["leaders_holding"] = "None"
    else:
        result["leaders_holding"] = "N/A"

    pullback = df_all[(df_all["Pos52"] >= 50) & (df_all["Pos52"] <= 85) & (df_all["RSI"] >= 40) & (df_all["RSI"] <= 55)]
    if len(pullback) > 0:
        bought = pullback[pullback["vsMA50"] > -3]
        result["pullbacks_bought"] = "Yes" if len(bought) / len(pullback) >= 0.5 else "No"
    else:
        result["pullbacks_bought"] = "N/A"

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
