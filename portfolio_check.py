"""
Portfolio Price & RSI Checker
------------------------------
Requires: pip install yfinance pandas

Run: python portfolio_check.py
"""

import sys
try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("Missing dependencies. Run: pip install yfinance pandas")
    sys.exit(1)

from datetime import datetime

# ── Edit your tickers and cost basis here ──────────────────────────────────────
PORTFOLIO = {
    # Core holds
    "IREN":  15.84,
    "CIFR":  6.00,
    "RKLB":  29.18,
    "NBIS":  44.42,
    "ONDS":  6.55,
    "ASTS":  47.78,
    "PLTR":  104.86,
    # Moderate
    "CRCL":  64.44,
    "TSLA":  333.00,
    "OSS":   7.59,
    "HOOD":  97.19,
    "UAMY":  9.09,
    "LTRX":  7.84,
    # Sell candidates
    "MSTR":  407.58,
    "BMNR":  41.57,
    "BITF":  4.06,
    "AREC":  4.49,
    "ASPI":  8.70,
    "INFQ":  19.20,
}

RSI_PERIOD = 14


def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 1)


def rsi_signal(rsi: float) -> str:
    if rsi < 30:
        return "🟢 OVERSOLD"
    elif rsi > 70:
        return "🔴 OVERBOUGHT"
    elif rsi < 40:
        return "↘ soft oversold"
    elif rsi > 60:
        return "↗ soft overbought"
    else:
        return "— neutral"


def main():
    tickers = list(PORTFOLIO.keys())
    print(f"\nFetching data for {len(tickers)} tickers...\n")

    # Download all at once (faster)
    raw = yf.download(tickers, period="60d", auto_adjust=True, progress=False)
    closes = raw["Close"]

    print(f"{'Ticker':<6} {'Price':>8} {'Cost':>8} {'Return':>8} {'RSI':>6}  Signal")
    print("─" * 62)

    for ticker in tickers:
        try:
            series = closes[ticker].dropna()
            price = series.iloc[-1]
            cost = PORTFOLIO[ticker]
            ret = ((price - cost) / cost) * 100
            rsi = calc_rsi(series, RSI_PERIOD)
            signal = rsi_signal(rsi)

            ret_str = f"{ret:+.1f}%"
            print(f"{ticker:<6} ${price:>7.2f} ${cost:>7.2f} {ret_str:>8} {rsi:>6}  {signal}")
        except Exception as e:
            print(f"{ticker:<6}  ERROR: {e}")

    print("─" * 62)
    print(f"Data as of: {datetime.now().strftime('%Y-%m-%d %H:%M')} local time")
    print("Note: prices may be delayed 15 min during market hours.\n")


if __name__ == "__main__":
    main()
