import sys
import subprocess

# Install yfinance if needed
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

TICKERS = [
    "IREN", "CIFR", "RKLB", "ONDS", "NBIS", "CRCL", "ASTS", "HOOD",
    "AREC", "ASPI", "INFQ", "MSTR", "BITF", "BMNR", "LTRX", "OSS",
    "UAMY", "TSLA", "PLTR"
]

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

def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None

        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        price = close.iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else float("nan")
        rsi = calc_rsi(close).iloc[-1]
        atr_pct = calc_atr_pct(high, low, close)

        week52_high = high.max()
        week52_low = low.min()
        range_pos = ((price - week52_low) / (week52_high - week52_low) * 100) if week52_high != week52_low else float("nan")

        vs_ma50 = ((price / ma50) - 1) * 100
        vs_ma200 = ((price / ma200) - 1) * 100 if not np.isnan(ma200) else float("nan")

        return {
            "Ticker": ticker,
            "Price": round(price, 2),
            "RSI(14)": round(rsi, 1),
            "vs MA50%": round(vs_ma50, 1),
            "vs MA200%": round(vs_ma200, 1) if not np.isnan(vs_ma200) else "N/A",
            "ATR%": round(atr_pct, 1),
            "52wk Pos%": round(range_pos, 1),
            "52wk Low": round(week52_low, 2),
            "52wk High": round(week52_high, 2),
        }
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None

def rsi_signal(rsi):
    if rsi < 30: return "OVERSOLD"
    if rsi > 70: return "OVERBOUGHT"
    return "NEUTRAL"

def main():
    print(f"\n{'='*80}")
    print(f"  PORTFOLIO TECHNICAL ANALYSIS  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}\n")
    print(f"Fetching data for {len(TICKERS)} tickers...\n")

    results = []
    for t in TICKERS:
        print(f"  Fetching {t}...", end="", flush=True)
        r = analyze_ticker(t)
        if r:
            results.append(r)
            print(f" ${r['Price']}  RSI:{r['RSI(14)']}  52wkPos:{r['52wk Pos%']}%")
        else:
            print(" SKIPPED")

    if not results:
        print("No data retrieved.")
        return

    df = pd.DataFrame(results)

    # Display full table
    print(f"\n{'='*80}")
    print("FULL RESULTS TABLE")
    print(f"{'='*80}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.1f}" if isinstance(x, float) else str(x))
    print(df.to_string(index=False))

    # RSI signals
    print(f"\n{'='*80}")
    print("RSI SIGNALS")
    print(f"{'='*80}")
    for _, row in df.iterrows():
        sig = rsi_signal(row["RSI(14)"])
        if sig != "NEUTRAL":
            print(f"  {row['Ticker']:6s}  RSI={row['RSI(14)']:5.1f}  [{sig}]")

    # MA trend (above/below MA50 and MA200)
    print(f"\n{'='*80}")
    print("MOVING AVERAGE TREND")
    print(f"{'='*80}")
    print(f"  {'Ticker':<7} {'Price':>7}  {'vs MA50%':>9}  {'vs MA200%':>10}  Trend")
    print(f"  {'-'*55}")
    for _, row in df.iterrows():
        vs50 = row["vs MA50%"]
        vs200 = row["vs MA200%"]
        trend = []
        if vs50 > 0: trend.append("above MA50")
        else: trend.append("below MA50")
        if vs200 != "N/A":
            if float(vs200) > 0: trend.append("above MA200")
            else: trend.append("below MA200")
        print(f"  {row['Ticker']:<7} ${row['Price']:>6.2f}  {vs50:>+8.1f}%  {str(vs200):>9}%  {', '.join(trend)}")

    # Sorted by 52wk position
    print(f"\n{'='*80}")
    print("SORTED BY 52-WEEK RANGE POSITION (highest = closest to 52wk high)")
    print(f"{'='*80}")
    sorted_df = df.sort_values("52wk Pos%", ascending=False)
    print(f"  {'Ticker':<7} {'52wk Pos%':>10}  {'52wk Low':>9}  {'52wk High':>10}  {'Price':>7}")
    print(f"  {'-'*55}")
    for _, row in sorted_df.iterrows():
        bar_len = int(row["52wk Pos%"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {row['Ticker']:<7} {row['52wk Pos%']:>9.1f}%  ${row['52wk Low']:>8.2f}  ${row['52wk High']:>9.2f}  ${row['Price']:>6.2f}  [{bar}]")

    # Save to CSV
    out_file = f"portfolio_tech_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(out_file, index=False)
    print(f"\n  Results saved to: {out_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
