import sys, subprocess
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime

TICKERS = [
    "IREN", "CIFR", "RKLB", "ONDS", "NBIS", "CRCL", "ASTS", "HOOD",
    "AREC", "ASPI", "MSTR", "BITF", "BMNR", "LTRX", "OSS",
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
        w52h = high.max(); w52l = low.min()
        range_pos = ((price - w52l) / (w52h - w52l) * 100) if w52h != w52l else float("nan")
        vs_ma50 = ((price / ma50) - 1) * 100
        vs_ma200 = ((price / ma200) - 1) * 100 if not np.isnan(ma200) else float("nan")
        return dict(Ticker=ticker, Price=price, RSI=rsi,
                    vsMA50=vs_ma50, vsMA200=vs_ma200,
                    ATRpct=atr_pct, Pos52=range_pos,
                    Low52=w52l, High52=w52h)
    except:
        return None

print("Fetching data...")
rows = [r for t in TICKERS if (r := analyze_ticker(t)) is not None]
df = pd.DataFrame(rows).sort_values("Pos52", ascending=True)
tickers = df["Ticker"].tolist()
n = len(tickers)

# ── colour helpers ──────────────────────────────────────────────────────────
def rsi_color(v):
    if v >= 70: return "#e74c3c"
    if v <= 30: return "#2ecc71"
    return "#3498db"

def ma_color(v):
    if np.isnan(v): return "#888888"
    return "#2ecc71" if v >= 0 else "#e74c3c"

def pos_color(v):
    if v >= 70: return "#27ae60"
    if v >= 40: return "#f39c12"
    return "#e74c3c"

def atr_color(v):
    if v >= 10: return "#e74c3c"
    if v >= 7:  return "#f39c12"
    return "#2ecc71"

# ── figure ──────────────────────────────────────────────────────────────────
BG   = "#0d1117"
FG   = "#e6edf3"
GRID = "#21262d"

fig = plt.figure(figsize=(20, 22), facecolor=BG)
fig.suptitle(
    f"Portfolio Technical Analysis  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    color=FG, fontsize=15, fontweight="bold", y=0.98
)

gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
              left=0.08, right=0.97, top=0.94, bottom=0.04)

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(axis="x", color=GRID, linewidth=0.6, linestyle="--")

# ── 1. 52-week range position (horizontal bar with range line) ───────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "52-Week Range Position  (% from 52wk Low to High)")
y_pos = range(n)
for i, row in enumerate(df.itertuples()):
    # grey background bar (full range)
    ax1.barh(i, 100, color="#21262d", height=0.6)
    # coloured fill bar
    ax1.barh(i, row.Pos52, color=pos_color(row.Pos52), height=0.6, alpha=0.85)
    ax1.text(row.Pos52 + 1, i, f"{row.Pos52:.1f}%", va="center",
             color=FG, fontsize=7.5)
ax1.set_yticks(list(y_pos))
ax1.set_yticklabels(tickers, fontsize=8.5)
ax1.set_xlim(0, 115)
ax1.axvline(50, color="#555", linewidth=0.8, linestyle=":")
ax1.set_xlabel("Position within 52-week range (%)", color=FG, fontsize=9)
low_p  = mpatches.Patch(color="#e74c3c", label="< 40%  (near lows)")
mid_p  = mpatches.Patch(color="#f39c12", label="40–70%")
high_p = mpatches.Patch(color="#27ae60", label="> 70%  (near highs)")
ax1.legend(handles=[low_p, mid_p, high_p], facecolor="#161b22",
           edgecolor=GRID, labelcolor=FG, fontsize=8, loc="lower right")

# ── 2. RSI ───────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, "RSI (14)")
df_rsi = df.sort_values("RSI")
colors = [rsi_color(v) for v in df_rsi["RSI"]]
bars = ax2.barh(df_rsi["Ticker"], df_rsi["RSI"], color=colors, height=0.6)
ax2.axvline(70, color="#e74c3c", linewidth=1, linestyle="--", label="Overbought 70")
ax2.axvline(30, color="#2ecc71", linewidth=1, linestyle="--", label="Oversold 30")
ax2.axvline(50, color="#555", linewidth=0.6, linestyle=":")
ax2.set_xlim(0, 90)
for bar, val in zip(bars, df_rsi["RSI"]):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}", va="center", color=FG, fontsize=7.5)
ax2.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=FG, fontsize=7.5)

# ── 3. ATR% ──────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3, "ATR %  (14-day Average True Range as % of Price)")
df_atr = df.sort_values("ATRpct")
colors3 = [atr_color(v) for v in df_atr["ATRpct"]]
bars3 = ax3.barh(df_atr["Ticker"], df_atr["ATRpct"], color=colors3, height=0.6)
ax3.axvline(10, color="#e74c3c", linewidth=1, linestyle="--", label="High vol 10%")
ax3.axvline(7,  color="#f39c12", linewidth=1, linestyle="--", label="Med vol 7%")
for bar, val in zip(bars3, df_atr["ATRpct"]):
    ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", color=FG, fontsize=7.5)
ax3.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=FG, fontsize=7.5)

# ── 4. vs MA50 ───────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
style_ax(ax4, "Price vs 50-Day MA (%)")
df_ma50 = df.sort_values("vsMA50")
colors4 = [ma_color(v) for v in df_ma50["vsMA50"]]
bars4 = ax4.barh(df_ma50["Ticker"], df_ma50["vsMA50"], color=colors4, height=0.6)
ax4.axvline(0, color=FG, linewidth=0.8, linestyle="-")
for bar, val in zip(bars4, df_ma50["vsMA50"]):
    x_off = val + 0.5 if val >= 0 else val - 0.5
    ha = "left" if val >= 0 else "right"
    ax4.text(x_off, bar.get_y() + bar.get_height()/2,
             f"{val:+.1f}%", va="center", ha=ha, color=FG, fontsize=7.5)

# ── 5. vs MA200 ──────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
style_ax(ax5, "Price vs 200-Day MA (%)")
df_ma200 = df.dropna(subset=["vsMA200"]).sort_values("vsMA200")
colors5 = [ma_color(v) for v in df_ma200["vsMA200"]]
bars5 = ax5.barh(df_ma200["Ticker"], df_ma200["vsMA200"], color=colors5, height=0.6)
ax5.axvline(0, color=FG, linewidth=0.8)
for bar, val in zip(bars5, df_ma200["vsMA200"]):
    x_off = val + 0.5 if val >= 0 else val - 0.5
    ha = "left" if val >= 0 else "right"
    ax5.text(x_off, bar.get_y() + bar.get_height()/2,
             f"{val:+.1f}%", va="center", ha=ha, color=FG, fontsize=7.5)

# ── 6. Scatter: RSI vs 52wk position ─────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, :])
style_ax(ax6, "RSI vs 52-Week Position  (size = ATR volatility)")
sc_colors = [pos_color(r.Pos52) for r in df.itertuples()]
sizes = (df["ATRpct"] * 18) ** 1.3
sc = ax6.scatter(df["Pos52"], df["RSI"], c=sc_colors, s=sizes,
                 alpha=0.85, edgecolors="#0d1117", linewidths=0.6, zorder=3)
for _, row in df.iterrows():
    ax6.annotate(row["Ticker"],
                 (row["Pos52"], row["RSI"]),
                 textcoords="offset points", xytext=(0, 7),
                 ha="center", color=FG, fontsize=8, fontweight="bold")
ax6.axhline(70, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7, label="RSI 70 overbought")
ax6.axhline(30, color="#2ecc71", linewidth=0.8, linestyle="--", alpha=0.7, label="RSI 30 oversold")
ax6.axvline(50, color="#555", linewidth=0.8, linestyle=":", alpha=0.7)
ax6.set_xlabel("52-Week Range Position (%)", color=FG, fontsize=9)
ax6.set_ylabel("RSI (14)", color=FG, fontsize=9)
ax6.set_xlim(-5, 105); ax6.set_ylim(20, 85)

# quadrant labels
for (x, y, label) in [(75, 78, "STRONG\n(high pos + high RSI)"),
                       (5,  78, "RECOVERING\n(low pos + high RSI)"),
                       (75, 22, "PULLBACK\n(high pos + low RSI)"),
                       (5,  22, "WEAK\n(low pos + low RSI)")]:
    ax6.text(x, y, label, color="#888", fontsize=7, ha="center", va="center",
             style="italic")
ax6.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=FG, fontsize=8)

out = "portfolio_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
plt.show()
