import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# HOMEWORK 1
# AAPL vs SPY
# ============================================================

ticker = "AAPL"
benchmark = "SPY"

# ------------------------------------------------------------
# 1. Get one year of daily closing prices
# ------------------------------------------------------------

stock_data = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
spy_data = yf.Ticker(benchmark).history(period="1y", auto_adjust=False)

stock = stock_data["Close"].dropna()
spy = spy_data["Close"].dropna()

# Keep only dates where both have data
data = pd.concat([stock, spy], axis=1, join="inner")
data.columns = [ticker, benchmark]
data = data.dropna()

# ------------------------------------------------------------
# 2. Print basic information
# ------------------------------------------------------------

print("\n================ DATA =================")

for name in [ticker, benchmark]:
    series = data[name]

    print(f"\n{name}")
    print(f"Trading days: {len(series)}")
    print(f"First date:   {series.index[0].date()}")
    print(f"Last date:    {series.index[-1].date()}")

# ------------------------------------------------------------
# 3. Calculate returns
# ------------------------------------------------------------

stock_return = data[ticker].iloc[-1] / data[ticker].iloc[0] - 1
spy_return = data[benchmark].iloc[-1] / data[benchmark].iloc[0] - 1

# ------------------------------------------------------------
# 4. Calculate annualized volatility
# ------------------------------------------------------------

stock_daily_returns = data[ticker].pct_change().dropna()
spy_daily_returns = data[benchmark].pct_change().dropna()

stock_volatility = stock_daily_returns.std() * np.sqrt(252)
spy_volatility = spy_daily_returns.std() * np.sqrt(252)

# ------------------------------------------------------------
# 5. Print results
# ------------------------------------------------------------

print("\n================ RESULTS =================")

print(f"\n{ticker}")
print(f"Last close:          ${data[ticker].iloc[-1]:.2f}")
print(f"Year return:         {stock_return:.2%}")
print(f"Annualized volatility: {stock_volatility:.2%}")

print(f"\n{benchmark}")
print(f"Last close:          ${data[benchmark].iloc[-1]:.2f}")
print(f"Year return:         {spy_return:.2%}")
print(f"Annualized volatility: {spy_volatility:.2%}")

# ------------------------------------------------------------
# 6. Find biggest single-day move for our stock
# ------------------------------------------------------------

biggest_move_date = stock_daily_returns.abs().idxmax()
biggest_move = stock_daily_returns.loc[biggest_move_date]

print("\n================ BIGGEST DAILY MOVE =================")

print(f"Date: {biggest_move_date.date()}")
print(f"Move: {biggest_move:.2%}")

# ------------------------------------------------------------
# 7. Rebase both series to 100
# ------------------------------------------------------------

rebased = data / data.iloc[0] * 100

# ------------------------------------------------------------
# 8. Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(rebased.index, rebased[ticker], label=ticker)
plt.plot(rebased.index, rebased[benchmark], label=benchmark)

plt.xlabel("Date")
plt.ylabel("Rebased Price (Start = 100)")
plt.title(f"{ticker} vs {benchmark}: One-Year Performance")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("hw1_aapl_vs_spy.png", dpi=300)

plt.show()

print("\nPlot saved as: hw1_aapl_vs_spy.png")