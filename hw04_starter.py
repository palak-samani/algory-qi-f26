
"""Homework 4 starter — Algory QI Education, Fall 2026."""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

LOOKBACK = 60


# ---------------------------------------------------------------- Q1
def load_and_split(split_date="2011-09-09", prices=None, warmup=LOOKBACK):
    """Split prices into training and test data, retaining warm-up history."""

    if prices is None:
        downloaded = yf.download(
            "^SP500TR",
            start="1990-01-01",
            auto_adjust=True,
            progress=False
        )

        if downloaded.empty:
            raise ValueError("No prices downloaded. Check your internet connection.")

        if isinstance(downloaded.columns, pd.MultiIndex):
            prices = downloaded["Close"].squeeze()
        else:
            prices = downloaded["Close"]

    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()

    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series or a one-column DataFrame.")

    prices = prices.dropna().sort_index()

    split_day = pd.Timestamp(split_date).normalize()
    index_dates = pd.DatetimeIndex(prices.index)

    # Normalize timezone-aware indices for date comparison.
    if index_dates.tz is not None:
        index_dates = index_dates.tz_localize(None)

    index_dates = index_dates.normalize()
    split_position = index_dates.searchsorted(split_day, side="left")

    if (
        split_position >= len(prices)
        or index_dates[split_position] != split_day
    ):
        raise ValueError("split_date must be a trading day in the price series.")

    training = prices.iloc[:split_position + 1]
    test = prices.iloc[max(0, split_position - warmup + 1):]

    return training, test


# ---------------------------------------------------------------- Q2, Q4, Q5

def always_hold(window):
    """Always invest."""
    return True


def always_cash(window):
    """Never invest."""
    return False


def make_random_hold(p=0.7, seed=0):
    """Return a reproducible random investment strategy."""
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    def random_hold(window):
        return bool(rng.random() < p)

    return random_hold


def five_day_reversal(window):
    """Hold unless the five-day return ending today is positive."""
    window = np.asarray(window, dtype=float)

    five_day_return = window[-1] / window[-6] - 1
    return bool(five_day_return <= 0)


def make_momentum_strategy(periods):
    """Invest when today's close exceeds the close 'periods' observations ago."""

    def momentum_strategy(window):
        window = np.asarray(window, dtype=float)

        if len(window) <= periods:
            raise ValueError("Window is too short for this momentum period.")

        return bool(window[-1] > window[-periods - 1])

    return momentum_strategy


def my_strategy(window):
    """
    Initial personal rule: invest when 20-day momentum is positive.
    Otherwise, hold cash.
    """
    return make_momentum_strategy(20)(window)


# ---------------------------------------------------------------- Q3
def backtest(prices, strategy, lookback=LOOKBACK):
    """
    Walk forward one day at a time.

    The strategy sees the lookback closes ending today.
    If invested, it earns tomorrow's close-to-close return.
    """

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    if isinstance(prices, (pd.Series, pd.DataFrame)):
        prices = prices.squeeze().to_numpy(dtype=float)
    else:
        prices = np.asarray(prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("prices must be one-dimensional.")

    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Prices must be finite and strictly positive.")

    if len(prices) <= lookback:
        raise ValueError("Need more prices than the lookback length.")

    daily_returns = []
    invested_days = 0

    for i in range(lookback - 1, len(prices) - 1):
        # Includes today's close, but never tomorrow's close.
        window = prices[i - lookback + 1:i + 1]

        hold = bool(strategy(window))

        if hold:
            daily_return = prices[i + 1] / prices[i] - 1
            invested_days += 1
        else:
            daily_return = 0.0

        daily_returns.append(daily_return)

    daily_returns = np.asarray(daily_returns, dtype=float)
    days = len(daily_returns)

    # Compound the daily returns.
    equity = np.concatenate([
        [1.0],
        np.cumprod(1 + daily_returns)
    ])

    total_return = float(equity[-1] - 1)

    annualised_return = float(
        equity[-1] ** (252 / days) - 1
    )

    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1
    max_drawdown = float(np.min(drawdowns))

    fraction_invested = invested_days / days

    return {
        "total_return": total_return,
        "annualised_return": annualised_return,
        "max_drawdown": max_drawdown,
        "fraction_invested": fraction_invested,
    }


# ---------------------------------------------------------------- Main
def main():
    train, test = load_and_split()

    # Q4/Q5: baseline strategies.
    baseline_strategies = [
        ("always hold", always_hold),
        ("always cash", always_cash),
        ("random 70%", make_random_hold(0.7, seed=0)),
        ("5-day reversal", five_day_reversal),
    ]

    print("\nTRAINING RESULTS")
    print("-" * 72)

    for name, strat in baseline_strategies:
        r = backtest(train, strat)
        print(
            f"{name:<22} "
            f"total {r['total_return']:+.2%}   "
            f"annual {r['annualised_return']:+.2%}   "
            f"max DD {r['max_drawdown']:+.2%}   "
            f"invested {r['fraction_invested']:.1%}"
        )

    # Q6: print every momentum variant tested.
    variants = [
        ("5-close momentum", make_momentum_strategy(5)),
        ("20-day momentum", make_momentum_strategy(20)),
        ("59-interval momentum", make_momentum_strategy(59)),
    ]

    print("\nSTRATEGY VARIANTS")
    print("-" * 72)

    for name, strat in variants:
        r = backtest(train, strat)
        print(
            f"{name:<22} "
            f"total {r['total_return']:+.2%}   "
            f"annual {r['annualised_return']:+.2%}   "
            f"max DD {r['max_drawdown']:+.2%}   "
            f"invested {r['fraction_invested']:.1%}"
        )

    # Q7/Q8: select the predefined personal rule and test it out of sample.
    # Do not use test results to choose or tune the strategy.
    test_results = backtest(test, my_strategy)

    print("\nOUT-OF-SAMPLE TEST: MY STRATEGY")
    print("-" * 72)
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")

    # Q9: what does the rule say using the latest available 60 closes?
    latest_window = test.iloc[-LOOKBACK:].to_numpy(dtype=float)
    decision = my_strategy(latest_window)

    print("\nLATEST STRATEGY DECISION")
    print("-" * 72)
    print(f"Latest close: {test.index[-1].date()}")
    print(f"Decision: {'INVEST' if decision else 'HOLD CASH'}")

    # Plot the training price history.
    plt.figure(figsize=(12, 5))
    train.plot(title="S&P 500 Total Return Index: Training Data")
    plt.xlabel("Date")
    plt.ylabel("Adjusted close")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()