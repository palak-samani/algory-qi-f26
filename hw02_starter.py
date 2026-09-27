"""Homework 2 starter — Algory QI Education, Fall 2026

Fill in every function marked TODO. Do not rename them: the checker looks for
these exact names. Run `python check_hw02.py` before you submit.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- Q1
def present_value(cash_flows, rate):
    """Present value of a list of cash flows, the first arriving in one year.

    present_value([10, 15, 20], 0.10) -> 36.51
    """
    total = 0

    for t, cash_flow in enumerate(cash_flows, start=1):
        total += cash_flow / (1 + rate) ** t

    return total


# ---------------------------------------------------------------- Q2
def bond_price(face, coupon_rate, years, market_rate):
    """Price of a bond paying an annual coupon and repaying face at maturity.

    The final year pays the coupon AND the face value. That is the usual bug.
    bond_price(1000, 0.04, 10, 0.04) -> exactly 1000.0
    """
    coupon = face * coupon_rate

    price = 0

    for t in range(1, years + 1):
        price += coupon / (1 + market_rate) ** t

    price += face / (1 + market_rate) ** years

    return price


# ---------------------------------------------------------------- Q4/Q5
def annualised_return(prices):
    """Annualised return from a price series, using 252 trading days."""
    prices = np.asarray(prices)

    daily_return = prices[-1] / prices[0]

    years = (len(prices) - 1) / 252

    return daily_return ** (1 / years) - 1


def annualised_volatility(prices):
    """Annualised standard deviation of daily returns."""
    prices = np.asarray(prices)

    daily_returns = prices[1:] / prices[:-1] - 1

    return np.std(daily_returns, ddof=1) * np.sqrt(252)


def beta(stock_prices, market_prices):
    """Beta of a stock against the market.

    Covariance of the two RETURN series divided by the variance of the market's.
    Computing this on prices instead of returns is a common and silent error.
    beta(spy, spy) -> 1.0
    """
    stock_prices = np.asarray(stock_prices)
    market_prices = np.asarray(market_prices)

    stock_returns = stock_prices[1:] / stock_prices[:-1] - 1
    market_returns = market_prices[1:] / market_prices[:-1] - 1

    covariance = np.cov(stock_returns, market_returns, ddof=1)[0, 1]
    market_variance = np.var(market_returns, ddof=1)

    return covariance / market_variance


# ---------------------------------------------------------------- your answers
def main():
    """Everything the assignment asks you to print goes here."""
    print("Q1  present_value([10, 15, 20], 0.10) =", present_value([10, 15, 20], 0.10))
    face=1000
    coupon_rate=0.04
    years=10
    market_rates = np.array([0.02, 0.04, 0.06])

    bond_prices = [
        bond_price(face, coupon_rate, years, rate)
        for rate in market_rates
    ]

    print("\nQ3 Bond prices:")
    for rate, price in zip(market_rates, bond_prices):
        print(f"Market rate: {rate:.2%} -> Bond price: ${price:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(market_rates * 100, bond_prices, marker="o")
    plt.xlabel("Market Interest Rate (%)")
    plt.ylabel("Bond Price ($)")
    plt.title("10-Year Bond Price vs Market Interest Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hw2_bond_price_vs_rate.png", dpi=300)
    plt.show()
    # TODO: Q3 — three bond prices, then the price-vs-rate plot
    tickers = ["AAPL", "NVDA", "MSFT"]
    benchmark = "SPY"

    all_tickers = tickers + [benchmark]

    prices = {}

    print("\nQ4  Trading days:")

    for ticker in all_tickers:
        data = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
        close = data["Close"].dropna()

        prices[ticker] = close

        print(f"{ticker}: {len(close)} trading days")
    # TODO: Q4 — download your three tickers and SPY, print trading days for each
        # Q5 — return, volatility and beta

    print("\nQ5  Return, volatility and beta:")
    print(f"{'Ticker':<8}{'Return':>12}{'Volatility':>15}{'Beta':>10}")

    for ticker in tickers:
        ret = annualised_return(prices[ticker])
        vol = annualised_volatility(prices[ticker])
        b = beta(prices[ticker], prices[benchmark])

        print(f"{ticker:<8}{ret:>11.2%}{vol:>14.2%}{b:>10.2f}")
    # TODO: Q5 — the table of return, volatility and beta
    # TODO: Q6 — both rankings


if __name__ == "__main__":
    main()
