"""Homework 3 starter — Algory QI Education, Fall 2026

Fill in every function marked TODO. Keep the names: the checker looks for them.
Run `python check_hw03.py` as you go.
"""
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- Q1
def simulate_dice(trials, seed=0):
    """Pick a 4-sided or 6-sided die at random, roll it, repeat.

    Return the estimated P(picked the 4-sided die | rolled a 1).
    The exact answer is 0.6. Take a seed so your result is reproducible.
    """
    np.random.seed(seed)

    ones=0
    four_and_one=0
    for _ in range(trials):
        sides=np.random.choice([4,6])
        roll=np.random.randint(1,sides+1)
        if roll==1:
            ones+=1
            if sides==4:
                four_and_one+=1
    return four_and_one/ones
# ---------------------------------------------------------------- Q2
def simulate_coins(trials, seed=0):
    """Flip three fair coins, get paid (heads x tails). Return the mean payout.

    The exact answer is 1.5.
    """
    np.random.seed(seed)

    total_payout = 0

    for _ in range(trials):
        heads = np.random.randint(0, 2, 3)
        num_heads = np.sum(heads)
        num_tails = 3 - num_heads

        payout = num_heads * num_tails
        total_payout += payout

    return total_payout / trials


# ---------------------------------------------------------------- Q5/Q6
def p_down(returns):
    """Fraction of days with a negative return. Count them yourself."""
    down_days = 0

    for r in returns:
        if r < 0:
            down_days += 1

    return down_days / len(returns)


def p_down_given_down(returns):
    """P(tomorrow is down | today was down), counted directly from the series."""
    down_today = 0
    down_tomorrow = 0

    for i in range(len(returns) - 1):
        if returns.iloc[i] < 0:
            down_today += 1

            if returns.iloc[i + 1] < 0:
                down_tomorrow += 1

    return down_tomorrow / down_today


def p_down_given_big_drop(returns, threshold=-0.02):
    """P(tomorrow is down | today fell more than the threshold).

    Also report how many days the estimate uses. A handful of days is a much
    weaker claim than thousands, and the count is what tells a reader which of
    the two this is.
    """
    big_drop_days = 0
    down_tomorrow = 0

    for i in range(len(returns) - 1):
        if returns.iloc[i] < threshold:
            big_drop_days += 1

            if returns.iloc[i + 1] < 0:
                down_tomorrow += 1

    return down_tomorrow / big_drop_days


# ---------------------------------------------------------------- Q7
def expected_present_value(cash_flows, rate, survival_prob):
    """Present value where the company survives EACH year with survival_prob.

    Year 1 is certain. Year 2 arrives with probability survival_prob, year 3
    with survival_prob squared, and so on. This is a yearly hazard rate.

    Note this is deliberately more general than the Session 3 slide, which had a
    single shutdown event after year 1 and came to 16.98. A yearly 50% survival
    is a harsher assumption and gives 15.10. Getting 16.98 here means you applied
    the probability once instead of compounding it.
    """
    total = 0

    for i, cash_flow in enumerate(cash_flows):
        survival = survival_prob ** i
        discount = (1 + rate) ** (i + 1)
        total += cash_flow * survival / discount

    return total

def main():
    print("Q1  P(4-sided | rolled a 1) =", simulate_dice(100_000))
    print("Q2  expected three-coin payout =", simulate_coins(100_000))
    trial_counts = [100, 1000, 10000, 100000]
    estimates = []

    for n in trial_counts:
        estimate = simulate_coins(n)
        estimates.append(estimate)
        print("Q3", n, "trials:", estimate)

    plt.plot(trial_counts, estimates, marker="o")
    plt.axhline(1.5, linestyle="--")
    plt.xlabel("Number of trials")
    plt.ylabel("Estimated expected payout")
    plt.title("Three-Coin Simulation")
    plt.show()

    spy = yf.download("SPY", period="10y", auto_adjust=True)

    closes = spy["Close"]["SPY"]
    returns = closes.pct_change().dropna()

    print("Q4 trading days =", len(returns))
    print("Q4 mean daily return =", returns.mean())
    print("Q5 P(down) =", p_down(returns))
    print("Q5 P(down tomorrow | down today) =", p_down_given_down(returns))

    big_drop_count = sum(1 for r in returns[:-1] if r < -0.02)
    print("Q6 P(down tomorrow | today down > 2%) =",
          p_down_given_big_drop(returns))
    print("Q6 number of big-drop days =", big_drop_count)
    print("Q7  EPV =", expected_present_value([10, 10, 10], 0.10, 0.5))


if __name__ == "__main__":
    main()