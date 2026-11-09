"""
Homework 7: Machine learning on a real prediction problem
Algory QI Education, Fall 2026

Run:
    python hw07_starter.py

Dependencies:
    pip install yfinance numpy pandas matplotlib scikit-learn

Place hw04_starter.py in the same folder to use the Homework 4 backtester.
The script includes a compatible fallback only if that module is unavailable.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TICKER = "SPY"
YEARS = 15
RANDOM_STATE = 42

# 61 prices give us today's price plus 60 prior trading days.
# The backtester uses the same lookback convention.
LOOKBACK = 61

FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "price_vs_sma50",
    "sma20_vs_sma50",
]


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def _features_from_window(window):
    """Calculate features using only the prices in a window ending today."""
    p = np.asarray(window, dtype=float)

    if len(p) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} prices to calculate features.")

    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("Prices must be finite and strictly positive.")

    daily_returns = p[1:] / p[:-1] - 1.0

    return np.array([
        p[-1] / p[-2] - 1.0,                         # 1-day return
        p[-1] / p[-6] - 1.0,                         # 5-day return
        p[-1] / p[-11] - 1.0,                        # 10-day return
        p[-1] / p[-21] - 1.0,                        # 20-day return
        p[-1] / p[-61] - 1.0,                        # 60-day return
        np.std(daily_returns[-20:], ddof=1),          # 20-day realised volatility
        p[-1] / np.mean(p[-50:]) - 1.0,              # distance from 50-day SMA
        np.mean(p[-20:]) / np.mean(p[-50:]) - 1.0,   # short vs long moving average
    ], dtype=float)


def build_features(prices, lookbacks=(1, 5, 20, 60)):
    """
    One row per eligible trading day: [features..., next_day_return].

    Features use prices through day i only. The label is the return from
    day i to day i+1. The first eligible day has 60 prior trading days.
    """
    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("prices must be a one-dimensional sequence.")
    if len(prices) <= LOOKBACK:
        return []

    rows = []
    for i in range(LOOKBACK - 1, len(prices) - 1):
        window = prices[i - LOOKBACK + 1:i + 1]
        features = _features_from_window(window)
        next_day_return = prices[i + 1] / prices[i] - 1.0
        rows.append(np.append(features, next_day_return))

    return rows


def split_by_time(rows, train=0.6, val=0.2):
    """Split rows chronologically. No shuffling; every row appears once."""
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("train and val must be positive and sum to less than 1.")

    n = len(rows)
    train_end = int(n * train)
    val_end = train_end + int(n * val)

    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def up_day_baseline(prices):
    """Fraction of observed daily returns that are positive."""
    prices = np.asarray(prices, dtype=float)
    returns = prices[1:] / prices[:-1] - 1.0
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return float("nan")
    return float(np.mean(returns > 0))


# ---------------------------------------------------------------------
# Data and reporting helpers
# ---------------------------------------------------------------------

def download_spy():
    """Download approximately 15 years of adjusted daily SPY prices."""
    data = yf.download(
        TICKER,
        period=f"{YEARS}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        actions=False,
    )

    if data.empty:
        raise RuntimeError("Yahoo Finance returned no SPY data.")

    # yfinance may return either ordinary columns or MultiIndex columns.
    close_data = data["Close"]
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]

    close_data = close_data.dropna().astype(float)
    dates = pd.DatetimeIndex(close_data.index)
    prices = close_data.to_numpy(dtype=float)

    if len(prices) <= LOOKBACK + 10:
        raise RuntimeError("Not enough SPY observations were downloaded.")

    return dates, prices


def print_split(name, dates, start, end):
    """Print a split's date range and row count; end is exclusive."""
    if start >= end:
        print(f"{name}: 0 rows")
        return
    print(
        f"{name}: {end - start} rows | "
        f"{dates[start].date()} through {dates[end - 1].date()}"
    )


def safe_correlation(actual, predicted):
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def plot_tuning_curves(curves):
    """Plot validation R-squared against each tuned hyperparameter."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, curve in zip(axes, curves):
        values = curve["values"]
        scores = curve["scores"]
        ax.plot(range(len(values)), scores, marker="o")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(curve["parameter"])
        ax.set_ylabel("Validation R-squared")
        ax.set_title(curve["model"])
        ax.grid(True, alpha=0.3)

        best_index = int(np.nanargmax(scores))
        ax.scatter(
            [best_index], [scores[best_index]],
            marker="*", s=180, zorder=3
        )

    fig.suptitle("Homework 7: Hyperparameter tuning on validation data")
    fig.tight_layout()
    fig.savefig("hw07_tuning_curves.png", dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Homework 4 backtester integration
# ---------------------------------------------------------------------

def _fallback_backtest(prices, strategy, lookback=LOOKBACK):
    """
    Compatible fallback for the known HW4 API:
    strategy receives prices[i-lookback+1:i+1] and earns the next-day return.
    """
    prices = np.asarray(prices, dtype=float)
    daily_strategy_returns = []
    invested = []

    for i in range(lookback - 1, len(prices) - 1):
        window = prices[i - lookback + 1:i + 1]
        is_invested = bool(strategy(window))
        next_return = prices[i + 1] / prices[i] - 1.0
        daily_strategy_returns.append(next_return if is_invested else 0.0)
        invested.append(is_invested)

    daily_strategy_returns = np.asarray(daily_strategy_returns, dtype=float)
    invested = np.asarray(invested, dtype=bool)

    if len(daily_strategy_returns) == 0:
        raise ValueError("Not enough prices to run the backtest.")

    equity = np.cumprod(1.0 + daily_strategy_returns)
    total_return = float(equity[-1] - 1.0)
    annualised_return = float(equity[-1] ** (252.0 / len(equity)) - 1.0)

    running_peak = np.maximum.accumulate(np.insert(equity, 0, 1.0))[1:]
    drawdowns = equity / running_peak - 1.0
    max_drawdown = float(np.min(drawdowns))

    return {
        "total_return": total_return,
        "annualised_return": annualised_return,
        "max_drawdown": max_drawdown,
        "fraction_invested": float(np.mean(invested)),
    }


def get_backtest_function():
    """Prefer the actual Homework 4 backtester when it is available."""
    try:
        from hw04_starter import backtest
        print("\nUsing backtest() imported from hw04_starter.py.")
        return backtest
    except (ImportError, AttributeError):
        warnings.warn(
            "Could not import backtest from hw04_starter.py. "
            "Using the compatible HW4-API fallback included in this file."
        )
        return _fallback_backtest


# ---------------------------------------------------------------------
# Main assignment
# ---------------------------------------------------------------------

def main():
    # Q1: Download data and build the feature table.
    dates, prices = download_spy()
    rows = build_features(prices)

    if not rows:
        raise RuntimeError("Feature table is empty.")

    feature_dates = dates[LOOKBACK - 1:-1]
    table = np.asarray(rows, dtype=float)

    print("\n" + "=" * 72)
    print("Q1. FEATURE TABLE")
    print("=" * 72)
    print(f"Feature table shape: {table.shape}")
    print(f"Feature columns: {FEATURE_NAMES + ['next_day_return']}")
    print(pd.DataFrame(table, columns=FEATURE_NAMES + ["next_day_return"])
          .head(3).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Hand-check one row: all features end on the row's date; label is next day.
    check_i = min(100, len(prices) - 2)
    check_features = _features_from_window(
        prices[check_i - LOOKBACK + 1:check_i + 1]
    )
    check_label = prices[check_i + 1] / prices[check_i] - 1.0
    print("\nHand check of one feature row:")
    print(f"Feature date: {dates[check_i].date()}")
    print(f"Last price used by features: {prices[check_i]:.4f}")
    print(f"Next day's price (used only for label): {prices[check_i + 1]:.4f}")
    print(f"Computed next-day label: {check_label:.8f}")
    print(f"Features recomputed from prices through that date: "
          f"{np.allclose(check_features, table[check_i - (LOOKBACK - 1), :-1])}")

    # Q2: Chronological 60/20/20 split.
    train_rows, val_rows, test_rows = split_by_time(rows, train=0.6, val=0.2)

    n_train = len(train_rows)
    n_val = len(val_rows)
    n_test = len(test_rows)
    train_end = n_train
    val_end = n_train + n_val

    print("\n" + "=" * 72)
    print("Q2. CHRONOLOGICAL SPLIT")
    print("=" * 72)
    print_split("Training", feature_dates, 0, train_end)
    print_split("Validation", feature_dates, train_end, val_end)
    print_split("Test", feature_dates, val_end, len(rows))
    print(f"Total rows: {n_train + n_val + n_test}")
    print(f"Up-day baseline over downloaded prices: {up_day_baseline(prices):.3%}")

    X_train = np.asarray(train_rows)[:, :-1]
    y_train = np.asarray(train_rows)[:, -1]
    X_val = np.asarray(val_rows)[:, :-1]
    y_val = np.asarray(val_rows)[:, -1]
    X_test = np.asarray(test_rows)[:, :-1]
    y_test = np.asarray(test_rows)[:, -1]

    # Q3: Fit the four required models.
    models = {
        "Linear Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            loss="squared_error",
            random_state=RANDOM_STATE,
        ),
    }

    fitted_models = {}
    validation_scores = {}

    print("\n" + "=" * 72)
    print("Q3. BASELINE MODEL PERFORMANCE")
    print("=" * 72)
    print(f"{'Model':<22} {'Train R2':>12} {'Validation R2':>16}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_score = r2_score(y_train, model.predict(X_train))
        val_score = r2_score(y_val, model.predict(X_val))
        fitted_models[name] = model
        validation_scores[name] = float(val_score)
        print(f"{name:<22} {train_score:>12.6f} {val_score:>16.6f}")

    # Q4: Tune one hyperparameter for each of the last three models.
    tuning_curves = []

    # Ridge: alpha
    ridge_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge_scores = []
    ridge_candidates = []
    for alpha in ridge_values:
        candidate = Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])
        candidate.fit(X_train, y_train)
        score = r2_score(y_val, candidate.predict(X_val))
        ridge_candidates.append(candidate)
        ridge_scores.append(float(score))

    ridge_best_i = int(np.argmax(ridge_scores))
    fitted_models["Ridge Regression"] = ridge_candidates[ridge_best_i]
    validation_scores["Ridge Regression"] = ridge_scores[ridge_best_i]
    tuning_curves.append({
        "model": "Ridge Regression",
        "parameter": "alpha",
        "values": ridge_values,
        "scores": ridge_scores,
    })

    # Random forest: maximum tree depth
    rf_values = [2, 3, 5, 8, 12, None]
    rf_scores = []
    rf_candidates = []
    for depth in rf_values:
        candidate = RandomForestRegressor(
            n_estimators=300,
            max_depth=depth,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        candidate.fit(X_train, y_train)
        score = r2_score(y_val, candidate.predict(X_val))
        rf_candidates.append(candidate)
        rf_scores.append(float(score))

    rf_best_i = int(np.argmax(rf_scores))
    fitted_models["Random Forest"] = rf_candidates[rf_best_i]
    validation_scores["Random Forest"] = rf_scores[rf_best_i]
    tuning_curves.append({
        "model": "Random Forest",
        "parameter": "max_depth",
        "values": rf_values,
        "scores": rf_scores,
    })

    # Gradient boosting: number of boosting stages
    gb_values = [50, 100, 200, 400, 600]
    gb_scores = []
    gb_candidates = []
    for n_estimators in gb_values:
        candidate = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.03,
            max_depth=2,
            loss="squared_error",
            random_state=RANDOM_STATE,
        )
        candidate.fit(X_train, y_train)
        score = r2_score(y_val, candidate.predict(X_val))
        gb_candidates.append(candidate)
        gb_scores.append(float(score))

    gb_best_i = int(np.argmax(gb_scores))
    fitted_models["Gradient Boosting"] = gb_candidates[gb_best_i]
    validation_scores["Gradient Boosting"] = gb_scores[gb_best_i]
    tuning_curves.append({
        "model": "Gradient Boosting",
        "parameter": "n_estimators",
        "values": gb_values,
        "scores": gb_scores,
    })

    print("\n" + "=" * 72)
    print("Q4. HYPERPARAMETER TUNING")
    print("=" * 72)
    print(f"Ridge alpha: {ridge_values[ridge_best_i]} | "
          f"validation R2 = {ridge_scores[ridge_best_i]:.6f}")
    print(f"Random Forest max_depth: {rf_values[rf_best_i]} | "
          f"validation R2 = {rf_scores[rf_best_i]:.6f}")
    print(f"Gradient Boosting n_estimators: {gb_values[gb_best_i]} | "
          f"validation R2 = {gb_scores[gb_best_i]:.6f}")

    plot_tuning_curves(tuning_curves)

    # Q5: Select strictly on validation performance, then test once.
    best_name = max(validation_scores, key=validation_scores.get)
    best_validation_score = validation_scores[best_name]
    best_model = clone(fitted_models[best_name])

    # Refit on train + validation after model selection, before the one test run.
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    best_model.fit(X_train_val, y_train_val)

    test_predictions = best_model.predict(X_test)
    test_r2 = r2_score(y_test, test_predictions)
    test_corr = safe_correlation(y_test, test_predictions)

    print("\n" + "=" * 72)
    print("Q5. FINAL TEST PERFORMANCE")
    print("=" * 72)
    print(f"Selected model: {best_name}")
    print(f"Best validation R-squared: {best_validation_score:.6f}")
    print(f"Test R-squared: {test_r2:.6f}")
    print(f"Correlation(predicted, actual): {test_corr:.6f}")

    # Q6: Use the HW4 backtester. The test price segment includes the 60
    # preceding prices needed to calculate the first test-day feature row.
    test_first_feature_index = LOOKBACK - 1 + val_end
    test_last_feature_index = LOOKBACK - 1 + len(rows) - 1
    price_segment_start = test_first_feature_index - (LOOKBACK - 1)
    price_segment_end_exclusive = test_last_feature_index + 2
    test_prices_for_backtest = prices[
        price_segment_start:price_segment_end_exclusive
    ]

    def model_strategy(price_window):
        x = _features_from_window(price_window).reshape(1, -1)
        prediction = float(best_model.predict(x)[0])
        return prediction > 0.0

    def buy_and_hold_strategy(price_window):
        return True

    backtest = get_backtest_function()
    strategy_results = backtest(
        test_prices_for_backtest, model_strategy, lookback=LOOKBACK
    )
    buy_hold_results = backtest(
        test_prices_for_backtest, buy_and_hold_strategy, lookback=LOOKBACK
    )

    print("\n" + "=" * 72)
    print("Q6. TEST-PERIOD STRATEGY BACKTEST")
    print("=" * 72)
    print("Strategy: hold SPY if predicted next-day return > 0; otherwise cash.")
    print(f"{'Metric':<25} {'Model strategy':>18} {'Buy and hold':>18}")
    for key, label in [
        ("annualised_return", "Annualised return"),
        ("max_drawdown", "Maximum drawdown"),
        ("fraction_invested", "Fraction invested"),
    ]:
        print(
            f"{label:<25} "
            f"{strategy_results[key]:>17.3%} "
            f"{buy_hold_results[key]:>17.3%}"
        )

    # Q7: Rank feature importance or absolute standardised coefficients.
    print("\n" + "=" * 72)
    print("Q7. FEATURE IMPORTANCE")
    print("=" * 72)

    if best_name in ("Random Forest", "Gradient Boosting"):
        importances = best_model.feature_importances_
        ranking = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda item: item[1],
            reverse=True,
        )
        print("Tree feature importances:")
        for feature, importance in ranking:
            print(f"{feature:<22} {importance:.6f}")
    else:
        coefficients = best_model.named_steps["model"].coef_
        ranking = sorted(
            zip(FEATURE_NAMES, coefficients),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        print("Standardised linear coefficients (ranked by absolute value):")
        for feature, coefficient in ranking:
            print(f"{feature:<22} {coefficient:+.8f}")

    print("\nSaved tuning plot to hw07_tuning_curves.png")
    print("\nReminder: the backtest does not include transaction costs, slippage,")
    print("taxes, or interest earned on cash. Discuss these limitations in the write-up.")


if __name__ == "__main__":
    main()
