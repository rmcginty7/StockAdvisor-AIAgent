import itertools
import pandas as pd

from stock_advisor.data_fetcher import DataFetcher
from stock_advisor.technical_indicators import TechnicalIndicators
from stock_advisor.analyzer import StockAnalyzer
from stock_advisor.decision_engine import DecisionEngine
from stock_advisor.backtester import Backtester


# ---------------------------------------------------------------------------
# Helpers for tuning
# ---------------------------------------------------------------------------

def set_rsi_thresholds(oversold: int, overbought: int) -> None:
    """
    Globally set the RSI oversold/overbought thresholds used by StockAnalyzer.
    """
    StockAnalyzer.DEFAULTS["RSI"]["oversold"] = oversold
    StockAnalyzer.DEFAULTS["RSI"]["overbought"] = overbought


def get_weight_profiles() -> dict:
    """
    Predefined indicator weight profiles for experimentation.
    Keys are profile names, values are weight dicts for *_signal columns.
    """
    return {
        "equal": {
            "RSI_signal": 1.0,
            "MACD_signal": 1.0,
            "SMA_signal": 1.0,
            "EMA_signal": 1.0,
            "BB_signal": 1.0,
        },
        "trend_heavy": {
            "RSI_signal": 0.5,
            "MACD_signal": 1.0,
            "SMA_signal": 1.5,
            "EMA_signal": 1.5,
            "BB_signal": 0.5,
        },
        "momentum_heavy": {
            "RSI_signal": 1.5,
            "MACD_signal": 1.5,
            "SMA_signal": 1.0,
            "EMA_signal": 1.0,
            "BB_signal": 0.5,
        },
    }


def run_single_backtest(
    symbol: str,
    oversold: int,
    overbought: int,
    weights: dict,
    start_date: str,
    end_date: str | None,
) -> dict:
    """
    Run one backtest for a given (symbol, RSI thresholds, weight profile)
    and return stats as a flat dict.
    """
    # 1) Apply RSI thresholds
    set_rsi_thresholds(oversold, overbought)

    # 2) Fetch historical data
    fetcher = DataFetcher()
    data_map = fetcher.fetch_multiple([symbol])
    if symbol not in data_map:
        raise ValueError(f"No data returned for symbol '{symbol}'")
    df = data_map[symbol]

    # 3) Slice to desired date range
    df = df.loc[start_date:end_date]

    # 4) Compute indicators
    ti = TechnicalIndicators(df, symbol=symbol)
    ti.calculate_all_indicators()

    # 5) Set up decision engine + backtester
    initial_equity = 10_000.0

    engine = DecisionEngine(
        symbol=symbol,
        account_equity=initial_equity,
        indicator_weights=weights,
        # you can also tune max_risk_per_trade / high_vol_threshold later
        # max_risk_per_trade=0.01,
        # high_vol_threshold=0.04,
    )

    bt = Backtester(
        initial_equity=initial_equity,
        trading_cost_pct=0.0005,   # 0.05% per side
        # sl_pct=0.03,
        # tp_pct=0.06,
    )

    results = bt.run(ti.data, engine=engine)
    stats = dict(results["stats"])

    # Add context columns
    stats["symbol"] = symbol
    stats["oversold"] = oversold
    stats["overbought"] = overbought

    return stats


# ---------------------------------------------------------------------------
# Single sanity test (keep this as your quick "smoke test")
# ---------------------------------------------------------------------------

def single_test() -> None:
    """
    Quick end-to-end sanity check for a single symbol and one config.
    This is the one you already had; just reusing helpers now.
    """
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = None

    # Fetch data
    fetcher = DataFetcher()
    data_map = fetcher.fetch_multiple([symbol])
    if symbol not in data_map:
        raise ValueError(f"No data returned for symbol '{symbol}'")
    df = data_map[symbol]

    if start_date or end_date:
        df = df.loc[start_date:end_date]

    # Compute indicators
    ti = TechnicalIndicators(df, symbol=symbol)
    ti.calculate_all_indicators()

    # Choose a weight profile (you can change this later to a tuned profile)
    initial_equity = 10_000.0

    # Uses DEFAULT_WEIGHTS_MOMENTUM_HEAVY and RSI 25/75 by default
    engine = DecisionEngine(
        symbol=symbol,
        account_equity=initial_equity,
        # no indicator_weights arg → uses tuned default
    )

    bt = Backtester(
        initial_equity=initial_equity,
        trading_cost_pct=0.0005,
    )

    results = bt.run(ti.data, engine=engine)

    print("\n=== Stats ===")
    for k, v in results["stats"].items():
        print(f"{k}: {v}")

    print("\n=== Sample trades ===")
    print(results["trades"].head())

    print("\n=== Equity curve tail ===")
    print(results["equity_curve"].tail())


# ---------------------------------------------------------------------------
# Grid search tuner
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run a small grid search over RSI thresholds and weight profiles
    for one or more symbols and print the top-performing configs.
    """
    symbols = ["AAPL"]  # later you can add ["AAPL", "MSFT", "SPY", ...]
    oversold_values = [20, 25, 30]
    overbought_values = [70, 75, 80]
    weight_profiles = get_weight_profiles()

    start_date = "2020-01-01"
    end_date = None

    rows: list[dict] = []

    for symbol in symbols:
        for (oversold, overbought), (w_name, weights) in itertools.product(
            itertools.product(oversold_values, overbought_values),
            weight_profiles.items(),
        ):
            try:
                stats = run_single_backtest(
                    symbol=symbol,
                    oversold=oversold,
                    overbought=overbought,
                    weights=weights,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as e:
                print(f"[ERROR] {symbol} RSI=({oversold}/{overbought}) weights={w_name}: {e}")
                continue

            row = {
                "symbol": symbol,
                "oversold": oversold,
                "overbought": overbought,
                "weights_profile": w_name,
            }
            row.update(stats)
            rows.append(row)

            print(
                f"{symbol} RSI=({oversold}/{overbought}) weights={w_name} -> "
                f"ret={stats.get('total_return_pct', 0):.3f}% | "
                f"dd={stats.get('max_drawdown_pct', 0):.3f}% | "
                f"trades={stats.get('num_trades', 0):.0f}"
            )

    if not rows:
        print("No successful runs.")
        return

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["total_return_pct", "max_drawdown_pct"],
        ascending=[False, True],  # max return, then min drawdown
    )

    print("\n=== Top configs ===")
    print(results_df.head(15).to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the tuner by default
    main()

    # Uncomment this if you want to run only the single sanity test instead:
    # single_test()
