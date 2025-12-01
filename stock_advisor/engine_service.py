# stock_advisor/engine_service.py

from typing import Optional, Dict, Any

import pandas as pd

from stock_advisor.data_fetcher import DataFetcher
from stock_advisor.technical_indicators import TechnicalIndicators
from stock_advisor.decision_engine import DecisionEngine
from stock_advisor.backtester import Backtester
from stock_advisor.analyzer import StockAnalyzer


def run_latest_decision(
    symbol: str,
    lookback_days: int = 252,
    account_equity: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Get a single latest decision for a symbol.

    - Fetch recent data
    - Compute indicators
    - Run StockAnalyzer to add *_signal columns
    - Run DecisionEngine.run() on that enriched data
    - Return a JSON-serializable dict including an indicator snapshot
    """
    symbol = symbol.strip().upper()

    # 1) Fetch OHLCV + indicators
    full_df = _fetch_indicator_data(symbol, start_date=None, end_date=None)

    # 2) Limit to last N days if requested
    if lookback_days is not None and len(full_df) > lookback_days:
        df = full_df.iloc[-lookback_days:].copy()
    else:
        df = full_df.copy()

    # 3) Run analyzer here so we definitely have *_signal columns
    analyzer = StockAnalyzer(df.copy(), symbol=symbol)
    analyzer.analyze_indicators()
    analyzer.generate_signal()
    enriched_df = analyzer.data

    # 4) Run decision engine on the enriched data
    engine = DecisionEngine(
        symbol=symbol,
        account_equity=account_equity,
        # uses your DEFAULT_WEIGHTS_MOMENTUM_HEAVY by default
    )
    decision = engine.run(enriched_df)

    # 5) Build snapshot from the enriched last row
    last_row = enriched_df.iloc[-1]
    as_of = enriched_df.index[-1]

    # --- Build history for chart (recent window) ---
    history_rows = []
    # Use the same lookback window or just last 252 rows
    history_df = enriched_df.iloc[-min(len(enriched_df), lookback_days or 252):]

    for ts, row in history_df.iterrows():
        item: Dict[str, Any] = {
            "time": str(ts),
            "Close": float(row["Close"]) if "Close" in row and pd.notna(row["Close"]) else None,
            "final_signal": int(row["final_signal"]) if "final_signal" in row and pd.notna(row["final_signal"]) else 0,
        }
        # You *could* add more indicator series later if you want.
        history_rows.append(item)
    # --- Indicator snapshot (raw values) ---
    indicator_snapshot: Dict[str, float] = {}

    # Use the same names as your analyzer DEFAULTS so it's always in sync
    # RSI
    if "RSI" in last_row.index:
        indicator_snapshot["RSI"] = float(last_row["RSI"])

    # MACD + signal line
    macd_col = StockAnalyzer.DEFAULTS["MACD"]["macd"]
    sig_col = StockAnalyzer.DEFAULTS["MACD"]["signal"]
    if macd_col in last_row.index:
        indicator_snapshot["MACD"] = float(last_row[macd_col])
    if sig_col in last_row.index:
        indicator_snapshot["Signal"] = float(last_row[sig_col])

    # SMA & EMA
    sma_s, sma_l = StockAnalyzer.DEFAULTS["SMA"]["short"], StockAnalyzer.DEFAULTS["SMA"]["long"]
    ema_s, ema_l = StockAnalyzer.DEFAULTS["EMA"]["short"], StockAnalyzer.DEFAULTS["EMA"]["long"]

    if sma_s in last_row.index:
        indicator_snapshot[sma_s] = float(last_row[sma_s])
    if sma_l in last_row.index:
        indicator_snapshot[sma_l] = float(last_row[sma_l])
    if ema_s in last_row.index:
        indicator_snapshot[ema_s] = float(last_row[ema_s])
    if ema_l in last_row.index:
        indicator_snapshot[ema_l] = float(last_row[ema_l])

    # Bollinger Bands
    bb_u = StockAnalyzer.DEFAULTS["BB"]["upper"]
    bb_l = StockAnalyzer.DEFAULTS["BB"]["lower"]
    if bb_u in last_row.index:
        indicator_snapshot[bb_u] = float(last_row[bb_u])
    if bb_l in last_row.index:
        indicator_snapshot[bb_l] = float(last_row[bb_l])

    # --- Signal snapshot (bullish/bearish/neutral) ---
    signal_snapshot: Dict[str, int] = {}
    for col in enriched_df.columns:
        if col.endswith("_signal"):
            try:
                signal_snapshot[col] = int(last_row[col])
            except (TypeError, ValueError):
                pass

    decision_out: Dict[str, Any] = {
        "symbol": symbol,
        "as_of": str(as_of),
        "last_close": float(last_row["Close"]) if "Close" in last_row else None,
        "indicator_snapshot": indicator_snapshot,
        "signal_snapshot": signal_snapshot,
        "history": history_rows,   
        **decision,
    }

    return decision_out


def _fetch_indicator_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "10y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data, optionally slice by date, and compute all indicators.
    """
    sym = symbol.strip().upper()

    fetcher = DataFetcher()
    data_map = fetcher.fetch_multiple([sym], period=period, interval=interval)
    if sym not in data_map:
        raise ValueError(f"No data returned for symbol '{sym}'")

    df = data_map[sym]
    if start_date or end_date:
        df = df.loc[start_date:end_date]

    ti = TechnicalIndicators(df, symbol=sym)
    ti.calculate_all_indicators()
    return ti.data


def run_backtest(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    account_equity: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Run a simple backtest for a symbol and return summary stats plus a small
    sample of trades for display.
    """
    df = _fetch_indicator_data(symbol, start_date=start_date, end_date=end_date)

    engine = DecisionEngine(
        symbol=symbol,
        account_equity=account_equity,
    )

    bt = Backtester(
        initial_equity=account_equity,
        trading_cost_pct=0.0005,
    )

    results = bt.run(df, engine=engine)
    equity_curve = results["equity_curve"]
    stats = results["stats"]
    trades_df = results["trades"].copy()

    equity_series = []
    for ts, value in equity_curve.items():
        equity_series.append({
            "time": str(ts),
            "equity": float(value),
        })

    # Make trades JSON-friendly
    trades_sample = []
    if not trades_df.empty:
        trades_df["time"] = trades_df["time"].astype(str)
        trades_sample = trades_df.head(10).to_dict(orient="records")
        
    out: Dict[str, Any] = {
        "symbol": symbol,
        "stats": stats,
        "num_data_points": int(len(df)),
        "trades_sample": trades_sample,
        "equity_series": equity_series, 
    }

    return out