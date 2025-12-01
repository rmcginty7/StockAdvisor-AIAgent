# StockAdvisor AI Agent 

An end-to-end **educational trading assistant** that:

- Fetches real market data with `yfinance`
- Computes technical indicators with TA-Lib (RSI, MACD, SMA/EMA, Bollinger Bands)
- Interprets indicators into Buy / Hold / Sell signals
- Applies **confidence scoring** and **volatility-based risk sizing**
- Runs **backtests** with stop-loss / take-profit and trading costs
- Exposes everything through a **Flask web UI** with:
  - Decision card
  - Indicator snapshot
  - Backtest summary

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only**.  
> It does **not** constitute financial advice or a recommendation to trade.

---

## Features

### Technical Indicators

Implemented in `stock_advisor/technical_indicators.py` using TA-Lib:

- **RSI** (Relative Strength Index)
- **MACD** + signal line
- **SMA** – e.g. 20-period, 50-period
- **EMA** – e.g. 12-period, 26-period
- **Bollinger Bands** – upper, middle, lower

These are appended as columns to the price DataFrame.  

---

### StockAnalyzer (`analyzer.py`)

The `StockAnalyzer`:

- Takes a price + indicators DataFrame
- Builds per-indicator signals:

  - `RSI_signal`
  - `MACD_signal`
  - `SMA_signal`
  - `EMA_signal`
  - `BB_signal`

- Uses values in `{ -1, 0, 1 }`:

  - `1`  = bullish (Buy)
  - `0`  = neutral
  - `-1` = bearish (Sell)

- Combines signals into:

  - `signal_sum` (weighted sum of signals)
  - `final_signal` (1 / 0 / -1)
  - `final_label` ("BUY" / "HOLD" / "SELL")

It also stores a small summary dict (`signals_summary`) and provides:

- `get_latest_signal()`
- `summarize_results()`

---

### DecisionEngine (`decision_engine.py`)

Wraps `StockAnalyzer` to produce a full decision object.

For a given symbol, it outputs:

- `action`: `"BUY"`, `"HOLD"`, or `"SELL"`
- `signal`: `1`, `0`, or `-1`
- `confidence`: `0–1` score based on indicator agreement
- `position_size_value`: suggested dollar allocation for this trade
- `volatility`: rolling volatility estimate (e.g. 20-day std of returns)
- `sentiment`: optional sentiment input (if provided)
- `reasoning`: human-readable explanation describing which indicators are bullish/bearish and why

Key pieces:

- **Confidence scoring** based on normalized `signal_sum` and optional sentiment
- **Volatility-aware position sizing** using:
  - account equity
  - max risk per trade (e.g. 1%)
  - volatility adjustment
  - confidence score

---

### Backtester (`backtester.py`)

Simulates a simple long-only strategy using the `DecisionEngine`:

- Iterates over historical bars
- At each step:
  - Calls `DecisionEngine.run(...)`
  - Executes buys/sells based on the engine’s `action`
  - Applies:
    - Trading cost as a % of notional
    - Stop-loss (`sl_pct`)
    - Take-profit (`tp_pct`)

Outputs:

- `equity_curve`: `pd.Series` of portfolio value over time
- `trades`: `pd.DataFrame` of executed trades
  - time, action (BUY / SELL / STOP_LOSS / TAKE_PROFIT), price, shares, cash_after
- `stats` dictionary:
  - `start_equity`
  - `end_equity`
  - `total_return_pct`
  - `max_drawdown_pct`
  - `num_trades`
  - `win_rate_pct`

---

### Orchestration (`engine_service.py`)

Convenience helpers that connect the components for the web app:

- `run_latest_decision(symbol: str, lookback_days: int, account_equity: float)`  
  - Fetches price data, computes indicators, runs analyzer + decision engine
  - Returns:
    - Decision object
    - Indicator snapshot
    - Signal snapshot
    - Price history for plotting (recent window)

- `run_backtest(symbol: str, start_date: str, account_equity: float = 10_000.0)`  
  - Fetches and prepares data
  - Runs backtest
  - Returns:
    - Stats
    - Sample trades
    - Equity series for plotting

---

### Web UI (`app.py` + `templates/` + `static/`)

Flask app exposing:

- **`GET /`** – main dashboard
  - Analyze a symbol:
    - Input: ticker, lookback days
    - Displays:
      - Latest decision (action + confidence + position size)
      - Indicator snapshot (RSI, MACD, SMA/EMA, Bollinger, signal health)
      - Reasoning text from the decision engine
      - (Optionally) recent price chart + signal markers
  - Quick backtest:
    - Input: ticker, start date
    - Displays:
      - Total return %, max drawdown %
      - Number of trades, win rate %
      - Sample trade table
      - (Optionally) equity curve chart

- **`GET /model`** – model summary page
  - Explains:
    - Which indicators are used
    - How signals are generated
    - How confidence is calculated
    - How volatility affects position sizing
    - How backtesting works
  - Includes a clear disclaimer

---

## Project Structure

Rough outline (may be slightly different in your repo):

```text
StockAdvisor-AIAgent/
  app.py                      # Flask entrypoint

  stock_advisor/
    __init__.py
    data_fetcher.py           # yfinance-based fetcher
    technical_indicators.py   # TA-Lib wrapper, computes indicators
    analyzer.py               # StockAnalyzer, builds per-indicator + final signals
    decision_engine.py        # DecisionEngine, adds confidence + risk logic
    backtester.py             # Backtester, runs historical simulations
    engine_service.py         # High-level run_latest_decision, run_backtest helpers
    # (plus any test modules, configs, etc.)

  templates/
    index.html                # Main dashboard
    model.html                # Model summary / documentation page

  static/
    styles.css                # UI styling (cards, layout, buttons)

  tests/
    test_*.py                 # pytest-based unit tests 

  README.md
  requirements.txt
```

---

## Getting started

1. Clone the repo
- git clone git clone https://github.com/rmcginty7/StockAdvisor-AIAgent.git
- cd StockAdvisor-AIAgent

2. Create a virtual environement (Optional but recommended)
- python3 -m venv .venv
- source .venv/bin/activate

3. Install Dependencies
- pip install -r requirements.txt

4. Run the app
- python app.py

---

