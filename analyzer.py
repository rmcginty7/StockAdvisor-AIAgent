import pandas as pd
from typing import Dict, List, Optional

class StockAnalyzer:
    """
    Interprets technical indicators and generates unified trading signals.

    Workflow
    --------
    1) Pass a DataFrame with indicator columns (from TechnicalIndicators).
    2) Call `analyze_indicators()` to create per-indicator signals (1/0/-1).
    3) Call `generate_signal()` to combine into a single final signal per row.

    Signals
    -------
    1  = BUY  (bullish)
    0  = HOLD (neutral / no consensus)
    -1 = SELL (bearish)
    """

    DEFAULTS = {
        "RSI": {"oversold": 30, "overbought": 70},
        "SMA": {"short": "SMA_20", "long": "SMA_50"},
        "EMA": {"short": "EMA_12", "long": "EMA_26"},
        "BB": {"upper": "BB_upper", "lower": "BB_lower", "price": "Close"},
        "MACD": {"macd": "MACD", "signal": "Signal"},
    }

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        indicator_weights: Optional[Dict[str, float]] = None,
        create_labels: bool = True,
    ):
        
        """
        Parameters
        ----------
        data : pd.DataFrame
            Prices + indicators (e.g., RSI, MACD/Signal, SMA_20/50, EMA_12/26, BB_*).
        symbol : str, optional
            Ticker for reporting.
        indicator_weights : dict[str, float], optional
            Optional weighting for each *_signal column when combining, e.g.:
            {"RSI_signal": 1.0, "MACD_signal": 1.2, "SMA_signal": 0.8}
            If None, all indicators are weighted equally.
        create_labels : bool
            If True, also create human-readable labels (BUY/SELL/HOLD).
        """
         
        self.data = df.copy()
        self.symbol = (symbol or "").strip().upper() or None
        self.weights = indicator_weights or {}
        self.create_labels = create_labels
        self.signals_summary: Optional[Dict[str, object]] = None


    def analyze_indicators(self):
        """
        Create per-indicator signal columns using built-in rules.
        Produces integer columns: RSI_signal, MACD_signal, SMA_signal, EMA_signal, BB_signal.
        Only indicators whose required columns exist will be evaluated.
        Missing ones are skipped (not created).
        """

        # RSI Signal: Buy when RSI < 30, Sell when RSI > 70
        if "RSI" in self.data.columns:
            lo, hi = self.DEFAULTS["RSI"]["oversold"], self.DEFAULTS["RSI"]["overbought"]
            self.data["RSI_signal"] = 0
            self.data.loc[self.data["RSI"] < lo, "RSI_signal"] = 1      # Buy signal
            self.data.loc[self.data["RSI"] > hi, "RSI_signal"] = -1     # Sell signal

        # MACD Crossover Signal: Buy when MACD crosses above signal line, Sell when below
        macd_col = self.DEFAULTS["MACD"]["macd"]
        sig_col = self.DEFAULTS["MACD"]["signal"]
        if {macd_col, sig_col}.issubset(self.data.columns):
            self.data["MACD_signal"] = 0
            self.data.loc[self.data[macd_col] > self.data[sig_col], "MACD_signal"] = 1
            self.data.loc[self.data[macd_col] < self.data[sig_col], "MACD_signal"] = -1

        # SMA Crossover Signal: Buy when short-term SMA crosses above long-term SMA, Sell when below
        sma_s, sma_l = self.DEFAULTS["SMA"]["short"], self.DEFAULTS["SMA"]["long"]
        if {sma_s, sma_l}.issubset(self.data.columns):
            self.data["SMA_signal"] = 0
            self.data.loc[self.data[sma_s] > self.data[sma_l], "SMA_signal"] = 1 # Buy signal
            self.data.loc[self.data[sma_s] < self.data[sma_l], "SMA_signal"] = -1 # Sell signal

        # EMA Crossover Signal: Buy when short-term EMA crosses above long-term EMA, Sell when below
        ema_s, ema_l = self.DEFAULTS["EMA"]["short"], self.DEFAULTS["EMA"]["long"]
        if {ema_s, ema_l}.issubset(self.data.columns):
            self.data["EMA_signal"] = 0
            self.data.loc[self.data[ema_s] > self.data[ema_l], "EMA_signal"] = 1 # Buy signal
            self.data.loc[self.data[ema_s] < self.data[ema_l], "EMA_signal"] = -1 # Sell signal

        # BB Signal: Buy when price crosses below lower band, Sell when above upper band
        bb_u, bb_l, price = (
            self.DEFAULTS["BB"]["upper"],
            self.DEFAULTS["BB"]["lower"],
            self.DEFAULTS["BB"]["price"],
        )
        if {bb_u, bb_l, price}.issubset(self.data.columns):
            self.data["BB_signal"] = 0
            self.data.loc[self.data[price] < self.data[bb_l], "BB_signal"] = 1   # Buy signal
            self.data.loc[self.data[price] > self.data[bb_u], "BB_signal"] = -1  # Sell signal
        
        return self.data

    def generate_signal(self):

        """
        Combine all *_signal columns into a unified signal per row.

        Steps:
        - Find every column ending with '_signal'
        - Optionally apply weights
        - Sum row-wise → signal_sum
        - Map to final_signal: >0 → 1, <0 → -1, ==0 → 0
        - (Optional) Add human labels BUY/SELL/HOLD
        """

        indicator_cols: List[str] = [c for c in self.data.columns if c.endswith("_signal")]
        if not indicator_cols:
            raise ValueError("No *_signal columns found. Run analyze_indicators() first.")
        
        self.data[indicator_cols] = self.data[indicator_cols].fillna(0)

        # Weighted or unweighted sum
        if self.weights:
            # Ensure any missing weights default to 1.0
            w = {col: float(self.weights.get(col, 1.0)) for col in indicator_cols}
            weighted = sum(self.data[col] * w[col] for col in indicator_cols)
            self.data["signal_sum"] = weighted
        else:
            self.data["signal_sum"] = self.data[indicator_cols].sum(axis=1)

        # Final unified decision
        self.data["final_signal"] = 0
        self.data.loc[self.data["signal_sum"] > 0, "final_signal"] = 1
        self.data.loc[self.data["signal_sum"] < 0, "final_signal"] = -1

        if self.create_labels:
            label_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
            self.data["final_label"] = self.data["final_signal"].map(label_map)

        last = self.data.iloc[-1]
        latest_label = None
        if self.create_labels and "final_label" in self.data.columns:
            latest_label = str(last["final_label"])
        
        self.signals_summary = {
            "symbol": self.symbol,
            "latest_date": self.data.index[-1],
            "latest_signal": int(last["final_signal"]),
            "latest_label": latest_label,
            "contributors": indicator_cols,
        }

        return self.data

    def get_latest_signal(self) -> Dict[str, object]:
        """
        Return a small dict with the most recent unified signal and metatdata"""

        if self.signals_summary is None:
            self.generate_signal()
        return dict(self.signals_summary)
    
    def summarize_results(self) -> str:
        """
        One-liner summary of the latest signal
        """

        if self.signals_summary is None:
            self.generate_signal()
        s = self.signals_summary
        label = s["latest_label"] if s["latest_label"] is not None else s["latest_signal"]
        return f"[{s['symbol'] or 'UNKNOWN'} @ {s['latest_date']}] Final: {label} (from {', '.join(s['contributors'])})"
