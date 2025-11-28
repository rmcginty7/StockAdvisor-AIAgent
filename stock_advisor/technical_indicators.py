from typing import Optional

import numpy as np
import pandas as pd

class TechnicalIndicators:

    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None):
        """Prepare a price DataFrame for technical indicator calculations."""
        self.data = df.copy()
        self.symbol = symbol.strip().upper() if isinstance(symbol, str) else None

        if isinstance(self.data.columns, pd.MultiIndex):
            top_level = self.data.columns.get_level_values(0).unique()

            if len(top_level) > 1:
                if not self.symbol:
                    choices = ", ".join(map(str, top_level))
                    raise ValueError(
                        "Multiple symbols detected in DataFrame. "
                        "Pass a single ticker via the 'symbol' argument "
                        f"or pre-select the desired columns. Available: {choices}"
                    )
                if self.symbol not in top_level:
                    available = ", ".join(map(str, top_level))
                    raise ValueError(
                        f"Requested symbol '{self.symbol}' not present in DataFrame. "
                        f"Available symbols: {available}"
                    )
                # Select the requested ticker level, dropping the symbol axis
                self.data = self.data.xs(self.symbol, axis=1, level=0)
            else:
                # Collapse the single-symbol MultiIndex
                self.symbol = self.symbol or str(top_level[0])
                self.data = self.data.xs(top_level[0], axis=1, level=0)

        # Normalize column names to lowercase strings for internal use
        self.data.columns = [str(col).strip().lower() for col in self.data.columns]
        # Preserve common OHLCV names in title case for downstream analyzers
        for base in ["open", "high", "low", "close", "volume"]:
            if base in self.data.columns:
                self.data[base.capitalize()] = self.data[base]


    # Add Simple Moving Average (SMA) calculation
    def calculate_sma(self, period=14):

        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")

        self.data[f"SMA_{period}"] = self.data["close"].rolling(window=period, min_periods=period).mean()
        return self

    # Add Relative Strength Index (RSI) calculation
    def calculate_rsi(self, period=14):
        
        # Uses only the close column 
        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")

        delta = self.data["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace({0: np.nan})
        rsi = 100 - (100 / (1 + rs))

        self.data[f"RSI_{period}"] = rsi
        # Analyzer expects a generic RSI column
        self.data["RSI"] = rsi
        return self

    # Add Exponential Moving Average (EMA) calculation
    def calculate_ema(self, period=14):

        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")

        self.data[f"EMA_{period}"] = self.data["close"].ewm(span=period, adjust=False).mean()
        return self
    
    # Add Moving Average Convergence Divergence (MACD) calculation
    def calculate_macd(self):
        
        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")

        fast_ema = self.data["close"].ewm(span=12, adjust=False).mean()
        slow_ema = self.data["close"].ewm(span=26, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        self.data["MACD"] = macd_line
        self.data["MACD_signal"] = macd_signal
        # Analyzer expects the signal column named 'Signal'
        self.data["Signal"] = macd_signal
        self.data["MACD_hist"] = macd_hist
        return self

    # Add Bollinger Bands calulation
    def calculate_bb(self, period=14):

        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")

        rolling_mean = self.data["close"].rolling(window=period, min_periods=period).mean()
        rolling_std = self.data["close"].rolling(window=period, min_periods=period).std()

        self.data["BB_upper"] = rolling_mean + (2 * rolling_std)
        self.data["BB_middle"] = rolling_mean
        self.data["BB_lower"] = rolling_mean - (2 * rolling_std)
        return self
    
    def calculate_all_indicators(self) -> None:
        """
        Convenience helper to compute all the indicators needed
        by the Analyzer / DecisionEngine.
        """

        (
            self.calculate_rsi(14)
            .calculate_macd()
            .calculate_sma(20)
            .calculate_sma(50)
            .calculate_ema(12)
            .calculate_ema(26)
            .calculate_bb(20)
        )


# Some testing
if __name__ == "__main__":
    import yfinance as yf

    market_data = yf.download(
        ["AAPL", "MSFT"],
        period="1mo",
        interval="1d",
        group_by="ticker",
        progress=False,
    )
    ti = TechnicalIndicators(market_data, symbol="AAPL")

    enriched = (
        ti.calculate_sma(14)
        .calculate_rsi(14)
        .calculate_ema(14)
        .calculate_macd()
        .calculate_bb(14)
        .data
    )
    print(
        enriched[
            [
                "close",
                "SMA_14",
                "RSI_14",
                "EMA_14",
                "MACD",
                "MACD_signal",
                "MACD_hist",
                "BB_upper",
                "BB_middle",
                "BB_lower",
            ]
        ].tail()
    )
