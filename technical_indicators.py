from typing import Optional

import talib
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


    # Add Simple Moving Average (SMA) calculation
    def calculate_sma(self, period=14):

        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")

        self.data[f"SMA_{period}"] = talib.SMA(self.data["close"], timeperiod=period)
        return self

    # Add Relative Strength Index (RSI) calculation
    def calculate_rsi(self, period=14):
        
        # Uses only the close column 
        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        self.data[f"RSI_{period}"] = talib.RSI(self.data["close"], timeperiod=period)
        return self

    # Add Exponential Moving Average (EMA) calculation
    def calculate_ema(self, period=14):

        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        self.data[f"EMA_{period}"] = talib.EMA(self.data["close"], timeperiod=period)
        return self
    
    # Add Moving Average Convergence Divergence (MACD) calculation
    def calculate_macd(self):
        
        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        self.data["MACD"], self.data["MACD_signal"], self.data["MACD_hist"] = talib.MACD(
            self.data["close"],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        return self

    # Add Bollinger Bands calulation
    def calculate_bb(self, period=14):

        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain 'close' column")

        upper, middle, lower = talib.BBANDS(
            self.data["close"], 
            timeperiod=period,
            nbdevup=2,          # Default standard deviation for upper band
            nbdevdn=2,          # Default standard deviation for lower band
            matype=0            # Simple Moving Average
            )
        self.data["BB_upper"] = upper
        self.data["BB_middle"] = middle
        self.data["BB_lower"] = lower 
        return self

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
