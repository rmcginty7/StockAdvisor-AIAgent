try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9 fallback
    ZoneInfo = None

import pandas as pd
import yfinance as yf


class DataFetcher:
    """Fetch and normalize market data via yfinance."""

    def __init__(self):
        pass

    def validate_symbol(self, symbol: str) -> str:
        """Return an uppercase ticker symbol after basic validation."""

        if not isinstance(symbol, str):
            raise ValueError("Ticker symbol must be a string")
        
        symbol = symbol.strip().upper()

        if not symbol or not symbol.replace("-", "").isalnum():
            raise ValueError(f"Invalid symbol format: {symbol}")
        return symbol

    def fetch_multiple(self, symbols: list[str], period="1mo", interval="1d") -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols at once.

        Returns a dictionary mapping symbol -> cleaned DataFrame.
        """
        if not symbols or not isinstance(symbols, list):
            raise ValueError("symbols must be a non-empty list of strings")

        validated = [self.validate_symbol(s) for s in symbols]

        try:
            data = yf.download(validated, period=period, interval=interval, group_by="ticker", progress=False)
        except Exception as exc:
            raise ConnectionError(f"Error fetching data: {exc}") from exc

        cleaned: dict[str, pd.DataFrame] = {}
        for sym in validated:
            if sym not in data:
                print(f"No data found for {sym}")
                continue
            cleaned[sym] = self._clean_data(data[sym], sym)

        return cleaned

    def _clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize columns and timestamps on a per-symbol DataFrame."""
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        
        utc_tz = ZoneInfo("UTC") if ZoneInfo else "UTC"

        #Time zone conversion
        if df.index.tz is None:
            df.index = df.index.tz_localize(utc_tz)
        else:
            df.index = df.index.tz_convert(utc_tz)

        #Adding meta data columns, 'asof_utc' when data was fetched
        df["symbol"] = symbol
        df["asof_utc"] = pd.Timestamp.now(tz=utc_tz)
        return df


#Testing the Class
if __name__ == "__main__":
    fetcher = DataFetcher()
    symbols = ["AAPL", "MSFT", "GOOG"]
    results = fetcher.fetch_multiple(symbols, period="1mo", interval="1d")

    for sym, frame in results.items():
        print(f"\n--- {sym} ---")
        print(frame.head())
