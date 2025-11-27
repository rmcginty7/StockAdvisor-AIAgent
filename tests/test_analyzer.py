import sys
from pathlib import Path

# --- Make project root importable (same pattern as your other tests) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from analyzer import StockAnalyzer as Analyzer


@pytest.fixture
def sample_df():
    """
    Small, controlled DataFrame that mimics the output of TechnicalIndicators.
    3 rows with indicators chosen so we can predict the signals exactly.
    """
    idx = pd.date_range("2024-01-01", periods=3, freq="D")

    data = {
        # Prices
        "Close":    [100.0, 103.0, 102.0],

        # RSI:
        #  row 0: 25 -> oversold (bullish)
        #  row 1: 75 -> overbought (bearish)
        #  row 2: 50 -> neutral
        "RSI":      [25.0, 75.0, 50.0],

        # MACD vs Signal:
        #  row 0: 0.5 > 0.3 -> bullish
        #  row 1: -0.2 < 0.1 -> bearish
        #  row 2: 0.1 < 0.2 -> bearish
        "MACD":     [0.5, -0.2, 0.1],
        "Signal":   [0.3,  0.1, 0.2],

        # SMA 20 vs SMA 50:
        #  row 0: 98  < 99   -> bearish
        #  row 1: 99  < 99.5 -> bearish
        #  row 2: 100 > 99.8 -> bullish
        "SMA_20":   [98.0, 99.0, 100.0],
        "SMA_50":   [99.0, 99.5, 99.8],

        # EMA 12 vs EMA 26 (we'll mirror SMA behavior for simplicity)
        "EMA_12":   [98.0, 99.0, 100.0],
        "EMA_26":   [99.0, 99.5, 99.8],

        # Bollinger Bands (price always between bands here -> neutral)
        "BB_upper": [105.0, 107.0, 106.0],
        "BB_lower": [95.0,  97.0,  96.0],
    }

    df = pd.DataFrame(data, index=idx)
    return df


def test_analyze_indicators_creates_signal_columns(sample_df):
    """
    Ensure analyze_indicators:
      - creates *_signal columns
      - applies the expected rules for RSI, MACD, SMA, EMA, BB.
    """
    an = Analyzer(sample_df, symbol="AAPL")
    df = an.analyze_indicators()

    # Check that the new signal columns exist
    for col in ["RSI_signal", "MACD_signal", "SMA_signal", "EMA_signal", "BB_signal"]:
        assert col in df.columns

    i0, i1, i2 = df.index[0], df.index[1], df.index[2]

    # --- RSI ---
    assert df.loc[i0, "RSI_signal"] == 1   # RSI 25 < 30 -> bullish
    assert df.loc[i1, "RSI_signal"] == -1  # RSI 75 > 70 -> bearish
    assert df.loc[i2, "RSI_signal"] == 0   # RSI 50 between 30 and 70 -> neutral

    # --- MACD ---
    assert df.loc[i0, "MACD_signal"] == 1   # 0.5 > 0.3
    assert df.loc[i1, "MACD_signal"] == -1  # -0.2 < 0.1
    assert df.loc[i2, "MACD_signal"] == -1  # 0.1 < 0.2

    # --- SMA ---
    assert df.loc[i0, "SMA_signal"] == -1  # 98 < 99
    assert df.loc[i1, "SMA_signal"] == -1  # 99 < 99.5
    assert df.loc[i2, "SMA_signal"] == 1   # 100 > 99.8

    # --- EMA ---
    assert df.loc[i0, "EMA_signal"] == -1
    assert df.loc[i1, "EMA_signal"] == -1
    assert df.loc[i2, "EMA_signal"] == 1

    # --- Bollinger Bands (always neutral in this sample) ---
    assert df["BB_signal"].tolist() == [0, 0, 0]


def test_generate_signal_combines_signals(sample_df):
    """
    After analyze_indicators, generate_signal should:
      - create signal_sum
      - create final_signal (1, 0, -1)
      - create final_label (BUY/SELL/HOLD)
      - compute the correct latest signal.
    """
    an = Analyzer(sample_df, symbol="AAPL")
    df = an.analyze_indicators()
    df = an.generate_signal()

    # Columns exist
    assert "signal_sum" in df.columns
    assert "final_signal" in df.columns
    assert "final_label" in df.columns

    i0, i1, i2 = df.index[0], df.index[1], df.index[2]

    # Manually compute expected sums for row 0, 1, 2:
    # Row 0:
    #   RSI_signal = 1
    #   MACD_signal = 1
    #   SMA_signal = -1
    #   EMA_signal = -1
    #   BB_signal = 0
    #   sum = 1 + 1 - 1 - 1 + 0 = 0 -> HOLD
    assert df.loc[i0, "signal_sum"] == 0
    assert df.loc[i0, "final_signal"] == 0
    assert df.loc[i0, "final_label"] == "HOLD"

    # Row 1:
    #   RSI_signal = -1
    #   MACD_signal = -1
    #   SMA_signal = -1
    #   EMA_signal = -1
    #   BB_signal = 0
    #   sum = -4 -> SELL
    assert df.loc[i1, "signal_sum"] == -4
    assert df.loc[i1, "final_signal"] == -1
    assert df.loc[i1, "final_label"] == "SELL"

    # Row 2:
    #   RSI_signal = 0
    #   MACD_signal = -1
    #   SMA_signal = 1
    #   EMA_signal = 1
    #   BB_signal = 0
    #   sum = 1 -> BUY
    assert df.loc[i2, "signal_sum"] == 1
    assert df.loc[i2, "final_signal"] == 1
    assert df.loc[i2, "final_label"] == "BUY"

    # Check get_latest_signal summary matches the last row
    latest = an.get_latest_signal()
    assert latest["symbol"] == "AAPL"
    assert latest["latest_date"] == i2
    assert latest["latest_signal"] == 1
    assert latest["latest_label"] == "BUY"
    # contributors should be the *_signal columns
    assert set(latest["contributors"]) == {
        "RSI_signal", "MACD_signal", "SMA_signal", "EMA_signal", "BB_signal"
    }


def test_generate_signal_raises_if_no_indicator_signals(sample_df):
    """
    If analyze_indicators() is not called first, generate_signal()
    should raise a clear error because no *_signal columns exist.
    """
    an = Analyzer(sample_df, symbol="AAPL")

    with pytest.raises(ValueError) as excinfo:
        an.generate_signal()

    assert "No *_signal columns found" in str(excinfo.value)
