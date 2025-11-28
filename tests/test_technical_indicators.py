import pandas as pd
import pytest 
import numpy as np

from stock_advisor.technical_indicators import TechnicalIndicators

@pytest.fixture
def sample_data():
    """Create a deterministic OHLCV DataFrame for testing."""
    idx = pd.date_range(start="2024-01-01", periods=30, freq="D")
    base = np.linspace(100, 129, num=30)
    data = pd.DataFrame(
        {
            "Open": base + 1,
            "High": base + 2,
            "Low": base - 2,
            "Close": base,
            "Volume": np.arange(1_000, 1_000 + len(idx)),
        },
        index=idx,
    )
    return data

# ---------- Unit tests for TechnicalIndicators class ----------

def test_calculate_sma(sample_data):

    ti = TechnicalIndicators(sample_data)
    ti.calculate_sma(10)
    assert f"SMA_10" in ti.data.columns, "SMA column missing"
    assert len(ti.data) == len(sample_data), "Length mismatch"
    assert pd.api.types.is_numeric_dtype(ti.data[f"SMA_10"]), "SMA column not numeric"

def test_calculate_rsi(sample_data):

    ti = TechnicalIndicators(sample_data)
    ti.calculate_rsi(14)
    assert "RSI_14" in ti.data.columns, "RSI column missing"
    assert ti.data["RSI_14"].dropna().between(0, 100).all(), "RSI values out of range"

def test_caculate_macd(sample_data):
    """Test that MACD-related columns are added."""
    ti = TechnicalIndicators(sample_data)
    ti.calculate_macd()
    for col in ["MACD", "MACD_signal", "MACD_hist"]:
        assert col in ti.data.columns, f"{col} missing from result"

def test_calculate_ema(sample_data):

    ti = TechnicalIndicators(sample_data)
    ti.calculate_ema(10)
    assert f"EMA_10" in ti.data.columns, "EMA column missing"
    assert len(ti.data) == len(sample_data), "Length mismatch"
    assert pd.api.types.is_numeric_dtype(ti.data[f"EMA_10"]), "EMA column not numeric"

def test_calculate_bb(sample_data):
    
    ti = TechnicalIndicators(sample_data)
    ti.calculate_bb(14)

    for col in ["BB_upper", "BB_middle", "BB_lower"]:
        assert col in ti.data.columns, f"{col} missing from result"
        assert pd.api.types.is_numeric_dtype(ti.data[col]), f"{col} column not numeric"
    assert len(ti.data) == len(sample_data), "Length mismatch"


def test_chaining_methods(sample_data):
    """Test that multiple indicators can be chained together."""
    ti = TechnicalIndicators(sample_data)
    ti.calculate_sma(10).calculate_rsi(14).calculate_ema(10).calculate_macd().calculate_bb(14)
    assert all(col in ti.data.columns for col in ["SMA_10", "RSI_14", "MACD"]), "Chained indicators missing"


def test_empty_dataframe():
    """Ensure code gracefully handles empty input."""
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    ti = TechnicalIndicators(df)
    ti.calculate_sma(10)
    assert f"SMA_10" in ti.data.columns
    assert len(ti.data) == 0


def test_multi_symbol_requires_symbol_arg(sample_data):
    """Ensure providing multi-symbol data without a symbol raises ValueError."""
    multi = pd.concat(
        {"AAPL": sample_data, "MSFT": sample_data.rename(columns=lambda c: f"{c}_2")},
        axis=1,
    )
    with pytest.raises(ValueError):
        TechnicalIndicators(multi)
