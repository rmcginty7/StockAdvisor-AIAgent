
import pandas as pd
import pytest

from data_fetcher import DataFetcher


# ---------- Unit tests for validate_symbol ----------

def test_validate_symbol_valid_uppercases_and_strips():
    f = DataFetcher()
    assert f.validate_symbol("  aapl ") == "AAPL"  # trims + uppercases


def test_validate_symbol_rejects_non_string():
    f = DataFetcher()
    with pytest.raises(ValueError):
        f.validate_symbol(123)  # not a string


def test_validate_symbol_rejects_bad_chars():
    f = DataFetcher()
    with pytest.raises(ValueError):
        f.validate_symbol("AAPL!")  # punctuation not allowed


# ---------- Helper to build a fake yfinance.download response ----------

def _make_multi_ticker_df(symbols):
    """
    Build a DataFrame shaped like yfinance.download(..., group_by='ticker')
    for the given list of symbols. Returns a wide DataFrame with a MultiIndex
    on columns: level 0 = symbol, level 1 = ['Open','High','Low','Close','Volume'].
    """
    frames = []
    for sym in symbols:
        idx = pd.to_datetime(
            ["2024-01-02 09:30", "2024-01-02 09:31"], utc=True
        )
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.5],
                "High": [101.0, 101.2],
                "Low": [99.8, 100.2],
                "Close": [100.8, 101.0],
                "Volume": [1_000, 900],
            },
            index=idx,
        )
        # give this small frame a top-level column of the symbol
        df.columns = pd.MultiIndex.from_product([[sym], df.columns])
        frames.append(df)

    if not frames:
        # yfinance returns an empty DataFrame when nothing comes back
        return pd.DataFrame()

    # concat all symbols side-by-side on columns
    return pd.concat(frames, axis=1)


# ---------- Integration-ish tests for fetch_multiple (with monkeypatch) ----------

def test_fetch_multiple_happy_path(monkeypatch):
    """Two valid tickers come back, cleaned into separate frames."""
    def fake_download(symbols, **kwargs):
        assert kwargs.get("group_by") == "ticker"
        # pretend yfinance only returns AAPL and MSFT successfully
        return _make_multi_ticker_df(["AAPL", "MSFT"])

    # swap yfinance.download with our fake
    import yfinance as yf
    monkeypatch.setattr(yf, "download", fake_download)

    f = DataFetcher()
    out = f.fetch_multiple(["AAPL", "MSFT"], period="1mo", interval="1d")

    assert set(out.keys()) == {"AAPL", "MSFT"}
    # verify columns are normalized and timestamp is tz-aware UTC
    for sym, df in out.items():
        assert df.index.tz is not None
        assert df.index.tz.key == "UTC"
        expected_cols = {"open", "high", "low", "close", "volume", "symbol", "asof_utc"}
        assert expected_cols.issubset(set(df.columns))
        assert (df["symbol"] == sym).all()


def test_fetch_multiple_skips_missing_symbol(monkeypatch, capsys):
    """If provider omits a symbol, it should be skipped (no KeyError)."""
    def fake_download(symbols, **kwargs):
        # simulate provider returning only AAPL; MSFT missing
        return _make_multi_ticker_df(["AAPL"])

    import yfinance as yf
    monkeypatch.setattr(yf, "download", fake_download)

    f = DataFetcher()
    out = f.fetch_multiple(["AAPL", "MSFT"], period="1mo", interval="1d")

    # only AAPL present
    assert list(out.keys()) == ["AAPL"]
    # optional: check the warning print happened
    captured = capsys.readouterr()
    assert "No data found for MSFT" in captured.out


def test_fetch_multiple_rejects_empty_list():
    f = DataFetcher()
    with pytest.raises(ValueError):
        f.fetch_multiple([], period="1mo", interval="1d")
