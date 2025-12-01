from typing import Optional, Dict, Any
import pandas as pd

from stock_advisor import analyzer
from stock_advisor.analyzer import StockAnalyzer

DEFAULT_WEIGHTS_MOMENTUM_HEAVY: Dict[str, float] = {
    "RSI_signal": 1.5,
    "MACD_signal": 1.5,
    "SMA_signal": 1.0,
    "EMA_signal": 1.0,
    "BB_signal": 0.5,
}



class DecisionEngine:
    """
    Wraps StockAnalyzer to produce:
      - final trading action (buy/sell/hold)
      - confidence score
      - risk-adjusted position size
      - explanation text
    """

    def __init__(
        self,
        symbol: str,
        account_equity: float,
        max_risk_per_trade: float = 0.01,
        indicator_weights: Optional[Dict[str, float]] = None,
        high_vol_threshold: float = 0.04,
    ):
        self.symbol = symbol.strip().upper()
        self.account_equity = float(account_equity)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.indicator_weights = indicator_weights or DEFAULT_WEIGHTS_MOMENTUM_HEAVY
        self.high_vol_threshold = float(high_vol_threshold)

    def run(
        self,
        data: pd.DataFrame,
        sentiment: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for a single decision.
        """

        # 1) Analyze indicators and generate signals
        analyzer = StockAnalyzer(
            df=data,
            symbol=self.symbol,
            indicator_weights=self.indicator_weights,
        )
        analyzer.analyze_indicators()
        analyzer.generate_signal()

        df = analyzer.data
        latest = df.iloc[-1]

        signal = int(latest["final_signal"])
        signal_sum = float(latest["signal_sum"])

        # 2) Confidence scoring
        confidence = self._compute_confidence(df, signal_sum, sentiment)

        # 3) Volatility estimation
        volatility = self._estimate_volatility(df)

        # 4) Volatility regime gating:
        #    if vol is very high and confidence is low → force HOLD
        if volatility > self.high_vol_threshold and confidence < 0.7:
            signal = 0

        # 5) Position sizing (uses possibly updated signal)
        position_value = self._position_size(signal, confidence, volatility)

        # 6) Build explanation
        reasoning = self._build_explanation(
            analyzer=analyzer,
            signal=signal,
            confidence=confidence,
            volatility=volatility,
            sentiment=sentiment,
        )

        action_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
        action = action_map.get(signal, "HOLD")

        return {
            "symbol": self.symbol,
            "action": action,
            "signal": signal,
            "confidence": confidence,
            "position_size_value": position_value,
            "volatility": volatility,
            "sentiment": sentiment,
            "reasoning": reasoning,
        }

    def _compute_confidence(
        self,
        df: pd.DataFrame,
        signal_sum: float,
        sentiment: Optional[float],
    ) -> float:
        """
        Turn signal_sum (+/- N) and optional sentiment into 0–1 confidence.
        """
        indicator_cols = [c for c in df.columns if c.endswith("_signal")]
        num_indicators = len(indicator_cols)

        if num_indicators == 0:
            return 0.0

        # Normalize magnitude of signal_sum by number of indicators
        base_conf = abs(signal_sum) / num_indicators
        base_conf = max(0.0, min(1.0, base_conf))

        if sentiment is not None:
            # Clip sentiment to [-1, 1] just in case
            s = max(-1.0, min(1.0, float(sentiment)))

            # If sentiment and signal direction agree, nudge confidence up.
            # If they disagree, nudge it down.
            if signal_sum > 0 and s > 0:
                base_conf = min(1.0, base_conf + 0.2 * abs(s))
            elif signal_sum < 0 and s < 0:
                base_conf = min(1.0, base_conf + 0.2 * abs(s))
            elif signal_sum != 0 and s != 0:
                base_conf = max(0.0, base_conf - 0.2 * abs(s))

        return float(base_conf)

    def _estimate_volatility(self, df: pd.DataFrame) -> float:
        """
        Simple volatility estimate based on rolling std of daily returns.

        Returns latest 20-period rolling std of Close returns.
        """
        if "Close" not in df.columns:
            return 0.0

        close = df["Close"].dropna()
        if len(close) < 5:
            return 0.0

        returns = close.pct_change().dropna()
        if len(returns) < 5:
            return 0.0

        vol = returns.rolling(window=20, min_periods=5).std().iloc[-1]
        if pd.isna(vol):
            return 0.0

        return float(vol)

    def _position_size(
        self,
        signal: int,
        confidence: float,
        volatility: float,
    ) -> float:
        """
        Decide how many dollars to allocate to this trade.
        """
        if signal == 0 or confidence < 0.2:
            return 0.0

        base_risk = self.account_equity * self.max_risk_per_trade

        vol = max(volatility, 0.0)
        vol_factor = 1.0 / (1.0 + vol)

        position_value = base_risk * confidence * vol_factor
        return float(position_value)

    def _build_explanation(
        self,
        analyzer: StockAnalyzer,
        signal: int,
        confidence: float,
        volatility: float,
        sentiment: Optional[float],
    ) -> str:
        """
        Build a human-readable explanation string.
        """
        latest_row = analyzer.data.iloc[-1]
        bullish = []
        bearish = []

        for col in analyzer.data.columns:
            if not col.endswith("_signal"):
                continue

    # Skip the unified final signal – we only want raw indicators
            if col == "final_signal":
                continue

            name = col.replace("_signal", "").upper()
            val = latest_row[col]

            if val > 0:
                bullish.append(name)
            elif val < 0:
                bearish.append(name)

        action_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
        action = action_map.get(signal, "HOLD")

        parts = []
        parts.append(
            f"Decision: {action} with confidence {confidence:.2f}. "
        )

        if bullish:
            parts.append(f"Bullish indicators: {', '.join(bullish)}. ")
        if bearish:
            parts.append(f"Bearish indicators: {', '.join(bearish)}. ")

        parts.append(f"Estimated volatility: {volatility:.4f}. ")

        if sentiment is not None:
            parts.append(f"Market sentiment input: {sentiment:+.2f}. ")

        return "".join(parts)
