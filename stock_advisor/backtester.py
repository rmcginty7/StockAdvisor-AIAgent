from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from stock_advisor.decision_engine import DecisionEngine  # or from .decision_engine import DecisionEngine


class Backtester:
    """
    Run a backtest using DecisionEngine decisions on historical data.

    Typical usage
    -------------
    >>> engine = DecisionEngine(symbol="AAPL", account_equity=10_000)
    >>> bt = Backtester(initial_equity=10_000)
    >>> results = bt.run(data=indicator_df, engine=engine)
    >>> results["stats"], results["trades"].head(), results["equity_curve"].tail()
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        trading_cost_pct: float = 0.0,
        sl_pct: float = 0.03,
        tp_pct: float = 0.06,
    ) -> None:
        """
        Parameters
        ----------
        initial_equity : float
            Starting capital for the backtest.
        trading_cost_pct : float
            Proportional cost per trade (e.g. 0.001 = 0.1% each side).
        sl_pct : float
            Stop-loss threshold (e.g. 0.03 = -3%).
        tp_pct : float
            Take-profit threshold (e.g. 0.06 = +6%).
        """
        self.initial_equity = float(initial_equity)
        self.trading_cost_pct = float(trading_cost_pct)
        self.sl_pct = float(sl_pct)
        self.tp_pct = float(tp_pct)

    def run(
        self,
        data: pd.DataFrame,
        engine: DecisionEngine,
        sentiment_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run a backtest over the given data using the provided DecisionEngine.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price + indicator data. Must contain a 'Close' column.
            Should already include any indicators the Analyzer/Engine rely on.
        engine : DecisionEngine
            Configured DecisionEngine instance (symbol, risk settings, etc.).
        sentiment_series : pd.Series, optional
            Optional series of sentiment values aligned with data.index.
            Each value should be in [-1, 1].

        Returns
        -------
        dict
            {
              "equity_curve": pd.Series,
              "trades": pd.DataFrame,
              "stats": Dict[str, float],
            }
        """
        if "Close" not in data.columns:
            raise ValueError("Backtester requires 'Close' column in data.")

        df = data.sort_index().copy()

        cash = self.initial_equity
        position_shares = 0.0
        entry_price: Optional[float] = None

        equity_history: List[Tuple[pd.Timestamp, float]] = []
        trades: List[Dict[str, Any]] = []

        index = df.index

        # Iterate from the second row onward (so there's at least some history)
        for i in range(1, len(df)):
            t = index[i]
            price = float(df.iloc[i]["Close"])

            # ---------- Stop-loss / Take-profit logic ----------
            if position_shares > 0 and entry_price is not None:
                pnl_pct = (price - entry_price) / entry_price

                force_sell = False
                reason = None

                if pnl_pct <= -self.sl_pct:
                    force_sell = True
                    reason = "STOP_LOSS"
                elif pnl_pct >= self.tp_pct:
                    force_sell = True
                    reason = "TAKE_PROFIT"

                if force_sell:
                    gross_proceeds = position_shares * price
                    fee = gross_proceeds * self.trading_cost_pct
                    net_proceeds = gross_proceeds - fee

                    cash += net_proceeds
                    trades.append({
                        "time": t,
                        "action": reason,   # distinguish from normal SELL
                        "price": price,
                        "shares": -position_shares,
                        "gross": gross_proceeds,
                        "fee": fee,
                        "cash_after": cash,
                    })
                    position_shares = 0.0
                    entry_price = None

                    # Record equity after forced exit and skip engine decision this bar
                    equity = cash + position_shares * price
                    equity_history.append((t, equity))
                    continue  # go to next bar

            # ---------- Normal engine-driven logic ----------
            window = df.iloc[: i + 1]

            sentiment = None
            if sentiment_series is not None and t in sentiment_series.index:
                sentiment = float(sentiment_series.loc[t])

            current_equity = cash + position_shares * price
            engine.account_equity = current_equity

            decision = engine.run(window, sentiment=sentiment)
            action = str(decision["action"]).upper()  # "BUY", "SELL", "HOLD"
            pos_value = float(decision["position_size_value"])

            # SELL: close any existing long position
            if action == "SELL" and position_shares > 0:
                gross_proceeds = position_shares * price
                fee = gross_proceeds * self.trading_cost_pct
                net_proceeds = gross_proceeds - fee

                cash += net_proceeds
                trades.append({
                    "time": t,
                    "action": "SELL",
                    "price": price,
                    "shares": -position_shares,
                    "gross": gross_proceeds,
                    "fee": fee,
                    "cash_after": cash,
                })
                position_shares = 0.0
                entry_price = None

            # BUY: if currently flat, open a long position
            elif action == "BUY" and position_shares == 0 and pos_value > 0:
                notional = min(pos_value, cash)  # can't allocate more than cash
                if notional > 0:
                    fee = notional * self.trading_cost_pct
                    net_notional = notional - fee
                    shares_to_buy = net_notional / price

                    cash -= notional
                    position_shares += shares_to_buy
                    entry_price = price

                    trades.append({
                        "time": t,
                        "action": "BUY",
                        "price": price,
                        "shares": shares_to_buy,
                        "gross": -notional,
                        "fee": fee,
                        "cash_after": cash,
                    })

            # HOLD: do nothing

            # Record equity at end of bar
            equity = cash + position_shares * price
            equity_history.append((t, equity))

        # If you want to force-close any open position at the very end,
        # you can add another sell here (currently we just mark its value).

        equity_curve = pd.Series(
            [e for _, e in equity_history],
            index=[t for t, _ in equity_history],
            name="equity",
        )

        trades_df = pd.DataFrame(trades)
        stats = self._compute_stats(equity_curve, trades_df)

        return {
            "equity_curve": equity_curve,
            "trades": trades_df,
            "stats": stats,
        }

    def _compute_stats(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute basic performance statistics from the equity curve and trades.
        """
        stats: Dict[str, float] = {}

        if equity_curve.empty:
            return stats

        start_value = float(equity_curve.iloc[0])
        end_value = float(equity_curve.iloc[-1])

        total_return = (end_value / start_value) - 1.0
        stats["start_equity"] = start_value
        stats["end_equity"] = end_value
        stats["total_return_pct"] = total_return * 100.0

        # Drawdown (peak-to-trough)
        roll_max = equity_curve.cummax()
        drawdowns = (equity_curve / roll_max) - 1.0
        max_drawdown = float(drawdowns.min())
        stats["max_drawdown_pct"] = max_drawdown * 100.0

        # Number of trades and rough win rate
        stats["num_trades"] = float(len(trades))
        if not trades.empty:
            pnl_list = []
            current_buy = None

            for _, row in trades.sort_values("time").iterrows():
                if row["action"] == "BUY":
                    current_buy = row
                elif row["action"] in ("SELL", "STOP_LOSS", "TAKE_PROFIT") and current_buy is not None:
                    pnl = row["cash_after"] - current_buy["cash_after"]
                    pnl_list.append(pnl)
                    current_buy = None

            if pnl_list:
                wins = sum(1 for x in pnl_list if x > 0)
                stats["win_rate_pct"] = 100.0 * wins / len(pnl_list)
            else:
                stats["win_rate_pct"] = 0.0
        else:
            stats["win_rate_pct"] = 0.0

        return stats

    
