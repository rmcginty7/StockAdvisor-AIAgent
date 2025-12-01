from flask import Flask, render_template, request, jsonify
from stock_advisor.engine_service import run_backtest, run_latest_decision

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """
    Render the main page with a simple form/JS to query the API.
    """
    return render_template("index.html")

@app.route("/model")
def model_summary():
    """
    Render a page that summarizes the AI trading model."""
    return render_template("model.html")

@app.route("/api/decision", methods=["GET"])
def api_decision():
    """
    API endpoint:
      GET /api/decision?symbol=AAPL
    Returns the latest AI trading decision for the given symbol.
    """
    symbol = request.args.get("symbol", "AAPL").strip().upper()
    lookback = request.args.get("lookback_days", default=252)

    try:
        lookback_days = int(lookback)
    except ValueError:
        lookback_days = 252
    
    try:
        decision = run_latest_decision(
            symbol,
            lookback_days=lookback_days,
        )
        return jsonify({"ok": True, "symbol": symbol, "decision": decision})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "symbol":symbol}), 400

@app.route("/api/backtest", methods=["GET"])
def api_backtest():
    """
    API endpoint:
      GET /api/backtest?symbol=AAPL&start_date=2020-01-01
    Returns backtest stats and a few sample trades.
    """
    symbol = request.args.get("symbol", "AAPL").strip().upper()
    start_date = request.args.get("start_date", "2020-01-01")
    end_date = request.args.get("end_date") or None

    try:
        bt_results = run_backtest(symbol, start_date=start_date, end_date=end_date)
        return jsonify({"ok": True, "symbol": symbol, **bt_results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "symbol": symbol}), 400
    


if __name__ == "__main__":
    app.run(debug=True)
