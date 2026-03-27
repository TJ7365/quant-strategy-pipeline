# Quant Strategy Pipeline

An end-to-end quantitative trading pipeline that backtests an EMA crossover strategy with ATR-based risk management on live equity data.

## Overview

- **Backtest Engine** — EMA crossover signals with VIX regime filtering and ATR-based position sizing
- **Data** — Live SPY + VIX data pulled via yfinance from 2015–present
- **Persistence** — SQLite database storing runs, trades, equity curves, and performance metrics
- **Dashboard** — Interactive HTML + Streamlit dashboard with equity curves, drawdown, P&L histogram, and trade log

## Results (Initial Run)

| Metric | Value |
|--------|-------|
| CAGR | 9.02% |
| Sharpe Ratio | 1.36 |
| Max Drawdown | -8.11% |
| Total Trades | 62 |

## Tech Stack

Python · Pandas · SQLite · Streamlit · yfinance · Matplotlib

## Structure
```
src/
  step1_backtest.py     # Backtest engine
  persist_sqlite.py     # Save results to DB
  app.py                # Streamlit dashboard
db/
  quant.db              # SQLite database
  dashboard.html        # Standalone HTML dashboard
data/                   # Raw price data
reports/                # Generated reports
```

## Usage
```bash
# Run backtest
python src/step1_backtest.py

# Launch dashboard
streamlit run src/app.py
```