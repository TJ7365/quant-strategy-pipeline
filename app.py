import sqlite3
import os

import pandas as pd
import streamlit as st

# ── DB path ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "db", "quant.db")

st.set_page_config(page_title="Quant Research Dashboard", layout="wide")


# ── data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_runs(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """SELECT run_id, created_at, ticker, start_date, end_date,
                      fast, slow, atr_period, atr_multiplier, risk_percentage,
                      initial_capital, notes
               FROM runs ORDER BY created_at DESC""",
            conn,
        )


@st.cache_data
def load_equity(db_path: str, run_id: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """SELECT date, equity_trade, equity_buyhold, signal, ema_fast, ema_slow, atr
               FROM equity WHERE run_id = ? ORDER BY date ASC""",
            conn, params=(run_id,),
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


@st.cache_data
def load_trades(db_path: str, run_id: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """SELECT trade_id, entry_date, exit_date, entry_price, exit_price,
                      shares, pnl, exit_type
               FROM trades WHERE run_id = ? ORDER BY trade_id ASC""",
            conn, params=(run_id,),
        )
    if not df.empty:
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"]  = pd.to_datetime(df["exit_date"])
        df["hold_days"]  = (df["exit_date"] - df["entry_date"]).dt.days
    return df


@st.cache_data
def load_metrics(db_path: str, run_id: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM metrics WHERE run_id = ?", conn, params=(run_id,)
        )


def compute_drawdown(equity: pd.Series) -> pd.Series:
    equity = equity.astype(float)
    return equity / equity.cummax() - 1.0


def run_label(row: pd.Series, idx: int) -> str:
    return (
        f"[{idx}] {row['created_at'][:16]} | {row['ticker']} "
        f"| EMA {row['fast']}/{row['slow']} "
        f"| ATR×{row['atr_multiplier']} | risk {row['risk_percentage']}"
    )


# ── load runs ─────────────────────────────────────────────────────────────────
st.title("📈 Quant Strategy Research Dashboard")

runs = pd.DataFrame()
try:
    runs = load_runs(DB_PATH)
except Exception as e:
    st.error(
        f"Could not read database at **{DB_PATH}**\n\n"
        f"Make sure you ran `step1_backtest.py` first.\n\nError: `{e}`"
    )
    st.stop()

if runs.empty:
    st.warning("No runs found yet. Run your backtest first.")
    st.stop()

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")

run_options = {
    run_label(row, idx): row["run_id"]
    for idx, (_, row) in enumerate(runs.iterrows())
}
run_labels = list(run_options.keys())

selected_label = st.sidebar.selectbox("Select a run", options=run_labels, index=0)
run_id = run_options[selected_label]

compare = st.sidebar.checkbox("Compare with another run")
compare_run_id = None
if compare:
    other = [l for l in run_labels if l != selected_label]
    if other:
        compare_run_id = run_options[st.sidebar.selectbox("Compare against", other)]
    else:
        st.sidebar.info("Only one run available.")

use_date_filter = st.sidebar.checkbox("Filter by date range")

# ── load data ─────────────────────────────────────────────────────────────────
equity  = load_equity(DB_PATH, run_id)
trades  = load_trades(DB_PATH, run_id)
metrics = load_metrics(DB_PATH, run_id)

# ── date filter ───────────────────────────────────────────────────────────────
start_dt = end_dt = None
if use_date_filter and not equity.empty:
    min_d = equity.index.min().date()
    max_d = equity.index.max().date()
    date_result = st.sidebar.date_input("Date range", (min_d, max_d))

    if isinstance(date_result, (list, tuple)) and len(date_result) == 2:
        start_dt, end_dt = date_result
        equity = equity.loc[str(start_dt): str(end_dt)]
        if not trades.empty:
            trades = trades[
                (trades["entry_date"] >= pd.Timestamp(start_dt)) &
                (trades["entry_date"] <= pd.Timestamp(end_dt))
            ]
    else:
        st.sidebar.warning("Select both a start and end date.")

# ── run summary ───────────────────────────────────────────────────────────────
run_row = runs[runs["run_id"] == run_id].iloc[0]
st.subheader("Run Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", run_row["ticker"])
c2.metric("Period", f"{run_row['start_date']} → {run_row['end_date']}")
c3.metric("EMA", f"{run_row['fast']}/{run_row['slow']}  ATR×{run_row['atr_multiplier']}")
c4.metric("Risk %", f"{float(run_row['risk_percentage']):.2%}")

# ── metrics ───────────────────────────────────────────────────────────────────
st.subheader("Performance Metrics")
if metrics.empty:
    st.info("No metrics found for this run.")
else:
    m = metrics.iloc[0].to_dict()
    col = st.columns(5)
    col[0].metric("CAGR",          f"{m.get('cagr', 0):.2%}")
    col[1].metric("Sharpe",        f"{m.get('sharpe', 0):.2f}")
    col[2].metric("Max Drawdown",  f"{m.get('max_drawdown', 0):.2%}")
    col[3].metric("Win Rate",      f"{m.get('win_rate', 0):.2%}")
    col[4].metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")

    col2 = st.columns(5)
    col2[0].metric("Volatility",    f"{m.get('volatility', 0):.2%}")
    col2[1].metric("Total Trades",  str(int(m.get('total_trades', 0))))
    col2[2].metric("Avg Win",       f"${m.get('avg_win', 0):.0f}")
    col2[3].metric("Avg Loss",      f"${m.get('avg_loss', 0):.0f}")
    col2[4].metric("Avg Hold Days", f"{m.get('avg_hold_days', 0):.1f}")

# ── equity curves ─────────────────────────────────────────────────────────────
st.subheader("Equity Curves")
if equity.empty:
    st.info("No equity data for selected range.")
else:
    plot_df = pd.DataFrame({
        "Strategy":   equity["equity_trade"].astype(float),
        "Buy & Hold": equity["equity_buyhold"].astype(float),
    })
    st.line_chart(plot_df, use_container_width=True)

    st.subheader("Drawdown")
    dd = compute_drawdown(equity["equity_trade"])
    st.line_chart(dd.rename("Drawdown"), use_container_width=True)

# ── compare mode ──────────────────────────────────────────────────────────────
if compare_run_id:
    st.subheader("Run Comparison")
    eq2 = load_equity(DB_PATH, compare_run_id)
    if start_dt and end_dt:
        eq2 = eq2.loc[str(start_dt): str(end_dt)]
    if not equity.empty and not eq2.empty:
        comp = pd.DataFrame({
            "Selected":  equity["equity_trade"].astype(float),
            "Compared":  eq2["equity_trade"].astype(float),
        }).dropna()
        st.line_chart(comp, use_container_width=True)

# ── trades ────────────────────────────────────────────────────────────────────
st.subheader("Trade Log")
if trades.empty:
    st.info("No trades for this run.")
else:
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Total Trades", len(trades))
    w2.metric("Total PnL",    f"${trades['pnl'].sum():,.0f}")
    w3.metric("Win Rate",     f"{(trades['pnl'] > 0).mean():.2%}")
    w4.metric("Avg Hold",     f"{trades['hold_days'].mean():.1f} days")

    st.dataframe(
        trades.style.applymap(
            lambda v: "color: #00cc66" if isinstance(v, float) and v > 0
                      else ("color: #ff4444" if isinstance(v, float) and v < 0 else ""),
            subset=["pnl"]
        ),
        use_container_width=True,
    )

# ── all runs leaderboard ──────────────────────────────────────────────────────
with st.expander("📊 All Runs Leaderboard"):
    with sqlite3.connect(DB_PATH) as conn:
        lb = pd.read_sql_query(
            """SELECT r.ticker, r.fast, r.slow, r.atr_multiplier, r.risk_percentage,
                      m.cagr, m.sharpe, m.max_drawdown, m.win_rate, m.total_trades
               FROM runs r LEFT JOIN metrics m ON m.run_id = r.run_id
               ORDER BY m.sharpe DESC""",
            conn
        )
    st.dataframe(lb, use_container_width=True)