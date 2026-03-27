import sqlite3
import uuid
from datetime import datetime, timezone
import pandas as pd

def new_run_id() -> str:
    return uuid.uuid4().hex  # simple unique id

def init_db(db_path: str, schema_path: str = "db/schema.sql"):
    with sqlite3.connect(db_path) as conn:
        with open(schema_path, "r") as f:
            conn.executescript(f.read())

def write_run(
    db_path: str,
    run_meta: dict,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    metrics: dict,
) -> str:
    run_id = new_run_id()
    created_at = datetime.now(timezone.utc).isoformat()

    # --- runs row ---
    runs_row = {
        "run_id": run_id,
        "created_at": created_at,
        **run_meta,
    }

    # --- equity table ---
    eq = equity_df.copy()
    eq = eq.reset_index()

    for col in ["Date", "Datetime", "index"]:
        if col in eq.columns:
            eq = eq.rename(columns={col: "date"})
            break
    eq["date"] = eq["date"].astype(str)
    eq["run_id"] = run_id

    keep_cols = ["run_id", "date", "equity_trade", "equity_buyhold", "signal", "ema_fast", "ema_slow", "atr"]
    for c in keep_cols:
        if c not in eq.columns:
            eq[c] = None
    eq = eq[keep_cols]

    # --- trades table ---
    tr = trades_df.copy()
    if tr.empty:
        tr = pd.DataFrame(columns=["entry_date","exit_date","entry_price","exit_price","shares","pnl","exit_type"])

    # make sure dates are strings
    for col in ["entry_date", "exit_date"]:
        if col in tr.columns:
            tr[col] = pd.to_datetime(tr[col]).astype(str)

    tr = tr.reset_index(drop=True)
    tr["trade_id"] = tr.index + 1
    tr["run_id"] = run_id

    tr_keep = ["run_id","trade_id","entry_date","exit_date","entry_price","exit_price","shares","pnl","exit_type"]
    for c in tr_keep:
        if c not in tr.columns:
            tr[c] = None
    tr = tr[tr_keep]

    # --- metrics table ---
    metrics_row = {"run_id": run_id, **metrics}

    with sqlite3.connect(db_path) as conn:
        pd.DataFrame([runs_row]).to_sql("runs", conn, if_exists="append", index=False)
        eq.to_sql("equity", conn, if_exists="append", index=False)
        tr.to_sql("trades", conn, if_exists="append", index=False)
        pd.DataFrame([metrics_row]).to_sql("metrics", conn, if_exists="append", index=False)

    return run_id