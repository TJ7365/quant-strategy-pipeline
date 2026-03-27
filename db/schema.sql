PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
    ticker TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL,
    fast INTEGER NOT NULL, slow INTEGER NOT NULL, atr_period INTEGER NOT NULL,
    atr_multiplier REAL NOT NULL, risk_percentage REAL NOT NULL,
    initial_capital REAL NOT NULL, notes TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    cagr REAL, volatility REAL, sharpe REAL, max_drawdown REAL,
    win_rate REAL, profit_factor REAL, avg_win REAL, avg_loss REAL,
    total_trades INTEGER, avg_hold_days REAL
);

CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    date TEXT NOT NULL, equity_trade REAL, equity_buyhold REAL,
    signal INTEGER, ema_fast REAL, ema_slow REAL, atr REAL
);

CREATE INDEX IF NOT EXISTS idx_equity_run_date ON equity(run_id, date);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    trade_id INTEGER NOT NULL, entry_date TEXT, exit_date TEXT,
    entry_price REAL, exit_price REAL, shares INTEGER, pnl REAL, exit_type TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
