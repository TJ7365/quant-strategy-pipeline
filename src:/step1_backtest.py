# ===============================
# IMPORTS
# ===============================
import os
import argparse
import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
if "MPLBACKEND" not in os.environ:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt


# ===============================
# SETTINGS
# ===============================
TICKER          = "SPY"
VIX_TICKER      = "^VIX"
START           = "2015-01-01"
MOMENTUM_WINDOW = 126        # ~6 months of trading days for trend signal
VOL_WINDOW      = 20         # rolling volatility lookback (days)
VIX_THRESHOLD   = 28.0       # above this = high-vol / danger zone (was 25.0)
VOL_TARGET      = 0.10       # annualised volatility target (10%)
INITIAL_CAPITAL = 100_000


# ===============================
# DATA DOWNLOAD
# ===============================
def download_data(ticker, vix_ticker, start, max_retries=3, backoff=1.0):
    """Download SPY + VIX with retries, merge on date index."""
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            spy = yf.download(
                ticker, start=start,
                auto_adjust=True, progress=False,
                actions=False, threads=False,
            )
            vix = yf.download(
                vix_ticker, start=start,
                auto_adjust=True, progress=False,
                actions=False, threads=False,
            )

            # flatten MultiIndex if present
            for df in (spy, vix):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

            if spy.empty or vix.empty:
                raise RuntimeError("yfinance returned empty dataframe")

            spy = spy[["Open", "High", "Low", "Close", "Volume"]].dropna()

            # use VIX close only
            vix = vix[["Close"]].rename(columns={"Close": "vix"}).dropna()

            df = spy.join(vix, how="left")
            df["vix"] = df["vix"].ffill()   # fill any missing VIX dates
            return df

        except Exception as e:
            last_exc = e
            logging.warning("Download attempt %d/%d failed: %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(backoff * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"Failed to download data.\nyfinance error: {last_exc}\n"
        "Suggestions: check network, upgrade yfinance, or provide local CSV."
    ) from last_exc


# ===============================
# INDICATORS
# ===============================
def add_indicators(df, momentum_window=MOMENTUM_WINDOW,
                   vol_window=VOL_WINDOW, vix_threshold=VIX_THRESHOLD):
    df = df.copy()

    # ── 1. Time-series momentum signal ──────────────────────────────────────
    # Positive when price is above where it was `momentum_window` bars ago
    # AND above its 200-day SMA (double confirmation)
    df["mom_return"]   = df["Close"] / df["Close"].shift(momentum_window) - 1
    df["sma200"]       = df["Close"].rolling(200, min_periods=50).mean()
    df["trend_signal"] = (
        (df["mom_return"] > 0) &
        (df["Close"] > df["sma200"])
    )

    # ── 2. Volatility regime filter ─────────────────────────────────────────
    # Realised vol (annualised) of daily returns over vol_window days
    daily_ret          = df["Close"].pct_change()
    df["realised_vol"] = daily_ret.rolling(vol_window, min_periods=5).std() * np.sqrt(252)

    # VIX-based danger zone: true when VIX is BELOW threshold (calm market)
    df["vix_ok"]       = df["vix"] < vix_threshold

    # ── 3. Combined regime: both trend AND vol must agree ───────────────────
    df["regime"] = df["trend_signal"] & df["vix_ok"]
    df["regime"] = df["regime"].fillna(False)

    # ── 4. Volatility-scaled position size (vol targeting) ──────────────────
    # Size down when vol is high, size up when vol is low — capped at 1.0
    df["vol_scalar"] = (VOL_TARGET / df["realised_vol"].replace(0, np.nan)).clip(upper=1.0)
    df["vol_scalar"] = df["vol_scalar"].fillna(0.5)

    return df


# ===============================
# BACKTEST ENGINE
# ===============================
def run_backtest(df, initial_capital=INITIAL_CAPITAL):
    df    = df.copy()
    n     = len(df)
    dates = df.index.tolist()

    # buy-and-hold benchmark
    df["equity_buyhold"] = (
        (1 + df["Close"].pct_change().fillna(0)).cumprod() * initial_capital
    )
    df["equity_trade"] = np.nan

    cash       = initial_capital
    shares     = 0
    in_position = False
    entry_price = None
    trade_log   = []

    for i in range(1, n):
        date      = dates[i]
        regime    = bool(df["regime"].iloc[i])
        vol_sc    = float(df["vol_scalar"].iloc[i])
        open_px   = float(df["Open"].iloc[i])
        close_px  = float(df["Close"].iloc[i])

        # ── ENTRY ────────────────────────────────────────────────────────────
        # Enter (or add) when regime flips ON and we're not already positioned
        if regime and not in_position:
            # vol-targeted position: allocate vol_scalar fraction of equity
            equity_now = cash
            alloc      = equity_now * vol_sc
            shares     = int(alloc / open_px)

            if shares > 0:
                entry_price  = open_px
                cash        -= shares * entry_price
                in_position  = True

        # ── EXIT ─────────────────────────────────────────────────────────────
        # Exit when regime flips OFF
        elif in_position and not regime:
            exit_price = open_px
            cash      += shares * exit_price
            trade_log.append({
                "entry_date":  dates[i - 1],   # approximate entry date
                "exit_date":   date,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "shares":      shares,
                "pnl":         (exit_price - entry_price) * shares,
                "exit_type":   _exit_reason(df, i),
            })
            shares      = 0
            in_position = False
            entry_price = None

        # ── EQUITY MARK ───────────────────────────────────────────────────────
        df.loc[date, "equity_trade"] = cash + shares * close_px

    # close any open position at last price
    if in_position and shares > 0:
        last_date  = dates[-1]
        exit_price = float(df["Close"].iloc[-1])
        cash      += shares * exit_price
        trade_log.append({
            "entry_date":  entry_price,
            "exit_date":   last_date,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "shares":      shares,
            "pnl":         (exit_price - entry_price) * shares,
            "exit_type":   "end_of_data",
        })
        df.loc[last_date, "equity_trade"] = cash

    df["equity_trade"] = df["equity_trade"].ffill().fillna(initial_capital)
    trades_df = pd.DataFrame(trade_log)
    return df, trades_df


def _exit_reason(df, i):
    """Classify why we exited: trend reversal, vix spike, or both."""
    trend_ok = bool(df["trend_signal"].iloc[i])
    vix_ok   = bool(df["vix_ok"].iloc[i])
    if not trend_ok and not vix_ok:
        return "trend+vix"
    if not trend_ok:
        return "trend"
    return "vix_spike"


# ===============================
# METRICS
# ===============================
def compute_metrics(df, trades_df, initial_capital=INITIAL_CAPITAL):
    equity  = df["equity_trade"].astype(float)
    returns = equity.pct_change().dropna()

    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr  = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else float("nan")

    vol    = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else float("nan")

    rolling_max = equity.cummax()
    max_dd      = float((equity / rolling_max - 1).min())

    out = {
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
    }

    if trades_df is not None and not trades_df.empty:
        wins   = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        out["win_rate"]      = float(len(wins) / len(trades_df))
        out["profit_factor"] = float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) else float("inf")
        out["avg_win"]       = float(wins["pnl"].mean())  if len(wins)   else 0.0
        out["avg_loss"]      = float(losses["pnl"].mean()) if len(losses) else 0.0
        out["total_trades"]  = int(len(trades_df))

        ed = pd.to_datetime(trades_df["entry_date"], errors="coerce")
        xd = pd.to_datetime(trades_df["exit_date"],  errors="coerce")
        out["avg_hold_days"] = float((xd - ed).dt.days.mean())
    else:
        out.update({
            "win_rate": float("nan"), "profit_factor": float("nan"),
            "avg_win": 0.0, "avg_loss": 0.0,
            "total_trades": 0, "avg_hold_days": float("nan"),
        })

    return out


# ===============================
# MAIN
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Hybrid Trend + Low-Vol SPY Strategy")
    parser.add_argument("--ticker",            default=TICKER)
    parser.add_argument("--vix-ticker",        default=VIX_TICKER)
    parser.add_argument("--start",             default=START)
    parser.add_argument("--momentum-window",   type=int,   default=MOMENTUM_WINDOW)
    parser.add_argument("--vol-window",        type=int,   default=VOL_WINDOW)
    parser.add_argument("--vix-threshold",     type=float, default=VIX_THRESHOLD)
    parser.add_argument("--vol-target",        type=float, default=VOL_TARGET)
    parser.add_argument("--initial-capital",   type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--run-name",          default=None)
    parser.add_argument("--reports-dir",       default="reports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── download ──────────────────────────────────────────────────────────────
    logging.info("Downloading %s + VIX from %s…", args.ticker, args.start)
    df = download_data(args.ticker, args.vix_ticker, args.start)
    logging.info("Downloaded %d rows", len(df))

    # ── strategy ──────────────────────────────────────────────────────────────
    df = add_indicators(
        df,
        momentum_window = args.momentum_window,
        vol_window      = args.vol_window,
        vix_threshold   = args.vix_threshold,
    )
    df, trades_df = run_backtest(df, args.initial_capital)
    m = compute_metrics(df, trades_df, args.initial_capital)

    logging.info(
        "CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  Trades=%d",
        m["cagr"] * 100, m["sharpe"], m["max_drawdown"] * 100, m["total_trades"]
    )

    # ── persist to SQLite ─────────────────────────────────────────────────────
    from persist_sqlite import init_db, write_run

    db_path = "db/quant.db"
    init_db(db_path)

    # add strategy-specific columns so persist_sqlite doesn't choke
    for col in ["ema_fast", "ema_slow", "atr", "signal"]:
        if col not in df.columns:
            df[col] = np.nan

    run_meta = {
        "ticker":          args.ticker,
        "start_date":      args.start,
        "end_date":        str(df.index[-1].date()),
        "fast":            args.momentum_window,   # repurpose field
        "slow":            args.vol_window,         # repurpose field
        "atr_period":      0,
        "atr_multiplier":  args.vix_threshold,
        "risk_percentage": args.vol_target,
        "initial_capital": args.initial_capital,
        "notes":           f"Hybrid Trend+LowVol | VIX<{args.vix_threshold} | VolTarget={args.vol_target}",
    }

    run_id = write_run(db_path, run_meta, df, trades_df, m)
    logging.info("Saved run_id=%s to %s", run_id, db_path)

    # ── reports ───────────────────────────────────────────────────────────────
    if args.run_name:
        run_name = args.run_name
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_name = (
            f"{args.ticker}_hybrid_mom{args.momentum_window}"
            f"_vix{args.vix_threshold:g}_vt{args.vol_target:g}_{ts}"
        )

    out_dir = os.path.join(args.reports_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, f"{run_name}_equity.csv"))
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(out_dir, f"{run_name}_trades.csv"), index=False)

    # equity curve plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df["equity_buyhold"], label="Buy & Hold", color="#ff6b35", linewidth=1.2)
    axes[0].plot(df["equity_trade"],   label="Hybrid Strategy", color="#00e5ff", linewidth=1.2)
    axes[0].set_title(f"Equity Curve — {run_name}")
    axes[0].legend()
    axes[0].set_ylabel("Portfolio Value ($)")

    rolling_max = df["equity_trade"].cummax()
    drawdown    = df["equity_trade"] / rolling_max - 1
    axes[1].fill_between(df.index, drawdown, 0, color="#ff3b5c", alpha=0.6)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_title("Strategy Drawdown")

    axes[2].plot(df["vix"], color="#ffd700", linewidth=0.8, label="VIX")
    axes[2].axhline(args.vix_threshold, color="#ff3b5c", linestyle="--", linewidth=0.8, label=f"VIX threshold ({args.vix_threshold})")
    axes[2].fill_between(df.index, 0, df["vix"],
                         where=df["regime"].astype(bool),
                         color="#00ff88", alpha=0.2, label="In market")
    axes[2].set_ylabel("VIX")
    axes[2].set_title("VIX + Regime")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{run_name}_equity.png"), dpi=150)
    plt.close()

    logging.info("Reports saved to %s", out_dir)


if __name__ == "__main__":
    main()