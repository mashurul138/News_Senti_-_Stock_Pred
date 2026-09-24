#!/usr/bin/env python3
"""
NEXUS Stock AI — Vectorized Backtesting & Financial Simulation Engine (Phase 26)
Provides institutional-grade performance attribution, transaction cost modeling,
drawdown tracking, and benchmark comparison for directional equity strategies.
"""

import os
import json
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class VectorizedBacktester:
    """
    Vectorized equity portfolio backtesting and attribution engine.
    Supports dynamic thresholds, transaction friction (bps), and equal-weighted portfolio aggregation.
    """

    def __init__(self, cost_bps: float = 10.0, risk_free_rate: float = 0.0):
        """
        Args:
            cost_bps: Transaction cost in basis points (10 bps = 0.10% = 0.0010) deducted on entry & exit.
            risk_free_rate: Annualized risk-free rate for Sharpe ratio calculation (default 0.0).
        """
        self.cost_rate = cost_bps / 10000.0  # 10 bps = 0.0010
        self.risk_free_rate = risk_free_rate

    def compute_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes leak-free forward 1-day close-to-close returns per ticker:
        R_forward_{t+1} = (close_{t+1} - close_t) / close_t.
        """
        df = df.copy()
        if "forward_return_1d" not in df.columns:
            df["forward_return_1d"] = df.groupby("stock_symbol")["close"].pct_change(1).shift(-1)
            # Fill terminal row per ticker with 0.0
            df["forward_return_1d"] = df["forward_return_1d"].fillna(0.0)
        return df

    def run_backtest(
        self,
        df: pd.DataFrame,
        prob_col: str = "up_probability",
        threshold: Union[float, str] = 0.50
    ) -> Dict[str, Any]:
        """
        Executes vectorized backtest for a given probability threshold.

        Args:
            df: DataFrame containing ['date', 'stock_symbol', 'close', prob_col, 'forward_return_1d']
            prob_col: Name of column containing P(UP)
            threshold: Float (e.g. 0.50, 0.51) or "median" for cross-sectional median threshold.

        Returns:
            Dictionary with portfolio equity series, daily returns, and performance metrics.
        """
        df = self.compute_forward_returns(df)
        df = df.sort_values(["date", "stock_symbol"]).reset_index(drop=True)

        # 1. Determine Position Signal
        if isinstance(threshold, str) and threshold.lower() == "median":
            # Dynamic cross-sectional median per date
            daily_medians = df.groupby("date")[prob_col].transform("median")
            df["position"] = (df[prob_col] >= daily_medians).astype(int)
        else:
            thresh_val = float(threshold)
            df["position"] = (df[prob_col] >= thresh_val).astype(int)

        # 2. Transaction Costs on Position Transitions
        # Shift position per ticker to detect entry (0 -> 1) and exit (1 -> 0)
        df["prev_position"] = df.groupby("stock_symbol")["position"].shift(1).fillna(0).astype(int)
        df["trade_action"] = (df["position"] != df["prev_position"]).astype(int)
        df["trans_cost"] = df["trade_action"] * self.cost_rate

        # 3. Net Stock Return
        df["net_return"] = df["position"] * df["forward_return_1d"] - df["trans_cost"]

        # 4. Equal-Weighted Portfolio Aggregation per Trading Date
        portfolio_daily = df.groupby("date").agg(
            strat_return=("net_return", "mean"),
            bench_return=("forward_return_1d", "mean"),
            active_stocks=("position", "sum"),
            total_stocks=("position", "count"),
            trades_today=("trade_action", "sum")
        ).reset_index()

        # Deduct initial benchmark entry cost on day 0
        if len(portfolio_daily) > 0:
            portfolio_daily.loc[0, "bench_return"] -= self.cost_rate

        # 5. Cumulative Equity Curves
        portfolio_daily["strat_equity"] = (1.0 + portfolio_daily["strat_return"]).cumprod()
        portfolio_daily["bench_equity"] = (1.0 + portfolio_daily["bench_return"]).cumprod()

        # 6. Drawdown Calculation
        portfolio_daily["strat_peak"] = portfolio_daily["strat_equity"].cummax()
        portfolio_daily["strat_drawdown"] = (portfolio_daily["strat_equity"] - portfolio_daily["strat_peak"]) / portfolio_daily["strat_peak"]

        portfolio_daily["bench_peak"] = portfolio_daily["bench_equity"].cummax()
        portfolio_daily["bench_drawdown"] = (portfolio_daily["bench_equity"] - portfolio_daily["bench_peak"]) / portfolio_daily["bench_peak"]

        # 7. Summary Metrics
        strat_metrics = self.compute_metrics(
            returns=portfolio_daily["strat_return"],
            equity=portfolio_daily["strat_equity"],
            drawdown=portfolio_daily["strat_drawdown"],
            trades=int(df["trade_action"].sum())
        )

        bench_metrics = self.compute_metrics(
            returns=portfolio_daily["bench_return"],
            equity=portfolio_daily["bench_equity"],
            drawdown=portfolio_daily["bench_drawdown"],
            trades=int(df["stock_symbol"].nunique())
        )

        return {
            "threshold": threshold,
            "portfolio_daily": portfolio_daily,
            "stock_trades_df": df,
            "strat_metrics": strat_metrics,
            "bench_metrics": bench_metrics
        }

    def compute_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        drawdown: pd.Series,
        trades: int
    ) -> Dict[str, float]:
        """Calculates institutional financial performance metrics."""
        n_days = len(returns)
        if n_days == 0:
            return {}

        total_return = float(equity.iloc[-1] - 1.0)
        cagr = float((equity.iloc[-1]) ** (252.0 / n_days) - 1.0) if equity.iloc[-1] > 0 else -1.0
        daily_mean = float(returns.mean())
        daily_std = float(returns.std())

        ann_vol = float(daily_std * np.sqrt(252.0))
        sharpe = float((daily_mean * 252.0 - self.risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0

        max_dd = float(abs(drawdown.min()))
        calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

        win_rate = float((returns > 0).mean())

        return {
            "cumulative_return": total_return,
            "annualized_return": cagr,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "daily_win_rate": win_rate,
            "total_trades": trades
        }

    def run_threshold_sweep(
        self,
        df: pd.DataFrame,
        prob_col: str = "up_probability",
        thresholds: Optional[List[Union[float, str]]] = None
    ) -> pd.DataFrame:
        """Runs comparative sweep across multiple conviction thresholds."""
        if thresholds is None:
            thresholds = [0.50, 0.51, 0.52, "median"]

        records = []
        for th in thresholds:
            res = self.run_backtest(df, prob_col=prob_col, threshold=th)
            m = res["strat_metrics"]
            name = f"NEXUS AI (P >= {th:.2f})" if isinstance(th, (int, float)) else f"NEXUS AI ({th.capitalize()})"
            records.append({
                "Strategy": name,
                "Cumulative Return": f"{m['cumulative_return'] * 100:+.2f}%",
                "Annualized Return": f"{m['annualized_return'] * 100:+.2f}%",
                "Annualized Vol": f"{m['annualized_volatility'] * 100:.2f}%",
                "Sharpe Ratio": f"{m['sharpe_ratio']:.2f}",
                "Max Drawdown": f"{m['max_drawdown'] * 100:.2f}%",
                "Calmar Ratio": f"{m['calmar_ratio']:.2f}",
                "Daily Win Rate": f"{m['daily_win_rate'] * 100:.1f}%",
                "Total Trades": m["total_trades"]
            })

        # Add Benchmark
        bench_res = self.run_backtest(df, prob_col=prob_col, threshold=0.0)
        bm = bench_res["bench_metrics"]
        records.append({
            "Strategy": "Buy & Hold Benchmark (Equal-Weight)",
            "Cumulative Return": f"{bm['cumulative_return'] * 100:+.2f}%",
            "Annualized Return": f"{bm['annualized_return'] * 100:+.2f}%",
            "Annualized Vol": f"{bm['annualized_volatility'] * 100:.2f}%",
            "Sharpe Ratio": f"{bm['sharpe_ratio']:.2f}",
            "Max Drawdown": f"{bm['max_drawdown'] * 100:.2f}%",
            "Calmar Ratio": f"{bm['calmar_ratio']:.2f}",
            "Daily Win Rate": f"{bm['daily_win_rate'] * 100:.1f}%",
            "Total Trades": bm["total_trades"]
        })

        return pd.DataFrame(records)

    def plot_equity_and_drawdown(
        self,
        backtest_result: Dict[str, Any],
        title: str = "NEXUS Stock AI — 2023 Out-of-Sample Backtest",
        save_path: Optional[str] = None
    ):
        """Generates 2-panel chart: Cumulative Equity Curve & Underwater Drawdown Profile."""
        daily = backtest_result["portfolio_daily"]
        sm = backtest_result["strat_metrics"]
        bm = backtest_result["bench_metrics"]
        thresh = backtest_result["threshold"]
        thresh_label = f"P(UP) >= {thresh:.2f}" if isinstance(thresh, (int, float)) else f"{thresh.capitalize()}"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.5]}, dpi=120)

        dates = pd.to_datetime(daily["date"])

        # Panel 1: Cumulative Equity Curve
        ax1.plot(dates, daily["strat_equity"], label=f"NEXUS AI ({thresh_label}) | Cum: {sm['cumulative_return']*100:+.1f}% | Sharpe: {sm['sharpe_ratio']:.2f}", color="#26a69a", linewidth=2.2)
        ax1.plot(dates, daily["bench_equity"], label=f"Buy & Hold Benchmark (Equal-Weight) | Cum: {bm['cumulative_return']*100:+.1f}% | Sharpe: {bm['sharpe_ratio']:.2f}", color="#848e9c", linewidth=1.8, linestyle="--")
        ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
        ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax1.set_ylabel("Portfolio Equity ($1.00 Base)", fontsize=11)
        ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Panel 2: Underwater Drawdown Profile
        ax2.fill_between(dates, daily["strat_drawdown"] * 100, 0, color="#26a69a", alpha=0.35, label=f"NEXUS AI Drawdown (Max: {sm['max_drawdown']*100:.1f}%)")
        ax2.plot(dates, daily["strat_drawdown"] * 100, color="#26a69a", linewidth=1.2)

        ax2.plot(dates, daily["bench_drawdown"] * 100, color="#ef5350", linewidth=1.2, linestyle="--", label=f"Benchmark Drawdown (Max: {bm['max_drawdown']*100:.1f}%)")
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Trading Date (2023 Out-of-Sample)", fontsize=11)
        ax2.legend(loc="lower left", frameon=True, framealpha=0.9)
        ax2.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ Saved backtest equity chart to: {save_path}")

        plt.show()
