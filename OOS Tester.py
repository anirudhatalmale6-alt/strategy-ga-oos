#!/usr/bin/env python3
"""
OOS Tester.py  —  Out-of-sample validator.
==========================================
Loads the top strategies the optimizer saved (ga_core.HOF_JSON), re-runs each on
the FULL timeline through the SAME engine the optimizer used (ga_core.run_backtest),
then slices the equity curve into the in-sample window and one or more
out-of-sample windows so you can see, per strategy, whether the edge held up
outside the training years.

Strategy logic is imported from ga_core.py — identical to the optimizer.
"""
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
plt.rcParams["text.parse_math"] = False   # treat '$' in titles/labels as literal, not LaTeX math
import numpy as np
import pandas as pd

import ga_core as core
from ga_core import (StrategyGenome, ConditionGene, run_backtest,
                     load_and_preprocess_data, load_hall_of_fame,
                     compute_performance_metrics)


# ---------------- human-readable logic ----------------
def format_condition(c: ConditionGene) -> str:
    if c.op in (">=", "<=", "<", ">"):
        return f"{c.var1}[t-{c.shift1}] {c.op} {c.var2}[t-{c.shift2}]"
    if c.op == "rising":
        return "Close[t] > Close[t-1]"
    if c.op == "falling":
        return "Close[t] < Close[t-1]"
    if c.op == "higher_x":
        return f"{c.var1}[t] > max({c.var1}, last {c.x_bars})"
    if c.op == "lower_x":
        return f"{c.var1}[t] < min({c.var1}, last {c.x_bars})"
    return str(c)


def print_strategy_logic(rank: int, g: StrategyGenome):
    trig = "High exceeds" if g.entry_trigger_type == "breakout" else "Low dips below"
    print("\n" + "-" * 70)
    print(f" STRATEGY RANK #{rank} LOGIC")
    print("-" * 70)
    print(f"  Entry cond : {format_condition(g.entry_cond) if g.use_entry_cond else 'Disabled'}")
    print(f"  Entry trig : {trig} {g.entry_ref_var}[t-{g.entry_shift}] "
          f"{'+' if g.entry_trigger_type=='breakout' else '-'} {g.entry_offset_ticks} ticks")
    if g.exit_style == "ticks":
        print(f"  Protective : STOP {g.exit_trigger_ticks} ticks below entry, "
              f"limit {g.exit_offset_ticks} ticks under the trigger (LE)")
    else:
        print(f"  Protective : ATR({g.atr_period}) stop x{g.atr_sl_mult} / target x{g.atr_pt_mult}")
    print(f"  Time exit  : {g.max_bars_hold} bars")
    print(f"  Custom exit: {format_condition(g.exit_cond) if g.use_exit_cond else 'Disabled'}")
    print("-" * 70)


# ---------------- slice trades by date ----------------
def slice_trades_by_range(trade_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if trade_df.empty:
        return trade_df
    s = pd.to_datetime(start)
    e = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return trade_df[(trade_df["exit_dt"] >= s) & (trade_df["exit_dt"] <= e)]


# ---------------- plot IS + OOS windows ----------------
def plot_windows(results_list: List[Tuple[int, Dict[str, Any]]],
                 is_range: Tuple[str, str], oos_ranges: List[Tuple[str, str]]):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (rank, stats) in enumerate(results_list):
        fig, ax = plt.subplots(figsize=(12, 6), num=f"Rank #{rank} IS/OOS")
        tdf = stats["trade_df"]
        if tdf.empty:
            ax.text(0.5, 0.5, f"Rank #{rank}: no trades", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.plot(tdf["exit_dt"], stats["equity_curve"], color=colors[idx % len(colors)],
                lw=2.0, label=f"Rank #{rank} equity")
        ax.axhline(0, color="black", ls="--", alpha=0.5)

        is_tr = slice_trades_by_range(tdf, is_range[0], is_range[1])
        if not is_tr.empty:
            m = compute_performance_metrics(is_tr)
            ax.axvspan(is_tr["exit_dt"].iloc[0], is_tr["exit_dt"].iloc[-1],
                       color="skyblue", alpha=0.20,
                       label=f"IS {is_range[0]}..{is_range[1]} | ${m['net_profit']:.0f} | Sh {m['sharpe_ratio']:.2f}")

        for k, (s, e) in enumerate(oos_ranges, start=1):
            oos_tr = slice_trades_by_range(tdf, s, e)
            if oos_tr.empty:
                continue
            m = compute_performance_metrics(oos_tr)
            ax.axvspan(oos_tr["exit_dt"].iloc[0], oos_tr["exit_dt"].iloc[-1],
                       color="gold", alpha=0.25,
                       label=f"OOS{k} {s}..{e} | ${m['net_profit']:.0f} | Sh {m['sharpe_ratio']:.2f}")
            ax.axvline(oos_tr["exit_dt"].iloc[0], color="red", ls="--", lw=1.2, alpha=0.8)

        ax.set_title(f"Rank #{rank} | net ${stats['net_profit']:,.0f} | "
                     f"PF {stats['profit_factor']:.2f} | Sharpe {stats['sharpe_ratio']:.2f} | "
                     f"Win {stats['win_rate']*100:.1f}% | Trades {stats['trade_count']} | "
                     f"MaxDD ${stats['max_dd']:,.0f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("Net Profit ($)")
        ax.grid(True, ls=":", alpha=0.6); ax.legend(loc="upper left", fontsize=8.5)
        plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DATA_FILEPATH = "C:/Users/Administrator/Desktop/Stock Data/SVXY P&F 3-3 2p.txt"

    # IS window = the optimizer's training years; everything else is OOS.
    IS_RANGE = (f"{core.IS_START_YEAR}-01-01", f"{core.IS_END_YEAR}-12-31")
    OOS_RANGES = [
        ("2019-01-01", "2023-12-31"),   # OOS 1 (historical, before training)
        ("2026-01-01", "2026-12-31"),   # OOS 2 (forward, after training)
    ]

    print("=" * 80)
    print(f"LOADING TOP STRATEGIES FROM '{core.HOF_JSON}'")
    print("=" * 80)
    try:
        top = load_hall_of_fame(core.HOF_JSON)
    except FileNotFoundError:
        print(f"ERROR: '{core.HOF_JSON}' not found. Run PyGeneticAlgo.py first.")
        raise SystemExit(1)
    print(f"Loaded {len(top)} strategies.")

    for rank, _, g in top:
        print_strategy_logic(rank, g)

    print("\nLoading full timeline ...")
    full_df = load_and_preprocess_data(DATA_FILEPATH)  # no year filter = all data
    print(f"Full bars: {len(full_df)}\n")

    results = []
    print("=" * 100)
    print(" FULL BACKTEST SUMMARY (IS + all OOS)")
    print("=" * 100)
    print(f"{'Rank':<5} | {'Net PnL':<11} | {'PF':<6} | {'Sharpe':<7} | {'Sortino':<7} | "
          f"{'$/Trade':<9} | {'Win%':<6} | {'Trades':<6} | {'MaxDD':<9}")
    print("-" * 100)
    for rank, _, g in top:
        stats = run_backtest(g, full_df)
        results.append((rank, stats))
        pf = "Inf" if stats["profit_factor"] == float("inf") else f"{stats['profit_factor']:.2f}"
        print(f"#{rank:<4} | ${stats['net_profit']:<10,.2f} | {pf:<6} | "
              f"{stats['sharpe_ratio']:<7.2f} | {stats['sortino_ratio']:<7.2f} | "
              f"${stats['avg_per_trade']:<8.2f} | {stats['win_rate']*100:<5.1f}% | "
              f"{stats['trade_count']:<6} | ${stats['max_dd']:<8,.2f}")
    print("=" * 100)

    plot_windows(results, is_range=IS_RANGE, oos_ranges=OOS_RANGES)
