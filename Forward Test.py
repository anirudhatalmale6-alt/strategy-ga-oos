#!/usr/bin/env python3
"""
Forward Test.py  —  Single-strategy, pinned backtest / forward-test runner.
==========================================================================
Purpose
-------
The GA (PyGeneticAlgo.py) discovers strategies; the OOS Tester validates the
whole hall of fame across IS / OOS windows. This file does something different
and deliberately narrow:

    take ONE chosen strategy, FREEZE it into this file, and run it on ANY data
    file you point at — so you can compare the backtest against your live /
    forward results as fresh bars come in.

It runs through the SAME ga_core engine as the optimizer and the OOS tester
(single source of truth), so tick snapping, entry spread, futures P&L, exits —
everything matches your live setup bar-for-bar.

How to pin a strategy (two ways)
--------------------------------
  A) Leave PINNED_GENOME = None and set PIN_FROM_HOF_RANK = 1 (or 2, 3...).
     The script loads that rank from your saved GA output (hall_of_fame_results
     .json), runs it, AND prints the exact genome so you can paste it below.

  B) Paste that printed genome dict into PINNED_GENOME. Now the strategy is
     FROZEN into this file — it no longer depends on the JSON, so re-running the
     GA (which overwrites the hall of fame) can never change your baseline.
     This is the recommended state for a forward-test file: one file = one
     locked strategy.

Instrument settings (tick size, point value, contracts, commission) still come
from ga_core.py — change them there, restart the kernel, done.
"""
import os
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import ga_core as core
from ga_core import (StrategyGenome, ConditionGene, run_backtest,
                     load_and_preprocess_data, load_hall_of_fame,
                     compute_performance_metrics, dict_to_genome, genome_to_dict)

# =====================================================================
# CONFIG — edit these three things
# =====================================================================
# The data file to run the pinned strategy on. Point this at your forward data.
DATA_FILEPATH = "C:/Users/Administrator/Desktop/Stock Data/ES P&F 10-5 RTH.txt"

# If PINNED_GENOME is None, the strategy is loaded live from the saved GA output
# at this rank (and printed so you can freeze it). Once you paste a genome into
# PINNED_GENOME below, this is ignored.
PIN_FROM_HOF_RANK = 1

# Paste a genome dict here to FREEZE the strategy into this file. Example:
#   PINNED_GENOME = {"entry_trigger_type": "breakout", "entry_ref_close": False, ...}
# Leave as None to pull it live from the hall of fame at PIN_FROM_HOF_RANK.
PINNED_GENOME: Optional[Dict[str, Any]] = None

# Where outputs (trade CSV, yearly CSV, equity CSV) land — next to this script.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PREFIX = "ForwardTest"   # output files start with this


# =====================================================================
# human-readable strategy logic (self-contained copy — no import from the
# OOS Tester, so this file stands alone)
# =====================================================================
def format_condition(c: ConditionGene) -> str:
    if c.op in (">=", "<=", "<", ">"):
        return f"{c.var1}[t-{c.shift1}] {c.op} {c.var2}[t-{c.shift2}]"
    if c.op == "rising":
        return "Close[t-1] > Close[t-2]"
    if c.op == "falling":
        return "Close[t-1] < Close[t-2]"
    if c.op == "higher_x":
        return f"{c.var1}[t-1] > max({c.var1}, t-2..t-{c.x_bars + 1})"
    if c.op == "lower_x":
        return f"{c.var1}[t-1] < min({c.var1}, t-2..t-{c.x_bars + 1})"
    return str(c)


def describe_strategy(g: StrategyGenome) -> str:
    if g.entry_trigger_type == "breakout":
        ref = "Close[i-1]" if g.entry_ref_close else "High[i-1]"
        trig = f"High[i] > {ref} + {g.entry_offset_ticks} ticks  (breakout, pays spread)"
    else:
        ref = "Close[i-1]" if g.entry_ref_close else "Low[i-1]"
        trig = f"Low[i] < {ref} - {g.entry_offset_ticks} ticks  (dip, limit / no slippage)"

    if g.exit_style == "ticks":
        ref = "Close[i-1]" if g.exit_ref_close else "Low[i-1]"
        prot = (f"[SELECTED] Trailing stop off {ref}: STOP = {ref} - {g.exit_trigger_ticks} ticks "
                f"(ratchet up, arms bar entry+1). Fill pays 1-tick spread (bid)")
        time_exit = "Ignored (N-bar not selected)"
    elif g.exit_style == "atr":
        prot = f"[SELECTED] ATR({g.atr_period}) stop x{g.atr_sl_mult} / target x{g.atr_pt_mult}"
        time_exit = "Ignored (N-bar not selected)"
    elif g.exit_style == "nbars":
        prot = "Ignored (trailing/PT-SL not selected)"
        time_exit = f"[SELECTED] N-bar stop after {g.max_bars_hold} bars  -> fills at Open[i+1] - spread"
    else:  # cond
        prot = "Ignored (trailing/PT-SL not selected)"
        time_exit = "Ignored (N-bar not selected)"

    if g.exit_style == "cond":
        custom = f"[SELECTED] {format_condition(g.exit_cond)}  -> fills at Open[i+1] - spread"
    elif g.use_exit_cond:
        custom = f"{format_condition(g.exit_cond)}  (overlay)  -> fills at Open[i+1] - spread"
    else:
        custom = "Disabled"

    lines = [
        f"  Entry filter : {format_condition(g.entry_cond) if g.use_entry_cond else 'Disabled'}",
        f"  Entry trigger: {trig}",
        f"  Protective   : {prot}",
        f"  Time exit    : {time_exit}",
        f"  EOD exit     : always on -> flat at session close - spread",
        f"  Custom exit  : {custom}",
    ]
    return "\n".join(lines)


# =====================================================================
# strategy loading — pinned dict OR live from the hall of fame
# =====================================================================
def resolve_strategy() -> StrategyGenome:
    if PINNED_GENOME is not None:
        print("Strategy source: PINNED (frozen into this file — independent of the GA output).")
        return dict_to_genome(PINNED_GENOME)

    print(f"Strategy source: hall of fame '{core.HOF_JSON}', rank #{PIN_FROM_HOF_RANK}.")
    try:
        top = load_hall_of_fame(core.HOF_JSON)
    except FileNotFoundError:
        print(f"ERROR: '{core.HOF_JSON}' not found. Run PyGeneticAlgo.py first, "
              f"or paste a genome into PINNED_GENOME.")
        raise SystemExit(1)

    match = [g for rank, _, g in top if rank == PIN_FROM_HOF_RANK]
    if not match:
        print(f"ERROR: rank #{PIN_FROM_HOF_RANK} not found (loaded {len(top)} strategies).")
        raise SystemExit(1)
    g = match[0]

    # Print the genome so the user can freeze it into PINNED_GENOME.
    print("\nTo FREEZE this exact strategy into this file, paste the following into")
    print("PINNED_GENOME at the top of the script:\n")
    print("PINNED_GENOME = " + json.dumps(genome_to_dict(g), indent=4))
    print()
    return g


# =====================================================================
# yearly breakdown
# =====================================================================
def yearly_breakdown(trade_df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar year (bucketed by EXIT date): trades, win rate,
    Sharpe, and average trade value in $. Uses the same metrics engine as the
    overall stats, so numbers are consistent."""
    if trade_df.empty:
        return pd.DataFrame()

    tdf = trade_df.copy()
    tdf["year"] = pd.to_datetime(tdf["exit_dt"]).dt.year

    rows = []
    for year in sorted(tdf["year"].unique()):
        yr = tdf[tdf["year"] == year]
        m = compute_performance_metrics(yr)
        rows.append({
            "year": int(year),
            "trades": m["trade_count"],
            "win_rate_%": round(m["win_rate"] * 100, 1),
            "sharpe": round(m["sharpe_ratio"], 2),
            "avg_trade_$": round(m["avg_per_trade"], 2),
            "net_pnl_$": round(m["net_profit"], 2),
        })
    return pd.DataFrame(rows)


# =====================================================================
# main
# =====================================================================
if __name__ == "__main__":
    print("=" * 78)
    print(" FORWARD TEST — single pinned strategy")
    print("=" * 78)

    genome = resolve_strategy()

    print("-" * 78)
    print(" STRATEGY LOGIC")
    print("-" * 78)
    print(describe_strategy(genome))
    print("-" * 78)

    print(f"\nInstrument (from ga_core): TICK_SIZE={core.TICK_SIZE}  "
          f"POINT_VALUE=${core.POINT_VALUE}/pt  POSITION_SIZE={core.POSITION_SIZE}  "
          f"COMMISSION/side=${core.COMMISSION_PER_SIDE}")
    print(f"Data file: {DATA_FILEPATH}")

    df = load_and_preprocess_data(DATA_FILEPATH)   # full timeline, no year filter
    print(f"Loaded {len(df)} bars "
          f"({df['Datetime'].min()}  ->  {df['Datetime'].max()})\n")

    stats = run_backtest(genome, df)
    tdf = stats["trade_df"].copy()

    # ---- overall summary ----
    pf = "Inf" if stats["profit_factor"] == float("inf") else f"{stats['profit_factor']:.2f}"
    print("=" * 78)
    print(" OVERALL RESULT")
    print("=" * 78)
    print(f"  Net P&L        : ${stats['net_profit']:,.2f}")
    print(f"  Trades         : {stats['trade_count']}")
    print(f"  Win rate       : {stats['win_rate'] * 100:.1f}%")
    print(f"  Profit factor  : {pf}")
    print(f"  Sharpe         : {stats['sharpe_ratio']:.2f}")
    print(f"  Sortino        : {stats['sortino_ratio']:.2f}")
    print(f"  Avg $/trade    : ${stats['avg_per_trade']:,.2f}")
    print(f"  Max drawdown   : ${stats['max_dd']:,.2f}")

    # ---- yearly breakdown ----
    ybd = yearly_breakdown(tdf)
    print("\n" + "=" * 78)
    print(" YEARLY BREAKDOWN")
    print("=" * 78)
    if ybd.empty:
        print("  (no trades)")
    else:
        print(f"  {'Year':<6} | {'Trades':>7} | {'Win %':>7} | {'Sharpe':>7} | "
              f"{'Avg $/Trade':>12} | {'Net P&L $':>13}")
        print("  " + "-" * 70)
        for _, r in ybd.iterrows():
            print(f"  {int(r['year']):<6} | {int(r['trades']):>7} | {r['win_rate_%']:>7.1f} | "
                  f"{r['sharpe']:>7.2f} | {r['avg_trade_$']:>12,.2f} | {r['net_pnl_$']:>13,.2f}")

    # =====================================================================
    # CSV OUTPUTS
    # =====================================================================
    if not tdf.empty:
        # split-out explicit time columns so entry/exit times are unmissable,
        # while entry_dt/exit_dt keep the full date+time stamp.
        tdf["entry_dt"] = pd.to_datetime(tdf["entry_dt"])
        tdf["exit_dt"] = pd.to_datetime(tdf["exit_dt"])
        tdf["entry_time"] = tdf["entry_dt"].dt.strftime("%H:%M:%S")
        tdf["exit_time"] = tdf["exit_dt"].dt.strftime("%H:%M:%S")
        tdf["pnl"] = tdf["pnl"].round(2)   # drop float noise (-5.70000001 -> -5.70)

        cols = ["entry_dt", "entry_time", "exit_dt", "exit_time",
                "entry_price", "exit_price", "reason", "pnl", "bars_held"]
        trades_csv = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_trades.csv")
        tdf[cols].to_csv(trades_csv, index=False)
        print(f"\nTrade log      -> {trades_csv}")

        # yearly breakdown CSV
        yearly_csv = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_yearly.csv")
        ybd.to_csv(yearly_csv, index=False)
        print(f"Yearly summary -> {yearly_csv}")

        # equity-curve CSV (running P&L by exit) — free add-on for charting live vs backtest
        eq = tdf[["exit_dt"]].copy()
        eq["cum_pnl"] = tdf["pnl"].cumsum().round(2)
        equity_csv = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_equity.csv")
        eq.to_csv(equity_csv, index=False)
        print(f"Equity curve   -> {equity_csv}")
    else:
        print("\nNo trades generated — nothing written.")

    print("\nDone.")
