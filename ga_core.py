#!/usr/bin/env python3
"""
ga_core.py  —  SINGLE SOURCE OF TRUTH for the GA generator + OOS tester.
========================================================================
Both PyGeneticAlgo.py (optimizer) and "OOS Tester.py" (validator) import
EVERYTHING strategy-related from here: the constants, the genome definition,
the data loader, the condition evaluator, and THE ONE backtest engine.

Why this file exists
--------------------
Your two scripts had drifted into two different strategies:
  - different genome fields (ATR stops in one, tick offsets in the other),
  - different exit engines (ATR SL/PT vs no stops at all),
  - different session times (08:31 vs 08:45),
  - a JSON handoff that didn't line up (filename + schema).
That is why every time one file was revised, the other broke. With a single
shared engine, the optimizer and the validator are GUARANTEED to run the exact
same simulation — an in-sample winner reproduces bar-for-bar out-of-sample.

Genome (superset — all of it is evolvable)
------------------------------------------
  entry : breakout / dip, with an entry_offset_ticks gene
  exit  : exit_style gene picks the protective exit -
            'ticks' -> stop at entry-exit_trigger_ticks, limit capped
                       exit_offset_ticks below it (your v2 / LE model)
            'atr'   -> ATR stop-loss / profit-target (your original generator)
          plus structural exits shared by both: max_bars_hold, optional
          exit condition, and end-of-day.

Fill model for the 'ticks' exit
-------------------------------
  FILL_MODE='stop'  -> pessimistic: fills at min(Open, limit)  (worst case)
  FILL_MODE='limit' -> LE cap: max(min(Open, trigger), limit) -> never worse
                       than the limit. The net gap between the two runs is
                       exactly what your LE exit is saving in slippage.
"""
import json
import random
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ==========================================
# 1. INSTRUMENT & EXECUTION CONSTANTS
# ==========================================
TICK_SIZE           = 0.01
POINT_VALUE         = 1.00
SPREAD_SLIPPAGE     = 0.01        # $/share adverse fill assumption (stop-style fills)
COMMISSION_PER_SIDE = 0.0035      # $/share/side  ($0.70 round-trip on 100 sh)
POSITION_SIZE       = 100

# ONE session, used by BOTH scripts (was 08:31 in generator, 08:45 in tester)
SESSION_START = "08:31:00"
SESSION_END   = "15:00:00"

PRICE_FIELDS = ["Open", "High", "Low", "Close"]

# In-sample year window the optimizer trains on (the tester treats everything
# outside this as out-of-sample).
IS_START_YEAR = 2024
IS_END_YEAR   = 2025

# JSON handoff — ONE filename, shared by both scripts.
HOF_JSON  = "hall_of_fame_results.json"
BEST_JSON = "best_strategy.json"

WARMUP = 20   # enough bars for ATR(20), higher_x(10), and shifts

# Default fill model for the 'ticks' exit style ('stop' or 'limit').
FILL_MODE = "stop"


# ==========================================
# 2. DATA LOAD & PREPROCESSING
# ==========================================
def load_and_preprocess_data(filepath: str,
                             start_year: int = None,
                             end_year: int = None) -> pd.DataFrame:
    """Load bar data, optionally filter to a year window, precompute ATR(5..20).

    Pass start_year/end_year=None to load the FULL timeline (what the OOS tester
    wants); pass a window (e.g. 2024..2025) for the optimizer's in-sample slice.
    """
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    if "Last" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Last"]

    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str), format="mixed"
    )
    df = df.sort_values("Datetime").reset_index(drop=True)

    if start_year is not None and end_year is not None:
        df = df[(df["Datetime"].dt.year >= start_year)
                & (df["Datetime"].dt.year <= end_year)].reset_index(drop=True)

    df["Time_Only"] = df["Datetime"].dt.time

    # ATR family (precomputed once so the engine just reads a column)
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close  = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    for period in range(5, 21):
        df[f"ATR_{period}"] = tr.rolling(period).mean()

    return df


# ==========================================
# 3. GENOME
# ==========================================
@dataclass
class ConditionGene:
    var1: str
    shift1: int
    op: str
    var2: str
    shift2: int
    x_bars: int


@dataclass
class StrategyGenome:
    # --- entry ---
    use_entry_cond: bool
    entry_cond: ConditionGene
    entry_ref_var: str
    entry_shift: int
    entry_trigger_type: str        # "breakout" or "dip"
    entry_offset_ticks: int

    # --- protective exit selector ---
    exit_style: str                # "ticks" or "atr"
    exit_trigger_ticks: int        # ticks stop below entry   (exit_style="ticks")
    exit_offset_ticks: int         # LE limit offset below the trigger
    atr_period: int
    atr_sl_mult: float
    atr_pt_mult: float

    # --- structural exits (shared) ---
    max_bars_hold: int
    use_exit_cond: bool
    exit_cond: ConditionGene


def random_condition() -> ConditionGene:
    return ConditionGene(
        var1=random.choice(PRICE_FIELDS),
        shift1=random.randint(1, 3),
        op=random.choice([">=", "<=", "<", ">", "rising", "falling", "higher_x", "lower_x"]),
        var2=random.choice(PRICE_FIELDS),
        shift2=random.randint(1, 3),
        x_bars=random.randint(2, 10),
    )


def create_random_genome() -> StrategyGenome:
    return StrategyGenome(
        use_entry_cond=random.choice([True, False]),
        entry_cond=random_condition(),
        entry_ref_var=random.choice(PRICE_FIELDS),
        entry_shift=random.randint(1, 3),
        entry_trigger_type=random.choice(["breakout", "dip"]),
        entry_offset_ticks=random.randint(0, 6),

        exit_style=random.choice(["ticks", "atr"]),
        exit_trigger_ticks=random.randint(1, 12),
        exit_offset_ticks=random.randint(0, 6),
        atr_period=random.randint(5, 20),
        atr_sl_mult=round(random.uniform(0.5, 2.5), 2),
        atr_pt_mult=round(random.uniform(0.5, 2.5), 2),

        max_bars_hold=random.randint(1, 10),
        use_exit_cond=random.choice([True, False]),
        exit_cond=random_condition(),
    )


# ==========================================
# 4. CONDITION EVALUATOR
# ==========================================
def evaluate_condition(cond: ConditionGene, df: pd.DataFrame, idx: int) -> bool:
    if idx < WARMUP:
        return False
    if idx - cond.shift1 < 0 or idx - cond.shift2 < 0:
        return False

    v1 = df.at[idx - cond.shift1, cond.var1]
    v2 = df.at[idx - cond.shift2, cond.var2]

    if cond.op == ">=":
        return v1 >= v2
    if cond.op == "<=":
        return v1 <= v2
    if cond.op == "<":
        return v1 < v2
    if cond.op == ">":
        return v1 > v2
    if cond.op == "rising":
        return df.at[idx, "Close"] > df.at[idx - 1, "Close"]
    if cond.op == "falling":
        return df.at[idx, "Close"] < df.at[idx - 1, "Close"]
    if cond.op == "higher_x":
        x = cond.x_bars
        if idx - x < 0:
            return False
        return df.at[idx, cond.var1] > df.loc[idx - x: idx - 1, cond.var1].max()
    if cond.op == "lower_x":
        x = cond.x_bars
        if idx - x < 0:
            return False
        return df.at[idx, cond.var1] < df.loc[idx - x: idx - 1, cond.var1].min()
    return False


# ==========================================
# 5. THE ONE BACKTEST ENGINE
# ==========================================
def run_backtest(genome: StrategyGenome, df: pd.DataFrame,
                 fill_mode: str = None) -> Dict[str, Any]:
    """Single engine used by BOTH the optimizer and the OOS tester."""
    if fill_mode is None:
        fill_mode = FILL_MODE

    n = len(df)
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    sl_price = pt_price = 0.0          # atr style
    trig_price = lim_price = 0.0       # ticks style

    trades = []
    start_t = pd.to_datetime(SESSION_START).time()
    end_t = pd.to_datetime(SESSION_END).time()

    for i in range(WARMUP, n):
        curr_time = df.at[i, "Time_Only"]

        # ---------------- POSITION MANAGEMENT ----------------
        if in_position:
            bars_held = i - entry_idx
            curr_high = df.at[i, "High"]
            curr_low  = df.at[i, "Low"]
            curr_open = df.at[i, "Open"]
            curr_close = df.at[i, "Close"]

            exit_triggered = False
            exec_exit = 0.0
            exit_reason = ""

            if genome.exit_style == "atr":
                hit_sl = curr_low <= sl_price
                hit_pt = curr_high > pt_price
                if hit_sl:                                  # SL priority (conservative)
                    exit_triggered = True
                    exec_exit = sl_price - SPREAD_SLIPPAGE
                    exit_reason = "SL"
                elif hit_pt:
                    exit_triggered = True
                    exec_exit = pt_price - SPREAD_SLIPPAGE
                    exit_reason = "PT"
            else:  # "ticks" -> stop / stop-limit (your v2 / LE model)
                if curr_low <= trig_price:
                    exit_triggered = True
                    if fill_mode == "limit":
                        # LE cap: fill where a stop would, but never worse than
                        # the limit. A straight gap below the limit rests unfilled
                        # live (the .cpp time-backstop markets you out); here it
                        # is capped -> treat 'limit' as the optimistic bound.
                        exec_exit = max(min(curr_open, trig_price), lim_price)
                    else:
                        # pessimistic stop fill
                        exec_exit = min(curr_open, lim_price)
                    exit_reason = "Stop"

            if not exit_triggered:
                if curr_time >= end_t:
                    exit_triggered = True
                    exec_exit = curr_close - SPREAD_SLIPPAGE
                    exit_reason = "EOD"
                elif bars_held >= genome.max_bars_hold:
                    exit_triggered = True
                    exec_exit = curr_close - SPREAD_SLIPPAGE
                    exit_reason = "N-Bars"
                elif genome.use_exit_cond and evaluate_condition(genome.exit_cond, df, i):
                    exit_triggered = True
                    exec_exit = curr_close - SPREAD_SLIPPAGE
                    exit_reason = "Exit Logic"

            if exit_triggered:
                gross_pnl = (exec_exit - entry_price) * POSITION_SIZE * POINT_VALUE
                net_pnl = gross_pnl - 2 * COMMISSION_PER_SIDE * POSITION_SIZE
                trades.append({
                    "entry_dt": df.at[entry_idx, "Datetime"],
                    "exit_dt":  df.at[i, "Datetime"],
                    "entry_price": entry_price,
                    "exit_price":  exec_exit,
                    "pnl": net_pnl,
                    "bars_held": bars_held,
                    "reason": exit_reason,
                })
                in_position = False
                continue

        # ---------------- ENTRY ----------------
        if not in_position:
            if not (start_t <= curr_time < end_t):
                continue
            if i - genome.entry_shift < 0:
                continue
            if genome.use_entry_cond and not evaluate_condition(genome.entry_cond, df, i):
                continue

            base_val = df.at[i - genome.entry_shift, genome.entry_ref_var]
            curr_open = df.at[i, "Open"]
            triggered = False

            if genome.entry_trigger_type == "breakout":
                ref_val = base_val + genome.entry_offset_ticks * TICK_SIZE
                if df.at[i, "High"] > ref_val:
                    triggered = True
                    entry_price = max(ref_val, curr_open) + SPREAD_SLIPPAGE  # adverse
            else:  # "dip" -> buy limit
                ref_val = base_val - genome.entry_offset_ticks * TICK_SIZE
                if df.at[i, "Low"] < ref_val:
                    triggered = True
                    entry_price = min(ref_val, curr_open)                     # limit, no adverse

            if triggered:
                in_position = True
                entry_idx = i
                if genome.exit_style == "atr":
                    atr_val = df.at[i, f"ATR_{genome.atr_period}"]
                    if pd.isna(atr_val) or atr_val <= 0:
                        atr_val = 0.50
                    sl_price = entry_price - atr_val * genome.atr_sl_mult
                    pt_price = entry_price + atr_val * genome.atr_pt_mult
                else:  # ticks
                    trig_price = entry_price - genome.exit_trigger_ticks * TICK_SIZE
                    lim_price  = trig_price - genome.exit_offset_ticks * TICK_SIZE

    return compute_performance_metrics(pd.DataFrame(trades))


# ==========================================
# 6. METRICS
# ==========================================
def compute_performance_metrics(trade_df: pd.DataFrame,
                                position_size: int = POSITION_SIZE) -> Dict[str, Any]:
    if trade_df.empty:
        return {"net_profit": 0.0, "trade_count": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_per_trade": 0.0, "avg_per_share": 0.0,
                "max_dd": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
                "fitness": -9999.0, "equity_curve": np.array([0]),
                "trade_df": pd.DataFrame()}

    pnls = trade_df["pnl"].values
    trade_count = len(pnls)
    net_profit = float(np.sum(pnls))

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / trade_count
    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 \
        else (float("inf") if gross_profit > 0 else 0.0)

    avg_per_trade = net_profit / trade_count
    avg_per_share = avg_per_trade / position_size if position_size else 0.0

    equity_curve = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity_curve)
    max_dd = float(np.max(peak - equity_curve)) if len(equity_curve) else 0.0

    # daily-resampled Sharpe / Sortino
    tmp = trade_df.copy()
    tmp["cum_pnl"] = tmp["pnl"].cumsum()
    daily_equity = (tmp.groupby("exit_dt")["cum_pnl"].last()
                    .resample("D").ffill().fillna(0))
    daily_pnl = daily_equity.diff().fillna(daily_equity.iloc[0] if len(daily_equity) else 0)
    std_d, mean_d = float(daily_pnl.std()), float(daily_pnl.mean())
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0.0
    neg = daily_pnl[daily_pnl < 0]
    dstd = float(neg.std()) if len(neg) > 1 else 0.0
    sortino = (mean_d / dstd) * np.sqrt(252) if dstd > 0 else 0.0

    # fitness: reward net, punish drawdown and thin samples
    fitness = net_profit - 1.5 * max_dd
    if trade_count < 10:
        fitness -= 500

    return {"net_profit": net_profit, "trade_count": trade_count, "win_rate": win_rate,
            "profit_factor": profit_factor, "avg_per_trade": avg_per_trade,
            "avg_per_share": avg_per_share, "max_dd": max_dd,
            "sharpe_ratio": sharpe, "sortino_ratio": sortino, "fitness": fitness,
            "equity_curve": equity_curve, "trade_df": trade_df}


# ==========================================
# 7. JSON HANDOFF  (write in optimizer, read in tester — one schema)
# ==========================================
def genome_to_dict(g: StrategyGenome) -> Dict[str, Any]:
    return asdict(g)


def dict_to_genome(d: Dict[str, Any]) -> StrategyGenome:
    d = dict(d)
    d["entry_cond"] = ConditionGene(**d["entry_cond"])
    d["exit_cond"] = ConditionGene(**d["exit_cond"])
    return StrategyGenome(**d)


def save_hall_of_fame(hall_of_fame: List[Tuple[StrategyGenome, Dict[str, Any]]],
                      filename: str = HOF_JSON) -> None:
    export = []
    for rank, (genome, stats) in enumerate(hall_of_fame, start=1):
        export.append({
            "rank": rank,
            "metrics": {
                "fitness": round(stats["fitness"], 2),
                "net_profit": round(stats["net_profit"], 2),
                "trade_count": stats["trade_count"],
                "win_rate": round(stats["win_rate"], 4),
                "max_dd": round(stats["max_dd"], 2),
            },
            "genome": genome_to_dict(genome),
        })
    with open(filename, "w") as f:
        json.dump(export, f, indent=4)
    print(f"Saved top {len(export)} strategies -> '{filename}'")
    if hall_of_fame:
        with open(BEST_JSON, "w") as f:
            json.dump(genome_to_dict(hall_of_fame[0][0]), f, indent=4)
        print(f"Rank-1 genome -> '{BEST_JSON}'")


def load_hall_of_fame(filename: str = HOF_JSON
                      ) -> List[Tuple[int, Dict[str, Any], StrategyGenome]]:
    with open(filename, "r") as f:
        data = json.load(f)
    out = []
    for item in data:
        out.append((item["rank"], item.get("metrics", {}), dict_to_genome(item["genome"])))
    return out
