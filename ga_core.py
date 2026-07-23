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

Genome (all of it is evolvable)
-------------------------------
  entry : breakout / dip, trigger off bar [i-1], with an entry_offset_ticks gene
  exit  : exit_style gene picks the protective exit -
            'ticks' -> trailing stop off prev Close or Low, exit_trigger_ticks away
            'atr'   -> ATR stop-loss / profit-target
          plus structural exits: max_bars_hold, optional exit condition, end-of-day.

Exit fills
----------
  Every exit pays a flat 1-tick spread (SPREAD_SLIPPAGE), i.e. fills at the bid.
  No LE / limit model — kept simple per the strategy spec.
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
EXIT_SPREAD_TICKS   = 1           # every exit pays this many ticks of spread (fills at bid)
SPREAD_SLIPPAGE     = EXIT_SPREAD_TICKS * TICK_SIZE
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
    # Optional filter condition (may look back over completed bars i-1..i-3,
    # e.g. C[i-1] > L[i-2]).  The TRIGGER itself keys off the most recent
    # completed bar [i-1].
    use_entry_cond: bool
    entry_cond: ConditionGene
    entry_trigger_type: str        # "breakout" or "dip"
    entry_ref_close: bool          # True -> reference Close[i-1]; False -> High[i-1] (breakout) / Low[i-1] (dip)
    entry_offset_ticks: int

    # --- protective exit selector ---
    exit_style: str                # "ticks" (trailing stop) or "atr"
    exit_ref_close: bool           # trail off Close[i-1] (True) or Low[i-1] (False)
    exit_trigger_ticks: int        # trail distance below the reference (exit_style="ticks")
    atr_period: int
    atr_sl_mult: float
    atr_pt_mult: float

    # --- structural exits (shared) ---
    max_bars_hold: int
    use_exit_cond: bool
    exit_cond: ConditionGene


def random_condition() -> ConditionGene:
    var1, shift1 = random.choice(PRICE_FIELDS), random.randint(1, 3)
    op = random.choice([">=", "<=", "<", ">", "rising", "falling", "higher_x", "lower_x"])
    var2, shift2 = random.choice(PRICE_FIELDS), random.randint(1, 3)
    # avoid a degenerate self-comparison (e.g. Low[t-2] > Low[t-2]) for the
    # binary comparison operators — it's always a tautology or contradiction.
    if op in (">=", "<=", "<", ">") and var1 == var2 and shift1 == shift2:
        shift2 = shift1 % 3 + 1
    return ConditionGene(var1=var1, shift1=shift1, op=op, var2=var2, shift2=shift2,
                         x_bars=random.randint(2, 10))


def create_random_genome() -> StrategyGenome:
    return StrategyGenome(
        use_entry_cond=random.choice([True, False]),
        entry_cond=random_condition(),
        entry_trigger_type=random.choice(["breakout", "dip"]),
        entry_ref_close=random.choice([True, False]),
        entry_offset_ticks=random.randint(0, 6),

        exit_style=random.choice(["ticks", "atr"]),
        exit_ref_close=random.choice([True, False]),
        exit_trigger_ticks=random.randint(0, 12),
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
    # NOTE: entry filters are evaluated while bar `idx` is still forming, so they
    # must ONLY reference COMPLETED bars.  The most recent completed bar is idx-1.
    # (The binary comparisons above already do this via shift1/shift2 >= 1.)
    if cond.op == "rising":                                    # last completed bar rose
        return df.at[idx - 1, "Close"] > df.at[idx - 2, "Close"]
    if cond.op == "falling":                                   # last completed bar fell
        return df.at[idx - 1, "Close"] < df.at[idx - 2, "Close"]
    if cond.op == "higher_x":                                  # last completed bar is highest of the prior x
        x = cond.x_bars
        if idx - 1 - x < 0:
            return False
        return df.at[idx - 1, cond.var1] > df.loc[idx - 1 - x: idx - 2, cond.var1].max()
    if cond.op == "lower_x":                                   # last completed bar is lowest of the prior x
        x = cond.x_bars
        if idx - 1 - x < 0:
            return False
        return df.at[idx - 1, cond.var1] < df.loc[idx - 1 - x: idx - 2, cond.var1].min()
    return False


# ==========================================
# 5. THE ONE BACKTEST ENGINE
# ==========================================
def run_backtest(genome: StrategyGenome, df: pd.DataFrame) -> Dict[str, Any]:
    """Single engine used by BOTH the optimizer and the OOS tester.

    Every exit pays a flat 1-tick spread (fills at the bid). No LE / limit model.
    """
    n = len(df)
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    sl_price = pt_price = 0.0          # atr style
    trail_stop = -1e18                 # ticks style, ratchet-up-only trailing stop
    pending_exit = False               # signal exit decided last bar, fills at THIS open
    pending_reason = ""

    trades = []
    start_t = pd.to_datetime(SESSION_START).time()
    end_t = pd.to_datetime(SESSION_END).time()

    trig_ticks = genome.exit_trigger_ticks * TICK_SIZE

    def close_trade(exit_i, exec_exit, reason):
        gross = (exec_exit - entry_price) * POSITION_SIZE * POINT_VALUE
        net = gross - 2 * COMMISSION_PER_SIDE * POSITION_SIZE
        trades.append({
            "entry_dt": df.at[entry_idx, "Datetime"], "exit_dt": df.at[exit_i, "Datetime"],
            "entry_price": entry_price, "exit_price": exec_exit,
            "pnl": net, "bars_held": exit_i - entry_idx, "reason": reason,
        })

    for i in range(WARMUP, n):
        curr_time  = df.at[i, "Time_Only"]
        curr_high  = df.at[i, "High"]
        curr_low   = df.at[i, "Low"]
        curr_open  = df.at[i, "Open"]
        curr_close = df.at[i, "Close"]

        # ---------------- POSITION MANAGEMENT ----------------
        if in_position:
            # (0) a signal exit (N-bars / exit-cond) decided on the PREVIOUS bar
            #     fills at THIS bar's open, minus spread.
            if pending_exit:
                close_trade(i, curr_open - SPREAD_SLIPPAGE, pending_reason)
                in_position = False
                pending_exit = False
                # fall through: this bar is now flat and may re-enter below
            else:
                exited = False

                # (1) protective exit — intrabar, same-bar fill. Free pass on the
                #     entry bar: the trailing stop only arms from entry_idx + 1.
                if i > entry_idx:
                    if genome.exit_style == "atr":
                        if curr_low <= sl_price:                       # SL priority
                            close_trade(i, sl_price - SPREAD_SLIPPAGE, "SL"); exited = True
                        elif curr_high > pt_price:
                            close_trade(i, pt_price - SPREAD_SLIPPAGE, "PT"); exited = True
                    else:  # "ticks" -> trailing stop off the PREVIOUS bar (v2)
                        # reference is prev Close (L<=C) or prev Low (L<=L[i-1]-ticks)
                        ref_base = df.at[i - 1, "Close"] if genome.exit_ref_close else df.at[i - 1, "Low"]
                        new_trig = ref_base - trig_ticks
                        if new_trig > trail_stop:                      # ratchet up only
                            trail_stop = new_trig
                        if curr_low <= trail_stop:
                            # stop sells into the bid -> pay the 1-tick spread
                            exec_exit = min(curr_open, trail_stop) - SPREAD_SLIPPAGE
                            close_trade(i, exec_exit, "Trail"); exited = True

                # (2) end-of-day — fills same bar at the close (session boundary)
                if not exited and curr_time >= end_t:
                    close_trade(i, curr_close - SPREAD_SLIPPAGE, "EOD"); exited = True

                # (3) N-bars / exit-condition — decide now, FILL AT NEXT OPEN
                if not exited:
                    bars_held = i - entry_idx
                    if bars_held >= genome.max_bars_hold:
                        pending_exit, pending_reason = True, "N-Bars"
                    elif genome.use_exit_cond and evaluate_condition(genome.exit_cond, df, i):
                        pending_exit, pending_reason = True, "Exit Logic"

                if exited:
                    in_position = False
                if in_position:
                    continue   # still holding; do not look for a new entry this bar

        # ---------------- ENTRY ----------------
        if not in_position:
            if not (start_t <= curr_time < end_t):
                continue
            if i < 1:
                continue
            # optional filter: may reference completed bars i-1..i-3
            if genome.use_entry_cond and not evaluate_condition(genome.entry_cond, df, i):
                continue

            triggered = False
            if genome.entry_trigger_type == "breakout":
                # trigger off the most recent completed bar [i-1]: High or Close
                base = df.at[i - 1, "Close"] if genome.entry_ref_close else df.at[i - 1, "High"]
                ref_val = base + genome.entry_offset_ticks * TICK_SIZE
                if curr_high > ref_val:
                    triggered = True
                    entry_price = max(ref_val, curr_open) + SPREAD_SLIPPAGE   # pays spread
            else:  # "dip" -> buy limit off [i-1]: Low or Close
                base = df.at[i - 1, "Close"] if genome.entry_ref_close else df.at[i - 1, "Low"]
                ref_val = base - genome.entry_offset_ticks * TICK_SIZE
                if curr_low < ref_val:
                    triggered = True
                    entry_price = min(ref_val, curr_open)                     # limit, no slippage

            if triggered:
                in_position = True
                entry_idx = i
                trail_stop = -1e18
                pending_exit = False
                if genome.exit_style == "atr":
                    atr_val = df.at[i, f"ATR_{genome.atr_period}"]
                    if pd.isna(atr_val) or atr_val <= 0:
                        atr_val = 0.50
                    sl_price = entry_price - atr_val * genome.atr_sl_mult
                    pt_price = entry_price + atr_val * genome.atr_pt_mult

    # close any dangling position at the last bar
    if in_position:
        close_trade(n - 1, df.at[n - 1, "Close"] - SPREAD_SLIPPAGE, "Data_End")

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
    # tolerate JSON written by an older genome layout: drop unknown keys.
    valid = set(StrategyGenome.__dataclass_fields__.keys())
    d = {k: v for k, v in d.items() if k in valid}
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
