#!/usr/bin/env python3
"""
Export Strategy.py  —  turn one GA strategy into a STANDALONE .py file.
======================================================================
The Forward Test file RUNS a pinned strategy through ga_core. This file does
something different: it WRITES OUT a brand-new, self-contained Python script for
one chosen strategy, with the entry/exit logic spelled out as real code (plain
if-statements, not a genome dict). You get one portable .py per strategy —
no ga_core import, no hall-of-fame dependency — that you can open, read, tweak
the instrument config on, and run anywhere.

Why not just fill your own template?
------------------------------------
Two reasons the generated file is safer than hand-filling a template:
  1. It reproduces the ga_core engine BAR-FOR-BAR — same tick snapping, entry
     spread, ATR, exit ordering, entry-bar SL/PT, no same-bar re-entry, AND the
     session-last-bar flatten that stops trades teleporting across the overnight
     gap. Every honesty guard is baked in, so the exported file's numbers TIE
     OUT with the GA and the Forward Test (this script verifies that on export).
  2. It handles EVERY strategy type the GA can produce — breakout / dip entry,
     ATR / trailing / N-bar / condition exits, optional entry filter and custom
     exit overlay — not just one hardcoded family.

Usage
-----
  - Point DATA_FILEPATH at a data file (used only to VERIFY the export matches).
  - Pick the strategy: PINNED_GENOME (frozen dict) OR PIN_FROM_HOF_RANK.
  - Run. It writes  <OUTPUT_DIR>/<OUT_PY_NAME>  and prints PASS/FAIL after
    checking the generated file reproduces ga_core exactly on your data.
Then edit the INSTRUMENT CONFIG block at the top of the generated file for your
instrument (tick / spread / point value / contracts / commission) and run it.
"""
import os
import json
import datetime as _dt
from typing import Any, Dict, Optional

import ga_core as core
from ga_core import (StrategyGenome, ConditionGene, load_hall_of_fame,
                     dict_to_genome, genome_to_dict)

# =====================================================================
# CONFIG
# =====================================================================
DATA_FILEPATH = "C:/Users/Administrator/Desktop/Stock Data/ES P&F 10-5 RTH.txt"

# Strategy to export: leave PINNED_GENOME = None to pull PIN_FROM_HOF_RANK from
# the hall of fame, or paste a genome dict to freeze a specific one.
PIN_FROM_HOF_RANK = 1
PINNED_GENOME: Optional[Dict[str, Any]] = None

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PY_NAME = "Strategy_Standalone.py"   # the generated file's name


# =====================================================================
# condition -> python expression (mirrors ga_core.evaluate_condition)
# =====================================================================
_VAR = {"Open": "O", "High": "H", "Low": "L", "Close": "C"}


def cond_to_expr(c: ConditionGene) -> str:
    """Return a Python expression (over arrays O/H/L/C, index i) that evaluates
    identically to ga_core.evaluate_condition for this ConditionGene."""
    if c.op in (">=", "<=", "<", ">"):
        return f"{_VAR[c.var1]}[i-{c.shift1}] {c.op} {_VAR[c.var2]}[i-{c.shift2}]"
    if c.op == "rising":
        return "C[i-1] > C[i-2]"
    if c.op == "falling":
        return "C[i-1] < C[i-2]"
    if c.op == "higher_x":
        v = _VAR[c.var1]
        return f"{v}[i-1] > {v}[i-1-{c.x_bars}:i-1].max()"
    if c.op == "lower_x":
        v = _VAR[c.var1]
        return f"{v}[i-1] < {v}[i-1-{c.x_bars}:i-1].min()"
    return "False"


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
        prot = (f"[SELECTED] Trailing stop: STOP = {ref} - {g.exit_trigger_ticks} ticks "
                f"(ratchet up, arms bar entry+1), fill pays 1-tick spread")
    elif g.exit_style == "atr":
        prot = f"[SELECTED] ATR({g.atr_period}) stop x{g.atr_sl_mult} / target x{g.atr_pt_mult}"
    elif g.exit_style == "nbars":
        prot = f"[SELECTED] N-bar stop after {g.max_bars_hold} bars -> fills at Open[i+1] - spread"
    else:
        prot = f"[SELECTED] condition exit -> fills at Open[i+1] - spread"
    if g.exit_style == "cond":
        custom = f"{format_condition(g.exit_cond)}  -> fills at Open[i+1] - spread (this IS the exit)"
    elif g.use_exit_cond:
        custom = f"{format_condition(g.exit_cond)}  (overlay) -> fills at Open[i+1] - spread"
    else:
        custom = "Disabled"
    return "\n".join([
        f"  Entry filter : {format_condition(g.entry_cond) if g.use_entry_cond else 'Disabled'}",
        f"  Entry trigger: {trig}",
        f"  Protective   : {prot}",
        f"  EOD exit     : always on -> flat at the session's last bar - spread",
        f"  Custom exit  : {custom}",
    ])


# =====================================================================
# code-block builders (return correctly-indented python source)
# =====================================================================
def build_entry_block(g: StrategyGenome) -> str:
    """The entry trigger + (optional) filter, as source lines at 12-space indent."""
    L = []
    ind = " " * 12
    if g.use_entry_cond:
        L.append(f"{ind}if not ({cond_to_expr(g.entry_cond)}):")
        L.append(f"{ind}    pass  # entry filter failed")
        L.append(f"{ind}else:")
        ind2 = " " * 16
    else:
        ind2 = " " * 12
    if g.entry_trigger_type == "breakout":
        base = "C[i-1]" if g.entry_ref_close else "H[i-1]"
        L.append(f"{ind2}ref_val = {base} + {g.entry_offset_ticks} * TICK")
        L.append(f"{ind2}if H[i] > ref_val:")
        L.append(f"{ind2}    entry_price = round_tick(max(ref_val, O[i]) + ENTRY_SPREAD)")
        L.append(f"{ind2}    triggered = True")
    else:  # dip
        base = "C[i-1]" if g.entry_ref_close else "L[i-1]"
        L.append(f"{ind2}ref_val = {base} - {g.entry_offset_ticks} * TICK")
        L.append(f"{ind2}if L[i] < ref_val:")
        L.append(f"{ind2}    entry_price = round_tick(min(ref_val, O[i]))")
        L.append(f"{ind2}    triggered = True")
    return "\n".join(L)


def build_intrabar_exit(g: StrategyGenome) -> str:
    """Section (1): the protective intrabar exit for bars AFTER entry."""
    ind = " " * 20
    if g.exit_style == "atr":
        return "\n".join([
            f"{ind}if L[i] <= sl_price:                      # SL = market: pays spread, gap-fills at worse of open/stop",
            f"{ind}    close_trade(i, min(O[i], sl_price) - SPREAD, 'SL'); exited = True",
            f"{ind}elif H[i] > pt_price + SPREAD:            # PT = limit: trade a spread through, fill AT target (no spread), gap-fill at better of open/target",
            f"{ind}    close_trade(i, max(O[i], pt_price), 'PT'); exited = True",
        ])
    if g.exit_style == "ticks":
        ref = "C[i-1]" if g.exit_ref_close else "L[i-1]"
        return "\n".join([
            f"{ind}new_trig = round_tick({ref} - {g.exit_trigger_ticks} * TICK)",
            f"{ind}if new_trig > trail_stop:                 # ratchet up only",
            f"{ind}    trail_stop = new_trig",
            f"{ind}if L[i] <= trail_stop:",
            f"{ind}    exec_exit = min(O[i], trail_stop) - SPREAD",
            f"{ind}    close_trade(i, exec_exit, 'Trail'); exited = True",
        ])
    return f"{ind}pass  # no intrabar protective exit for this style"


def build_signal_exit(g: StrategyGenome) -> str:
    """Section (3): signal exits that fill at NEXT open (N-bar / condition)."""
    ind = " " * 20
    L = [f"{ind}bars_held = i - entry_idx"]
    parts = []
    if g.exit_style == "nbars":
        parts.append((f"bars_held >= {g.max_bars_hold}", "'N-Bars'"))
    cond_active = (g.exit_style == "cond") or g.use_exit_cond
    if cond_active:
        parts.append((f"({cond_to_expr(g.exit_cond)})", "'Exit Logic'"))
    if not parts:
        L.append(f"{ind}pass  # no signal exit for this style")
        return "\n".join(L)
    first = True
    for expr, reason in parts:
        kw = "if" if first else "elif"
        L.append(f"{ind}{kw} {expr}:")
        L.append(f"{ind}    pending_exit, pending_reason = True, {reason}")
        first = False
    return "\n".join(L)


def build_entrybar_atr(g: StrategyGenome) -> str:
    """The entry-bar SL/PT arming (ATR only) + the SL/PT level setup."""
    ind = " " * 16
    if g.exit_style != "atr":
        return f"{ind}pass  # SL/PT levels only used by the ATR exit style"
    return "\n".join([
        f"{ind}atr_val = atr[i]",
        f"{ind}if atr_val != atr_val or atr_val <= 0:   # NaN or non-positive -> fallback",
        f"{ind}    atr_val = 0.50",
        f"{ind}sl_price = round_tick(entry_price - atr_val * {g.atr_sl_mult})",
        f"{ind}pt_price = round_tick(entry_price + atr_val * {g.atr_pt_mult})",
        f"{ind}# No free pass on the entry bar: arm SL/PT the instant we're filled.",
        f"{ind}# Filled at entry_price mid-bar, so no open-based gap fill here.",
        f"{ind}if L[i] <= sl_price:                      # SL priority (market, pays spread)",
        f"{ind}    close_trade(i, sl_price - SPREAD, 'SL'); in_position = False",
        f"{ind}elif H[i] > pt_price + SPREAD:            # PT = limit: fills AT target, no spread",
        f"{ind}    close_trade(i, pt_price, 'PT'); in_position = False",
    ])


# =====================================================================
# the generated-file template (fixed engine skeleton + tokens)
# =====================================================================
def generate_source(g: StrategyGenome) -> str:
    c = core
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S") if False else "(generated)"
    header = f'''#!/usr/bin/env python3
"""
STANDALONE backtest — ONE strategy, self-contained. {stamp}
Auto-generated by "Export Strategy.py" from ga_core (the single source of truth).
The logic below reproduces the ga_core engine BAR-FOR-BAR (verified on export):
tick snapping, entry spread, ATR, exit ordering, entry-bar SL/PT, no same-bar
re-entry, and the session-last-bar flatten (no overnight-gap teleport).

STRATEGY LOGIC:
{describe_strategy(g)}

Edit the INSTRUMENT CONFIG block for your instrument. Leave the STRATEGY block
and the engine alone unless you deliberately want to diverge from the optimizer.
"""
import csv
import os
import numpy as np

# ===== INSTRUMENT CONFIG (edit per instrument) =====
TICK               = {c.TICK_SIZE}       # tick size (ES/MES 0.25, stock 0.01, ...)
EXIT_SPREAD_TICKS  = {c.EXIT_SPREAD_TICKS}          # ticks of spread paid on every exit (fills at bid)
ENTRY_SPREAD_TICKS = {c.ENTRY_SPREAD_TICKS}          # ticks paid lifting the ask on a breakout entry
POINT_VALUE        = {c.POINT_VALUE}     # $ per 1.0 point move
TRADE_SIZE         = {c.POSITION_SIZE}         # contracts / shares
COMMISSION_PER_SIDE= {c.COMMISSION_PER_SIDE}       # $ per contract per side
SESSION_START      = "{c.SESSION_START}"
SESSION_END        = "{c.SESSION_END}"
DATA_FILEPATH      = r"{DATA_FILEPATH}"
OUTPUT_DIR         = os.path.dirname(os.path.abspath(__file__))
OUT_PREFIX         = "Standalone"
# ===================================================

# ===== STRATEGY (baked from the GA genome — do not edit) =====
ATR_PERIOD          = {g.atr_period}
REENTRY_COOLDOWN_BARS = {c.REENTRY_COOLDOWN_BARS}    # earliest re-entry = exit bar + this
WARMUP              = {c.WARMUP}
# ============================================================

SPREAD       = EXIT_SPREAD_TICKS * TICK
ENTRY_SPREAD = ENTRY_SPREAD_TICKS * TICK


def round_tick(v):
    return round(round(v / TICK) * TICK, 10)


def _parse_secs(ts):
    ts = str(ts).strip().split(".")[0]
    p = ts.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + (int(p[2]) if len(p) > 2 else 0)


def load_bars(fp):
    dates, times = [], []
    O, H, L, C = [], [], [], []
    with open(fp, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r = {{k.strip(): (v.strip() if v is not None else "") for k, v in r.items()}}
            dates.append(r["Date"]); times.append(r["Time"])
            O.append(float(r["Open"])); H.append(float(r["High"]))
            L.append(float(r["Low"])); C.append(float(r.get("Last", r.get("Close", "0"))))
    O = np.array([round_tick(x) for x in O]); H = np.array([round_tick(x) for x in H])
    L = np.array([round_tick(x) for x in L]); C = np.array([round_tick(x) for x in C])
    return dates, times, O, H, L, C


def compute_atr(H, L, C, period):
    n = len(C)
    tr = np.zeros(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i-1]), abs(L[i] - C[i-1]))
    atr = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr[i] = tr[i - period + 1:i + 1].mean()
    return atr


def session_last_bar_flags(dates, times, start_sec, end_sec):
    """True on the last bar of each trading session (so we can force-flatten
    there). Intraday session (start<=end) groups by calendar date; an overnight
    session (start>end) rolls bars at/after start into the next day."""
    n = len(dates)
    keys = []
    for d, t in zip(dates, times):
        # normalise the date to an ordinal so day arithmetic is trivial
        ds = d.replace("-", "/").split("/")
        if len(ds[0]) == 4:
            y, mo, da = int(ds[0]), int(ds[1]), int(ds[2])
        else:
            mo, da, y = int(ds[0]), int(ds[1]), int(ds[2])
        import datetime as _d
        ordv = _d.date(y, mo, da).toordinal()
        if start_sec > end_sec and _parse_secs(t) >= start_sec:
            ordv += 1
        keys.append(ordv)
    flags = np.zeros(n, dtype=bool)
    if n:
        flags[-1] = True
        for i in range(n - 1):
            flags[i] = keys[i + 1] != keys[i]
    return flags


def run_backtest(dates, times, O, H, L, C):
    n = len(C)
    atr = compute_atr(H, L, C, ATR_PERIOD)
    time_secs = np.array([_parse_secs(t) for t in times])
    start_sec = _parse_secs(SESSION_START)
    end_sec = _parse_secs(SESSION_END)
    is_session_last_bar = session_last_bar_flags(dates, times, start_sec, end_sec)

    in_position = False
    entry_price = 0.0
    entry_idx = 0
    sl_price = pt_price = 0.0
    trail_stop = -1e18
    pending_exit = False
    pending_reason = ""
    last_exit_idx = -10**9
    trades = []

    def close_trade(exit_i, exec_exit, reason):
        nonlocal last_exit_idx
        last_exit_idx = exit_i
        exec_exit = round_tick(exec_exit)
        gross = (exec_exit - entry_price) * POINT_VALUE * TRADE_SIZE
        net = gross - 2 * COMMISSION_PER_SIDE * TRADE_SIZE
        trades.append({{
            "entry_date": dates[entry_idx], "entry_time": times[entry_idx],
            "exit_date": dates[exit_i], "exit_time": times[exit_i],
            "entry_price": entry_price, "exit_price": exec_exit,
            "points": round(exec_exit - entry_price, 10),
            "pnl": net, "bars_held": exit_i - entry_idx, "reason": reason,
        }})

    for i in range(WARMUP, n):
        # ---------------- POSITION MANAGEMENT ----------------
        if in_position:
            if pending_exit:
                close_trade(i, O[i] - SPREAD, pending_reason)
                in_position = False
                pending_exit = False
            else:
                exited = False

                # (1) protective intrabar exit (bars AFTER entry)
                if i > entry_idx:
%INTRABAR%

                # (2) end-of-day / session flatten — never hold overnight
                if not exited and (time_secs[i] >= end_sec or is_session_last_bar[i]):
                    close_trade(i, C[i] - SPREAD, "EOD"); exited = True

                # (3) signal exits — fill at NEXT open
                if not exited:
%SIGNAL%

                if exited:
                    in_position = False
                if in_position:
                    continue

        # ---------------- ENTRY ----------------
        if not in_position:
            if i - last_exit_idx < REENTRY_COOLDOWN_BARS:
                continue
            if is_session_last_bar[i]:
                continue
            if not (start_sec <= time_secs[i] < end_sec):
                continue
            if i < 1:
                continue

            triggered = False
            entry_price = 0.0
%ENTRY%

            if triggered:
                in_position = True
                entry_idx = i
                trail_stop = -1e18
                pending_exit = False
%ENTRYBAR%

    if in_position:
        close_trade(n - 1, C[n - 1] - SPREAD, "Data_End")
    return trades
'''

    metrics_main = '''

# =====================================================================
# metrics + output (matches ga_core's compute_performance_metrics)
# =====================================================================
def summarize(trades):
    if not trades:
        return dict(net=0.0, n=0, wr=0.0, pf=0.0, avg=0.0, mdd=0.0, sharpe=0.0, sortino=0.0)
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    n = len(pnl); net = float(pnl.sum())
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    wr = len(wins) / n
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    avg = net / n
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq)
    mdd = float(np.max(peak - eq)) if n else 0.0
    # daily-resampled Sharpe / Sortino
    import datetime as _d
    def _od(ds):
        ds = ds.replace("-", "/").split("/")
        return (int(ds[0]), int(ds[1]), int(ds[2])) if len(ds[0]) == 4 else (int(ds[2]), int(ds[0]), int(ds[1]))
    exit_ord = []
    for t in trades:
        y, mo, da = _od(t["exit_date"]); exit_ord.append(_d.date(y, mo, da).toordinal())
    cum = np.cumsum(pnl)
    by_day = {}
    for o, cval in zip(exit_ord, cum):
        by_day[o] = cval  # last cum value that day
    days = sorted(by_day)
    if days:
        full = list(range(days[0], days[-1] + 1))
        eqd, last = [], 0.0
        for o in full:
            if o in by_day: last = by_day[o]
            eqd.append(last)
        eqd = np.array(eqd)
        dpnl = np.diff(eqd, prepend=0.0); dpnl[0] = eqd[0]
        sd, md = float(dpnl.std(ddof=1)), float(dpnl.mean())   # ddof=1 -> match pandas .std()
        sharpe = (md / sd) * np.sqrt(252) if sd > 0 else 0.0
        neg = dpnl[dpnl < 0]
        dstd = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
        sortino = (md / dstd) * np.sqrt(252) if dstd > 0 else 0.0
    else:
        sharpe = sortino = 0.0
    return dict(net=net, n=n, wr=wr, pf=pf, avg=avg, mdd=mdd, sharpe=sharpe, sortino=sortino)


def yearly(trades):
    from collections import defaultdict
    buckets = defaultdict(list)
    for t in trades:
        ds = t["exit_date"].replace("-", "/").split("/")
        yr = int(ds[0]) if len(ds[0]) == 4 else int(ds[2])
        buckets[yr].append(t)
    return [(yr, summarize(buckets[yr])) for yr in sorted(buckets)]


if __name__ == "__main__":
    print("=" * 74)
    print(" STANDALONE STRATEGY BACKTEST")
    print("=" * 74)
    print(__doc__.split("STRATEGY LOGIC:")[1].split('"""')[0].rstrip())
    print("-" * 74)
    dates, times, O, H, L, C = load_bars(DATA_FILEPATH)
    print(f"Loaded {len(C)} bars: {dates[0]} {times[0]}  ->  {dates[-1]} {times[-1]}")
    print(f"Instrument: TICK={TICK}  POINT_VALUE=${POINT_VALUE}/pt  SIZE={TRADE_SIZE}  COMM/side=${COMMISSION_PER_SIDE}\\n")

    trades = run_backtest(dates, times, O, H, L, C)
    m = summarize(trades)
    pf = "Inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    print("=" * 74)
    print(" OVERALL RESULT")
    print("=" * 74)
    print(f"  Net P&L      : ${m['net']:,.2f}")
    print(f"  Trades       : {m['n']}")
    print(f"  Win rate     : {m['wr']*100:.1f}%")
    print(f"  Profit factor: {pf}")
    print(f"  Sharpe       : {m['sharpe']:.2f}")
    print(f"  Sortino      : {m['sortino']:.2f}")
    print(f"  Avg $/trade  : ${m['avg']:,.2f}")
    print(f"  Max drawdown : ${m['mdd']:,.2f}")

    print("\\n" + "=" * 74)
    print(" YEARLY BREAKDOWN")
    print("=" * 74)
    print(f"  {'Year':<6} | {'Trades':>7} | {'Win %':>6} | {'Sharpe':>7} | {'Avg $/Trade':>12} | {'Net P&L $':>13}")
    print("  " + "-" * 66)
    for yr, ym in yearly(trades):
        print(f"  {yr:<6} | {ym['n']:>7} | {ym['wr']*100:>6.1f} | {ym['sharpe']:>7.2f} | {ym['avg']:>12,.2f} | {ym['net']:>13,.2f}")

    # ---- CSVs ----
    if trades:
        cols = ["entry_date", "entry_time", "exit_date", "exit_time",
                "entry_price", "exit_price", "reason", "points", "pnl", "bars_held"]
        tp = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_trades.csv")
        with open(tp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for t in trades:
                row = {k: t[k] for k in cols}; row["pnl"] = round(row["pnl"], 2)
                w.writerow(row)
        print(f"\\nTrade log    -> {tp}")
        yp = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_yearly.csv")
        with open(yp, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["year", "trades", "win_rate_%", "sharpe", "avg_trade_$", "net_pnl_$"])
            for yr, ym in yearly(trades):
                w.writerow([yr, ym["n"], round(ym["wr"]*100, 1), round(ym["sharpe"], 2), round(ym["avg"], 2), round(ym["net"], 2)])
        print(f"Yearly       -> {yp}")
    print("\\nDone.")
'''

    body = (header
            .replace("%INTRABAR%", build_intrabar_exit(g))
            .replace("%SIGNAL%", build_signal_exit(g))
            .replace("%ENTRY%", build_entry_block(g))
            .replace("%ENTRYBAR%", build_entrybar_atr(g)))
    return body + metrics_main


# =====================================================================
# resolve, generate, verify
# =====================================================================
def resolve_genome() -> StrategyGenome:
    if PINNED_GENOME is not None:
        print("Strategy source: PINNED genome.")
        return dict_to_genome(PINNED_GENOME)
    print(f"Strategy source: hall of fame rank #{PIN_FROM_HOF_RANK}.")
    top = load_hall_of_fame(core.HOF_JSON)
    match = [g for rank, _, g in top if rank == PIN_FROM_HOF_RANK]
    if not match:
        raise SystemExit(f"rank #{PIN_FROM_HOF_RANK} not found.")
    return match[0]


if __name__ == "__main__":
    genome = resolve_genome()
    print("-" * 74)
    print(describe_strategy(genome))
    print("-" * 74)

    src = generate_source(genome)
    out_path = os.path.join(OUTPUT_DIR, OUT_PY_NAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"\nGenerated standalone file -> {out_path}")

    # ---- VERIFY: generated file must reproduce ga_core bar-for-bar ----
    print("\nVerifying the generated file matches the ga_core engine on your data...")
    try:
        import importlib.util
        df = core.load_and_preprocess_data(DATA_FILEPATH)
        ref = core.run_backtest(genome, df)          # engine truth
        ref_tdf = ref["trade_df"]

        spec = importlib.util.spec_from_file_location("gen_standalone", out_path)
        gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
        dates, times, O, H, L, C = gen.load_bars(DATA_FILEPATH)
        gen_trades = gen.run_backtest(dates, times, O, H, L, C)

        ref_n = len(ref_tdf); gen_n = len(gen_trades)
        ref_net = float(ref_tdf["pnl"].sum()) if ref_n else 0.0
        gen_net = float(sum(t["pnl"] for t in gen_trades)) if gen_n else 0.0
        ok_count = (ref_n == gen_n)
        ok_net = abs(ref_net - gen_net) < 1e-6
        # per-trade entry/exit price + reason match
        ok_trades = ok_count
        if ok_count and ref_n:
            for k in range(ref_n):
                r = ref_tdf.iloc[k]; gt = gen_trades[k]
                if (abs(float(r["entry_price"]) - gt["entry_price"]) > 1e-9 or
                    abs(float(r["exit_price"]) - gt["exit_price"]) > 1e-9 or
                    str(r["reason"]) != gt["reason"]):
                    ok_trades = False
                    print(f"  MISMATCH at trade {k}: engine "
                          f"({r['entry_price']}->{r['exit_price']} {r['reason']}) vs "
                          f"generated ({gt['entry_price']}->{gt['exit_price']} {gt['reason']})")
                    break
        print(f"  engine   : {ref_n} trades, net ${ref_net:,.2f}")
        print(f"  generated: {gen_n} trades, net ${gen_net:,.2f}")
        if ok_count and ok_net and ok_trades:
            print("  RESULT: PASS — the standalone file reproduces the engine bar-for-bar.")
        else:
            print("  RESULT: FAIL — mismatch above. Do not ship this file; tell me and I'll fix the generator.")
    except Exception as e:
        print(f"  Verification could not run ({e}). The file was still written.")

    print("\nDone.")
