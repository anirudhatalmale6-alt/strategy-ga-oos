#!/usr/bin/env python3
"""
PyGeneticAlgo.py  —  GA strategy optimizer (in-sample).
=======================================================
Evolves strategies on the IN-SAMPLE year window (ga_core.IS_START_YEAR..IS_END_YEAR),
keeps a top-N hall of fame, and writes them to ga_core.HOF_JSON for the OOS tester.

All strategy logic (genome, engine, metrics, JSON) lives in ga_core.py so this
file and "OOS Tester.py" can never drift apart again.
"""
import os
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
plt.rcParams["text.parse_math"] = False   # treat '$' in titles/labels as literal, not LaTeX math

import ga_core as core

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
from ga_core import (StrategyGenome, create_random_genome, random_condition,
                     run_backtest, load_and_preprocess_data, save_hall_of_fame)

plt.ion()  # live plotting in Spyder


# ---------------- genetic operators ----------------
def crossover(a: StrategyGenome, b: StrategyGenome) -> Tuple[StrategyGenome, StrategyGenome]:
    c1 = StrategyGenome(**a.__dict__)
    c2 = StrategyGenome(**b.__dict__)
    # swap a spread of genes between the parents
    for attr in ("exit_style", "exit_trigger_ticks", "exit_offset_ticks",
                 "atr_period", "atr_sl_mult", "atr_pt_mult",
                 "entry_offset_ticks", "exit_cond"):
        setattr(c1, attr, getattr(b, attr))
        setattr(c2, attr, getattr(a, attr))
    return c1, c2


def mutate(g: StrategyGenome, rate: float = 0.2) -> StrategyGenome:
    if random.random() < rate:
        g.max_bars_hold = random.randint(1, 10)
    if random.random() < rate:
        g.entry_offset_ticks = random.randint(0, 6)
    if random.random() < rate:
        g.entry_trigger_type = random.choice(["breakout", "dip"])
    if random.random() < rate:
        g.entry_ref_close = random.choice([True, False])
    if random.random() < rate:
        g.exit_style = random.choice(["ticks", "atr"])
    if random.random() < rate:
        g.exit_ref_close = random.choice([True, False])
    if random.random() < rate:
        g.exit_trigger_ticks = max(0, min(12, g.exit_trigger_ticks + random.randint(-2, 2)))
    if random.random() < rate:
        g.exit_offset_ticks = max(0, min(6, g.exit_offset_ticks + random.randint(-2, 2)))
    if random.random() < rate:
        g.atr_sl_mult = round(max(0.5, min(2.5, g.atr_sl_mult + random.uniform(-0.3, 0.3))), 2)
    if random.random() < rate:
        g.atr_pt_mult = round(max(0.5, min(2.5, g.atr_pt_mult + random.uniform(-0.3, 0.3))), 2)
    if random.random() < rate:
        g.entry_cond = random_condition()
    if random.random() < rate:
        g.exit_cond = random_condition()
    return g


def update_hall_of_fame(hof, current, top_n=5):
    combined = hof + current
    combined.sort(key=lambda x: x[1]["fitness"], reverse=True)
    seen, unique = set(), []
    for genome, stats in combined:
        key = str(core.genome_to_dict(genome))
        if key not in seen:
            seen.add(key)
            unique.append((genome, stats))
        if len(unique) == top_n:
            break
    return unique


# ---------------- evolution loop ----------------
def run_ga_optimization(filepath, start_year=core.IS_START_YEAR, end_year=core.IS_END_YEAR,
                        pop_size=20, generations=15, top_n_save=5):
    print(f"Loading in-sample data {filepath} [{start_year}-{end_year}] ...")
    df = load_and_preprocess_data(filepath, start_year=start_year, end_year=end_year)
    print(f"In-sample bars: {len(df)}")

    population = [create_random_genome() for _ in range(pop_size)]
    hall_of_fame: List[Tuple[StrategyGenome, Dict[str, Any]]] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.show(block=False)

    for gen in range(generations):
        evals = [(g, run_backtest(g, df)) for g in population]
        hall_of_fame = update_hall_of_fame(hall_of_fame, evals, top_n=top_n_save)

        gbest = max(evals, key=lambda e: e[1]["fitness"])[1]
        obest = hall_of_fame[0][1]
        print(f"Gen {gen+1:02d}/{generations:02d} | "
              f"Gen best net ${gbest['net_profit']:.2f} | "
              f"All-time net ${obest['net_profit']:.2f} | "
              f"Trades {obest['trade_count']} | MaxDD ${obest['max_dd']:.2f}")

        best_tdf = obest["trade_df"]
        if not best_tdf.empty:
            ax.clear()
            ax.plot(best_tdf["exit_dt"], obest["equity_curve"],
                    color="#1f77b4", lw=1.5,
                    label=f"Rank 1 net ${obest['net_profit']:.2f}")
            ax.axhline(0, color="black", ls="--", alpha=0.5)
            ax.set_title(f"GA Gen {gen+1}/{generations} | best fitness {obest['fitness']:.2f}")
            ax.set_xlabel("Date"); ax.set_ylabel("Net Profit ($)")
            ax.grid(True, ls=":", alpha=0.6); ax.legend(loc="upper left")
            fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.1)

        # elitism + reproduction
        evals.sort(key=lambda e: e[1]["fitness"], reverse=True)
        survivors = [e[0] for e in evals[: max(2, pop_size // 2)]]
        nxt = survivors.copy()
        while len(nxt) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            c1, c2 = crossover(p1, p2)
            nxt.append(mutate(c1))
            if len(nxt) < pop_size:
                nxt.append(mutate(c2))
        population = nxt

    plt.ioff()
    out_png = os.path.join(OUTPUT_DIR, "GA_best_equity.png")
    fig.savefig(out_png, dpi=130)
    print(f"Saved in-sample best equity chart -> {out_png}")
    return hall_of_fame


if __name__ == "__main__":
    DATA_FILEPATH = "C:/Users/Administrator/Desktop/Stock Data/SVXY P&F 3-3 2p.txt"

    hof = run_ga_optimization(DATA_FILEPATH, pop_size=20, generations=15, top_n_save=5)

    print("\n" + "=" * 60)
    print(f"OPTIMIZATION COMPLETE — TOP {len(hof)} STRATEGIES")
    print("=" * 60)
    for rank, (genome, stats) in enumerate(hof, start=1):
        print(f"\n--- RANK {rank} ---")
        print(f"  Net Profit : ${stats['net_profit']:.2f}")
        print(f"  Trades     : {stats['trade_count']}")
        print(f"  Win Rate   : {stats['win_rate']*100:.1f}%")
        print(f"  Max DD     : ${stats['max_dd']:.2f}")
        print(f"  Fitness    : {stats['fitness']:.2f}")
        print(f"  Exit style : {genome.exit_style}  "
              f"(trig {genome.exit_trigger_ticks}t / off {genome.exit_offset_ticks}t "
              f"| atr {genome.atr_period} sl{genome.atr_sl_mult} pt{genome.atr_pt_mult})")

    save_hall_of_fame(hof)   # -> hall_of_fame_results.json (read by the OOS tester)
    plt.show()
