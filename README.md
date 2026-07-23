# Strategy GA Generator + OOS Tester

Two scripts that share one engine:

- **PyGeneticAlgo.py** — evolves strategies on the in-sample years (2024–2025 by default), keeps a top-5 hall of fame, saves them to `hall_of_fame_results.json`.
- **OOS Tester.py** — loads that JSON, re-runs each strategy on the full timeline through the *same* engine, and breaks the equity curve into the in-sample window and your out-of-sample windows (2019–2023 historical, 2026 forward).
- **ga_core.py** — the single source of truth: genome, backtest engine, metrics, data loader, JSON handoff. **Both scripts import everything strategy-related from here.**

## Why the rewrite

Your two files had drifted into two different strategies, which is why every Gemini revision broke something:

1. **Filename mismatch** — generator wrote `best_5_strategies.json`, tester read `hall_of_fame_results.json` → always "not found".
2. **Genome mismatch** — generator's genome had ATR fields and `"exceed"/"dip"`; tester's had `entry_offset_ticks` and `"breakout"/"dip"` with no ATR. `StrategyGenome(**data)` crashed on load.
3. **Different engines** — generator exited on ATR stop/target, tester had no stops at all, and they used different session times (08:31 vs 08:45). Results could never match.

With one shared `ga_core.py`, the genome and engine exist in exactly one place, so the optimizer and validator are guaranteed to run the identical simulation. They can't desync again.

## The genome (all evolvable)

- **Entry filter** (optional): a condition that may look back over completed bars i-1..i-3, e.g. `C[i-1] > L[i-2]`.
- **Entry trigger** — off the most recent completed bar `[i-1]`:
  - breakout → `High[i] > (High or Close)[i-1] + entry_offset_ticks` → fill pays the spread.
  - dip → `Low[i] < (Low or Close)[i-1] - entry_offset_ticks` → limit fill, no slippage.
- **Protective exit** (`exit_style` gene):
  - `ticks` → **trailing** stop off the previous bar: `(Close or Low)[i-1] - exit_trigger_ticks` (ratchets up, arms from entry bar +1). `exit_ref_close` picks Close vs Low.
  - `atr` → ATR stop-loss / profit-target.
- **Structural exits** (both styles): `max_bars_hold` and the optional exit condition both fill at **Open[i+1] − spread**; end-of-day fills at that bar's close.

## Exit fills

Every exit pays a flat **1-tick spread** (`EXIT_SPREAD_TICKS`), i.e. fills at the bid. No LE / limit model — kept simple per the strategy spec. Entry: breakout pays the spread, dip is a no-slippage limit.

## Outputs (saved to the script folder)

- `hall_of_fame_results.json` / `best_strategy.json` — the evolved strategies.
- `GA_best_equity.png` — the optimizer's in-sample best equity.
- `OOS_Rank{N}_equity.png` — per-strategy IS/OOS equity charts.
- `OOS_Rank{N}_trades.csv` — full trade list per strategy.
- `strategy_logic.txt` — human-readable entry/exit logic for every strategy.

## Run

```bash
python PyGeneticAlgo.py      # writes hall_of_fame_results.json
python "OOS Tester.py"       # reads it, prints IS-vs-OOS table + charts
```

Point `DATA_FILEPATH` at your data file in each script's `__main__`. All three files must sit in the same folder (the scripts `import ga_core`).

## Reading it

Look at the per-window numbers on each chart. If in-sample looks great but the out-of-sample windows fall apart, it's curve-fit. If the **per-trade** edge is stable across windows (as it was on the SVXY test: ~$13.5/trade IS, ~$13.8 historical OOS, ~$10.2 forward OOS), that's a strategy that generalised — subject to the usual reality check that the fills/slippage assumptions are honest.
