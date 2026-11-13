# Figure code

One script per figure. Each is standalone, prints every number it draws, and
writes PNG (600 dpi), PDF and SVG. Nothing is hard-coded except four constants
that are flagged in `style.py` and listed at the bottom of this file.

## Files

| Script | Figure | Inputs |
|---|---|---|
| `fig1_pipeline.py` | 1 — measurement pipeline | none (a drawing) |
| `fig2_corpus.py` | 2 — corpus and per-task noise floor | `noise_floor.csv` |
| `fig3_calibration.py` | 3 — the screening rule is mis-calibrated | `noise_floor.csv`, `analysis_table.csv` |
| `fig4_seedbudget.py` | 4 — seeds required per arm | none (a power calculation) |
| `fig5_direction.py` | 5 — geometry does not predict direction | `analysis_table.csv` |
| `fig6_overlap.py` | 6 — overlap effect is an artefact of pooling | `analysis_table.csv`, `boundary_by_overlap.csv` |
| `fig7_stacking.py` | 7 — stacking and screening | `multisource.csv`, `screening.csv` |
| `style.py` | shared palette, type, loaders, constants | — |
| `verify.py` | 106 assertions over every plotted number | all CSVs |

## Running

```bash
export RESULTS=../results        # or pass the path as argv[1]

python verify.py                 # exit 0 means every number checks out
python fig2_corpus.py            # ... one per figure
python fig3_calibration.py
python fig4_seedbudget.py
python fig5_direction.py
python fig6_overlap.py
python fig7_stacking.py
python fig1_pipeline.py
```

Each prints a `FIGURE n - values plotted` block. Read it against the figure.

Requires `numpy pandas scipy matplotlib scikit-learn`, and Liberation Sans or
Arial on the system (otherwise matplotlib substitutes and metrics shift).

## Output specification

- 600 dpi PNG; PDF and SVG with live text (Type 42 embedded, `svg.fonttype: none`)
- drawn at 7.0 in wide, placed at 6.5 in — type lands at ≈7 pt at final size
- text `#111111`, axis rules `#333333`
- palette `#3D6FA8` `#E08A2E` `#4E9A51` `#C14840` `#7E62A3` `#3C9CA6` `#946044` `#9A9A9A`
- colour never carries meaning alone — every bar is labelled with its value

## Conventions that matter for review

- **Figure 2** collections are ordered by median noise floor, ascending, in all
  three panels, so a and b/c read in the same direction.
- **Figure 5b** re-fits the direction model when the script runs; the printed
  numbers come from that fit, not from a stored value. Fold shuffling is seeded
  (`random_state=0`), so it reproduces.
- **Figure 6a** omits the 0.70–0.90 stratum, which holds 8 cleared pairs; the
  threshold is `n_cleared >= 20` and the caption states the omission.
- **Figure 6c** runs the Wilcoxon on fractions, not percentages. Scaling by 100
  perturbs exact ties and changes the p-value (0.73 → 0.67). Do not "simplify"
  this by plotting percentages and testing the same array.
- **Figure 6** bins are half-open `[a, b)`; 8 pairs sit exactly at 0.02.

## The four seed-null quantities (Figure 3b, 3c)

These are **derived, not transcribed** — `style.py` reads `results/seed_null.csv`
and computes them with the same four expressions `seed_null.py` prints:

| Quantity | Expression | Value |
|---|---|---|
| median sd(ΔAUPRC) | `df.sd_delta_ap.median()` | 0.00611 |
| median single-task σ | `df.base_sd.median()` | 0.00456 |
| % clearing 2σ seed-averaged | `100 * df.cleared_2s.mean()` | 1.00 |
| % surviving BH q<0.10 | `100 * (df.q_pair < 0.10).mean()` | 0.00 |

Two further quantities derive from these and are asserted in `verify.py`:
the 1.34 ratio (0.00611 / 0.00456 = 1.3399) and the 14% chance line (the
two-sided normal tail beyond 2 / 1.34 = 1.493 σ, which is 13.6%).

### If `seed_null.csv` is missing

`style.py` falls back to the values recorded from the 200-pair × 10-seed run
and prints a warning to stderr — it never substitutes silently. `verify.py`
ends with `INCOMPLETE` and names the four quantities it could not check.

To close it, on the machine that ran `seed_null.py`:

```bash
cp results/seed_null.csv <this results dir>/
python verify.py            # now ends with "no transcribed numbers remain"
python fig3_calibration.py  # 3b and 3c redrawn from the file
```

Nothing else in the figure set depends on a number that is typed rather than
computed.
