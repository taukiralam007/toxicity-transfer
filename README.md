# Measuring the detection limit of multitask transfer in chemical toxicity prediction

Code and results for the paper. Across 295 assays from eight public
collections, each single-task baseline is trained 20 times varying only the
random seed, giving a per-task noise floor. Single-source transfer is then
measured against that floor with both arms replicated across seeds.

## Pipeline

```bash
python transfer_law.py --download
python transfer_law.py --seeds 20 --max-pairs 25000   # noise floor, 25,000 pairs
python seed_null.py --pairs 200 --seeds 10            # paired replication
python multisource.py && python screening.py && python boundary.py
python gnn_null.py --stage floor --tasks 60 --seeds 10
python make_report.py                                 # regenerates RESULTS.md
```

`RESULTS.md` is generated from `results/*.csv`, so reported numbers cannot
drift from the data behind them.

## Figures

One script per figure, under `figcode/`. Each prints every number it draws.

| Script | Figure |
|---|---|
| `fig1_pipeline.py` | measurement pipeline |
| `fig2_corpus.py` | corpus and per-task noise floor |
| `fig3_calibration.py` | the screening rule is mis-calibrated |
| `fig4_seedbudget.py` | seeds required per arm |
| `fig5_direction.py` | geometry does not predict direction |
| `fig6_overlap.py` | pooled overlap effect is an artefact |
| `fig7_stacking.py` | stacking and screening |

```bash
python figcode/verify.py    # re-derives all 106 plotted numbers, non-zero on mismatch
python figcode/fig2_corpus.py
```

Requires `numpy pandas scipy matplotlib scikit-learn`.
