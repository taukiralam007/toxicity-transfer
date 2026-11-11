# Transfer in multitask chemical toxicity modelling

Generated 2026-11-11 by `make_report.py`. Every number is read from the result files; do not edit by hand.

## Summary

- **25,000** source-target task pairs measured
- **1,930 (7.7%)** changed target performance beyond a 20-seed noise floor
- Of those, **1,036 improved** and **894 degraded** (54% positive)
- No dataset-geometry descriptor predicts the direction
- Single-task baseline reproduces the published MTForestNet figure: zebrafish mean AUROC **0.633** vs 0.638 reported, under a *harder* (global scaffold) split

## Corpus

295 modelled tasks across 8 collections.

| collection | tasks | mean_auroc | mean_auprc | mean_noise_2sigma |
|---|---|---|---|---|
| chembl_sparse | 180 | 0.8648 | 0.8927 | 0.0079 |
| muv | 17 | 0.5537 | 0.1167 | 0.0238 |
| pcba | 38 | 0.626 | 0.0745 | 0.0314 |
| tox21 | 12 | 0.7665 | 0.4009 | 0.0311 |
| zf_embryo | 18 | 0.6425 | 0.2142 | 0.0341 |
| zf_larva | 24 | 0.6322 | 0.2773 | 0.0379 |
| zf_lipid | 2 | 0.6324 | 0.3609 | 0.088 |
| zf_morph | 4 | 0.5935 | 0.5979 | 0.0328 |

MUV sits near chance by construction (built so similarity-based methods fail); PCBA's low AUPRC reflects extreme class imbalance. Neither is anomalous.

## Method

- **Standardisation**: one RDKit pipeline for every collection.
- **Splitting**: a single Bemis-Murcko scaffold split over the *union* of all compounds, so a scaffold in one task's training set never appears in another task's test set.
- **Noise floor**: 20 seeds per task; an effect counts only if |ΔAUPRC| exceeds 2σ of that task's seed-to-seed spread. Fixed before any pair was examined.
- **Source features**: out-of-fold for the source's own training compounds.
- **Validation**: grouped by target task. Ungrouped folds inflate balanced accuracy from 0.56 to 0.87 through target memorisation.

## Does dataset geometry predict transfer?

| descriptor | rho | p_value | n |
|---|---|---|---|
| effective overlap (Tanimoto >= 0.35) | -0.004551 | 0.4718 | 25000 |
| exact chemical overlap | -0.001015 | 0.8725 | 25000 |
| bridging-set concordance (kappa) | 0.00316 | 0.7334 | 11622 |
| mean nearest-neighbour similarity | -0.01016 | 0.1081 | 25000 |
| prevalence shift | 0.004385 | 0.4882 | 25000 |
| target class imbalance | 0.06633 | 8.795e-26 | 25000 |

All correlations are indistinguishable from zero except `target_imbalance`, which is statistically detectable at n = 25,000 but corresponds to a difference of ~0.001 AUPRC - **practically negligible**.

## Is there an overlap threshold above which transfer becomes predictable?

| overlap | n_pairs | n_cleared | pct_cleared | pct_positive | mean_delta | cv_bal_acc |
|---|---|---|---|---|---|---|
| 0.00-0.02 | 15635 | 1237 | 7.912 | 49.39 | -0.0001 | 0.4665 |
| 0.02-0.05 | 3705 | 290 | 7.827 | 61.03 | -0.0006 | 0.5456 |
| 0.05-0.10 | 2447 | 183 | 7.479 | 60.11 | 0.0004 | 0.4973 |
| 0.10-0.20 | 1029 | 61 | 5.928 | 54.1 | 0.0012 | 0.5252 |
| 0.20-0.35 | 328 | 26 | 7.927 | 57.69 | -0 |  |
| 0.35-0.50 | 779 | 61 | 7.831 | 75.41 | -0.0018 | 0.5375 |
| 0.50-0.70 | 306 | 29 | 9.477 | 62.07 | 0.0016 |  |
| 0.70-0.90 | 140 | 8 | 5.714 | 62.5 | -0 |  |
| 0.90-1.00 | 631 | 35 | 5.547 | 60 | 0.0007 |  |

Best stratum reaches **0.546** balanced accuracy (target-grouped 5-fold CV; 0.50 is chance). No stratum is predictable.

A 2026 analysis of gradient-based task affinity placed the usable floor near 30-40% sample overlap, derived largely from synthetic controls, and noted that real bioactivity data falls below it. On real toxicology data, predictability does not appear even in the strata above that threshold.

## Does transfer appear when many sources are stacked?

| k | n | median_abs_delta | mean_delta | pct_cleared |
|---|---|---|---|---|
| 1 | 36 | 0.005 | -0.0017 | 11.11 |
| 5 | 36 | 0.0073 | -0.0031 | 0 |
| 20 | 36 | 0.0087 | -0.0084 | 16.67 |
| 50 | 36 | 0.0106 | -0.0025 | 30.56 |

Median |ΔAUPRC| rises from 0.0050 at k=1 to 0.0106 at k=50, while the mean goes from -0.0017 to -0.0025 and is negative at every k. Adding sources widens the spread of target performance without improving its centre: the additional pairs crossing the noise floor do so in both directions. Transfer does not accumulate into benefit.

## If selection cannot be predicted, can it be screened?

| arm | mean AUPRC | gain over single |
|---|---|---|
| target alone | 0.3777 | 0 |
| all sources | 0.3733 | -0.0044 |
| screened sources | 0.3737 | -0.004 |
| random k sources | 0.3789 | 0.0012 |

n = 23 targets, 40 candidate sources each; median 23 kept by the screen.

Screening does not beat the baselines. Reported as a further negative result.

## Limitations

- **One model family.** All results use random forests on 1024-bit ECFP with stacked prediction features. A graph neural network check is required before these conclusions can be stated as properties of the data rather than of the estimator.
- **One transfer mechanism.** Stacking source predictions is MTForestNet's mechanism; shared-representation architectures may behave differently.
- **Endpoint semantics untested at scale.** Only 60 of 295 tasks have interpretable names; the rest are accession identifiers.

## Reproducing

```bash
python transfer_law.py --download
python transfer_law.py --seeds 20 --max-pairs 25000
python multisource.py --targets 40
python screening.py --targets 30 --candidates 40
python boundary.py
python figures.py
python make_report.py
```
