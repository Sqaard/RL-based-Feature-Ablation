# Ablation Ladder v1

This folder contains the previous walk-forward / Horizon A baseline.

## Goal
To move beyond a single validation/test split and evaluate feature families under a more robust walk-forward setup.

## What this experiment established
This legacy baseline led to an important shift in the project:

- macro / exogenous features were more robust than HMM/GRU-style learned features,
- HMM and GRU are therefore currently treated as negative controls,
- the project focus moved from “more latent complexity” to “better OOS robustness and methodology”.

## Legacy limitations
This historical run did not include a daily test export, so:
- run-level results can be rebuilt correctly,
- selection reports can be rebuilt,
- but the new exogenous regime diagnostics cannot be fully reconstructed for this run.

## Recommended files for `comparison_outputs/`
- `walk_forward_results.csv`
- `unique_run_level_results.csv`
- `corrected_walk_forward_summary.csv`
- `selection_rule_comparison.csv`
- `selection_rule_summary.csv`
- `pairwise_permutation_tests_recomputed.csv`
- `validation_vs_test_winner_by_fold.csv`
- `artifact_index.json`
- `legacy_walk_forward_regime_breakdown.csv`

## Interpretation
Use this experiment as the **legacy baseline** for future README comparisons and for judging whether newer feature sets or selection rules actually improve OOS robustness.
