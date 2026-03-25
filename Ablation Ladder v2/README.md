# Ablation Ladder v2

This folder contains the current walk-forward experiment stage.

## Goal
To extend the v1 baseline with:
- stability-driven selection,
- improved reporting at unique run level,
- daily test export,
- exogenous regime diagnostics,
- controlled feature ladder comparisons.

## Expected structure
- `Experiments_Ablation_Ladder_v2.ipynb`
- `comparison_outputs/`
- `paper_figures/`

## Recommended files for `comparison_outputs/`
Run-level and summary:
- `walk_forward_results.csv`
- `unique_run_level_results.csv`
- `corrected_walk_forward_summary.csv`
- `selection_rule_comparison.csv`
- `selection_rule_summary.csv`
- `pairwise_permutation_tests_recomputed.csv`
- `validation_vs_test_winner_by_fold.csv`
- `artifact_index.json`

Daily and regime-aware artifacts:
- `walk_forward_daily_test_returns.csv`
- `regime_run_level_metrics.csv`
- `regime_summary_by_feature_set.csv`
- `regime_summary_by_fold.csv`

## Intended interpretation
This stage should answer a stricter question than v1:

> Which feature families and selection rules remain the most robust across walk-forward windows and regimes?

The comparison to v1 should focus on:
- run-level robustness,
- selection stability,
- fold-level consistency,
- regime-aware behavior.
