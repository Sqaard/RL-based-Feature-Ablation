# Ablation Ladder v1 Figure Index

- Comparison inputs: `Ablation Ladder v1\comparison_outputs`
- Figure outputs: `Ablation Ladder v1\paper_figures`
- Plotting tables: `Ablation Ladder v1\paper_figures\plotting_tables`

## Generated Figures

### `01_v1_v2_feature_set_test_sharpe_comparison` - Ablation Ladder v1 vs. v2: test Sharpe by feature set
Cross-version comparison of median test Sharpe with uncertainty bars derived from the run-level distributions.
- Files: `compare_with_v2\01_v1_v2_feature_set_test_sharpe_comparison.png`, `compare_with_v2\01_v1_v2_feature_set_test_sharpe_comparison.svg`
- Note: Primary copy is stored in the v2 paper_figures folder because v2 is the current main experiment.

### `02_v1_test_sharpe_distribution_by_feature_set` - Ablation Ladder v1: test Sharpe distribution by feature set
Run-level distribution of test Sharpe across folds and seeds for the v1 baseline.
- Files: `02_v1_test_sharpe_distribution_by_feature_set.png`, `02_v1_test_sharpe_distribution_by_feature_set.svg`
- Tables: `plotting_tables\02_v1_test_sharpe_distribution_by_feature_set.csv`

### `04_v1_validation_vs_test_scatter` - Ablation Ladder v1: validation-to-test transfer
Run-level validation Sharpe versus test Sharpe for the checkpoint-selected runs in the v1 baseline.
- Files: `04_v1_validation_vs_test_scatter.png`, `04_v1_validation_vs_test_scatter.svg`
- Tables: `plotting_tables\04_v1_validation_vs_test_scatter_points.csv`, `plotting_tables\04_v1_validation_vs_test_scatter_summary.csv`
- Note: Points reflect checkpoint-selected runs under `checkpoint_generalization_score`, not the post-hoc configuration-level rule comparison.

### `06_v1_generalization_gap_by_feature_set` - Ablation Ladder v1: generalization gap by feature set
Median validation-to-test Sharpe gap by feature family in v1.
- Files: `06_v1_generalization_gap_by_feature_set.png`, `06_v1_generalization_gap_by_feature_set.svg`
- Tables: `plotting_tables\06_v1_generalization_gap_by_feature_set.csv`, `plotting_tables\04_v1_validation_vs_test_scatter_summary.csv`
- Note: Gaps are computed on checkpoint-selected runs under `checkpoint_generalization_score`.

### `08_v1_fold_feature_heatmap_test_sharpe` - Ablation Ladder v1: fold-by-feature test Sharpe heatmap
Fold-level median test Sharpe in the v1 walk-forward baseline.
- Files: `08_v1_fold_feature_heatmap_test_sharpe.png`, `08_v1_fold_feature_heatmap_test_sharpe.svg`
- Tables: `plotting_tables\08_v1_fold_feature_heatmap_test_sharpe.csv`
- Note: Cell values are fold-level medians across seeds for runs selected with `checkpoint_generalization_score`.

### `10_v1_selection_rule_comparison` - Ablation Ladder v1: post-hoc configuration-level selection rule comparison
Post-hoc comparison of configuration-level Sharpe-only and robust rules rebuilt from fold-level feature-set summaries.
- Files: `10_v1_selection_rule_comparison.png`, `10_v1_selection_rule_comparison.svg`
- Tables: `plotting_tables\10_v1_selection_rule_comparison.csv`, `plotting_tables\10_v1_selection_rule_comparison_tidy.csv`
- Note: Post-hoc configuration-level comparison. These rows do not necessarily correspond to the rule used inside the original experiment run.

### `12_v1_fold_winner_map` - Ablation Ladder v1: post-hoc fold winner map
Fold-by-fold post-hoc map showing which feature set each rebuilt configuration rule would have selected and whether it matched the test winner.
- Files: `12_v1_fold_winner_map.png`, `12_v1_fold_winner_map.svg`
- Tables: `plotting_tables\12_v1_fold_winner_map.csv`
- Note: Post-hoc configuration-level map. It shows what each rebuilt rule would have selected per fold, not the literal rule used inside the original training run.

### `14_v1_pairwise_test_sharpe_matrix` - Ablation Ladder v1: pairwise test-Sharpe matrix
Permutation-test matrix of pairwise mean test-Sharpe differences for v1.
- Files: `14_v1_pairwise_test_sharpe_matrix.png`, `14_v1_pairwise_test_sharpe_matrix.svg`
- Tables: `plotting_tables\14_v1_pairwise_test_sharpe_matrix_long.csv`, `plotting_tables\14_v1_pairwise_test_sharpe_matrix_matrix.csv`
