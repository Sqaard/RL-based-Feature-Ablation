# Ablation Ladder v2 Figure Index

- Comparison inputs: `Ablation Ladder v2\comparison_outputs`
- Figure outputs: `Ablation Ladder v2\paper_figures`
- Plotting tables: `Ablation Ladder v2\paper_figures\plotting_tables`

## Generated Figures

### `01_v1_v2_feature_set_test_sharpe_comparison` - Ablation Ladder v1 vs. v2: test Sharpe by feature set
Cross-version comparison of median test Sharpe with uncertainty bars derived from the run-level distributions.
- Files: `01_v1_v2_feature_set_test_sharpe_comparison.png`, `01_v1_v2_feature_set_test_sharpe_comparison.svg`
- Tables: `plotting_tables\01_v1_v2_feature_set_test_sharpe_comparison.csv`
- Note: Copied to compare_with_v2 for v1-side reference.

### `03_v2_test_sharpe_distribution_by_feature_set` - Ablation Ladder v2: test Sharpe distribution by feature set
Run-level distribution of test Sharpe across folds and seeds for the current Horizon A experiment.
- Files: `03_v2_test_sharpe_distribution_by_feature_set.png`, `03_v2_test_sharpe_distribution_by_feature_set.svg`
- Tables: `plotting_tables\03_v2_test_sharpe_distribution_by_feature_set.csv`

### `05_v2_validation_vs_test_scatter` - Ablation Ladder v2: validation-to-test transfer
Run-level validation Sharpe versus test Sharpe for the checkpoint-selected runs in the current Horizon A experiment.
- Files: `05_v2_validation_vs_test_scatter.png`, `05_v2_validation_vs_test_scatter.svg`
- Tables: `plotting_tables\05_v2_validation_vs_test_scatter_points.csv`, `plotting_tables\05_v2_validation_vs_test_scatter_summary.csv`
- Note: Points reflect checkpoint-selected runs under `checkpoint_robust_score`, not the post-hoc configuration-level rule comparison.

### `07_v2_generalization_gap_by_feature_set` - Ablation Ladder v2: generalization gap by feature set
Median validation-to-test Sharpe gap by feature family in v2.
- Files: `07_v2_generalization_gap_by_feature_set.png`, `07_v2_generalization_gap_by_feature_set.svg`
- Tables: `plotting_tables\07_v2_generalization_gap_by_feature_set.csv`, `plotting_tables\05_v2_validation_vs_test_scatter_summary.csv`
- Note: Gaps are computed on checkpoint-selected runs under `checkpoint_robust_score`.

### `09_v2_fold_feature_heatmap_test_sharpe` - Ablation Ladder v2: fold-by-feature test Sharpe heatmap
Fold-level median test Sharpe in the current Horizon A experiment.
- Files: `09_v2_fold_feature_heatmap_test_sharpe.png`, `09_v2_fold_feature_heatmap_test_sharpe.svg`
- Tables: `plotting_tables\09_v2_fold_feature_heatmap_test_sharpe.csv`
- Note: Cell values are fold-level medians across seeds for runs selected with `checkpoint_robust_score`.

### `11_v2_selection_rule_comparison` - Ablation Ladder v2: post-hoc configuration-level selection rule comparison
Post-hoc comparison of configuration-level Sharpe-only and robust rules rebuilt from fold-level feature-set summaries.
- Files: `11_v2_selection_rule_comparison.png`, `11_v2_selection_rule_comparison.svg`
- Tables: `plotting_tables\11_v2_selection_rule_comparison.csv`, `plotting_tables\11_v2_selection_rule_comparison_tidy.csv`
- Note: Post-hoc configuration-level comparison. These rows do not necessarily correspond to the rule used inside the original experiment run.

### `13_v2_fold_winner_map` - Ablation Ladder v2: post-hoc fold winner map
Fold-by-fold post-hoc map showing which feature set each rebuilt configuration rule would have selected and whether it matched the test winner.
- Files: `13_v2_fold_winner_map.png`, `13_v2_fold_winner_map.svg`
- Tables: `plotting_tables\13_v2_fold_winner_map.csv`
- Note: Post-hoc configuration-level map. It shows what each rebuilt rule would have selected per fold, not the literal rule used inside the original training run.

### `15_v2_pairwise_test_sharpe_matrix` - Ablation Ladder v2: pairwise test-Sharpe matrix
Permutation-test matrix of pairwise mean test-Sharpe differences for v2.
- Files: `15_v2_pairwise_test_sharpe_matrix.png`, `15_v2_pairwise_test_sharpe_matrix.svg`
- Tables: `plotting_tables\15_v2_pairwise_test_sharpe_matrix_long.csv`, `plotting_tables\15_v2_pairwise_test_sharpe_matrix_matrix.csv`

### `16_v2_aligned_cumulative_return_paths_all_features` - Ablation Ladder v2: aligned cumulative return paths (all feature sets)
Aligned within-window cumulative return trajectories aggregated across runs for all v2 feature sets.
- Files: `16_v2_aligned_cumulative_return_paths_all_features.png`, `16_v2_aligned_cumulative_return_paths_all_features.svg`
- Tables: `plotting_tables\16_v2_aligned_cumulative_return_paths_all_features.csv`

### `17_v2_aligned_cumulative_return_paths_main_features` - Ablation Ladder v2: aligned cumulative return paths (main feature sets)
Aligned within-window cumulative return trajectories focused on the main v2 feature families.
- Files: `17_v2_aligned_cumulative_return_paths_main_features.png`, `17_v2_aligned_cumulative_return_paths_main_features.svg`
- Tables: `plotting_tables\17_v2_aligned_cumulative_return_paths_main_features.csv`

### `18_v2_regime_coverage_by_feature_set` - Ablation Ladder v2: regime coverage across test windows
Coverage diagnostics showing how many test days fall into each exogenous regime before any feature-specific interpretation.
- Files: `18_v2_regime_coverage_by_feature_set.png`, `18_v2_regime_coverage_by_feature_set.svg`
- Tables: `plotting_tables\18_v2_regime_coverage_by_feature_set.csv`
- Note: Coverage is shown once per regime because regime counts are identical across feature sets by construction.

### `19_v2_regime_sharpe_heatmap` - Ablation Ladder v2: regime-level Sharpe heatmap
Conservative heatmap of regime-level Sharpe, masking sparse cells instead of overstating them.
- Files: `19_v2_regime_sharpe_heatmap.png`, `19_v2_regime_sharpe_heatmap.svg`
- Tables: `plotting_tables\19_v2_regime_sharpe_heatmap.csv`
- Note: Cells with median regime coverage below 10 days were masked. Only Unknown / warm-up met that threshold.

### `20_v2_readme_summary_figure` - Ablation Ladder v2: README summary figure
Compact overview of median test Sharpe by feature set for use in the repository README.
- Files: `20_v2_readme_summary_figure.png`, `20_v2_readme_summary_figure.svg`
- Tables: `plotting_tables\20_v2_readme_summary_figure.csv`
- Note: Negative-control branches are labelled explicitly instead of being framed as co-equal hero candidates.

## Next-Cycle Final Figures

### `21_next_cycle_final_scoreboard` - Final Horizon A next-cycle ranking
Three-panel scorecard comparing median test Sharpe, median test return, and median benchmark-excess Sharpe across all historical and next-cycle feature sets.
- Files: `21_next_cycle_final_scoreboard.png`, `21_next_cycle_final_scoreboard.svg`
- Tables: `plotting_tables\21_next_cycle_final_scoreboard.csv`
- Note: This is the main final ranking figure. It shows why `base_macro` remains primary while vol, rates, and credit remain important candidates.

### `22_next_cycle_candidate_decision_heatmap` - Candidate decision heatmap
Cross-candidate normalized heatmap across Sharpe, return, positive-run rate, retention, and benchmark-relative metrics, now including the final interaction/gating branches.
- Files: `22_next_cycle_candidate_decision_heatmap.png`, `22_next_cycle_candidate_decision_heatmap.svg`
- Tables: `plotting_tables\22_next_cycle_candidate_decision_heatmap.csv`
- Note: This figure is intended to show complementary strengths rather than a single universal winner.

### `23_next_cycle_pairwise_delta_vs_base_macro` - Pairwise delta versus Base+Macro
Mean test-Sharpe deltas and permutation p-values for each feature set against `base_macro`.
- Files: `23_next_cycle_pairwise_delta_vs_base_macro.png`, `23_next_cycle_pairwise_delta_vs_base_macro.svg`
- Tables: `plotting_tables\23_next_cycle_pairwise_delta_vs_base_macro.csv`
- Note: This figure is the main guardrail against overclaiming a new family as better than the frozen reference.

### `24_next_cycle_fold_winner_map` - Fold-level actual test winners
Fold-level map of actual test winners after merging all next-cycle candidates with the historical panel.
- Files: `24_next_cycle_fold_winner_map.png`, `24_next_cycle_fold_winner_map.svg`
- Tables: `plotting_tables\24_next_cycle_fold_winner_map.csv`
- Note: The map highlights episodic candidate wins and motivates complementarity analysis.

### `25_next_cycle_selection_rule_degradation` - Selection-rule reliability after panel expansion
Comparison of historical v2 selection-rule performance versus the final all-candidate panel.
- Files: `25_next_cycle_selection_rule_degradation.png`, `25_next_cycle_selection_rule_degradation.svg`
- Tables: `plotting_tables\25_next_cycle_selection_rule_comparison.csv`
- Note: This is a key critical figure. Selection reliability deteriorated as the candidate panel expanded.

### `26_next_cycle_benchmark_relative_scatter` - Benchmark-relative candidate view
Scatter plot of median benchmark-excess return versus median benchmark-excess Sharpe, with point size reflecting Sharpe outperform rate.
- Files: `26_next_cycle_benchmark_relative_scatter.png`, `26_next_cycle_benchmark_relative_scatter.svg`
- Tables: `plotting_tables\26_next_cycle_benchmark_relative_scatter.csv`
- Note: The figure shows that no family clears the passive benchmark bar on median excess Sharpe.

### `27_next_cycle_regime_excess_return_heatmap` - Regime excess-return diagnostics
Median daily excess return versus benchmark by feature family and exogenous regime label.
- Files: `27_next_cycle_regime_excess_return_heatmap.png`, `27_next_cycle_regime_excess_return_heatmap.svg`
- Tables: `plotting_tables\27_next_cycle_regime_excess_return_heatmap.csv`
- Note: Regime cells are diagnostic only because some regime windows are short.

### `28_next_cycle_main_candidate_cumulative_returns` - Main retained candidate and interaction cumulative paths
Mean daily test cumulative-return paths for `base_macro`, the main retained candidate families, and the final interaction/gating branches.
- Files: `28_next_cycle_main_candidate_cumulative_returns.png`, `28_next_cycle_main_candidate_cumulative_returns.svg`
- Tables: `plotting_tables\28_next_cycle_main_candidate_cumulative_returns.csv`
- Note: This figure is meant for visual trajectory comparison, not for standalone statistical inference.

### `29_horizon_a_interaction_closeout_scoreboard` - Horizon A interaction closeout scoreboard
Focused closeout scorecard for the retained single-family inputs and all interaction/gating branches.
- Files: `29_horizon_a_interaction_closeout_scoreboard.png`, `29_horizon_a_interaction_closeout_scoreboard.svg`
- Tables: `plotting_tables\29_horizon_a_interaction_closeout_scoreboard.csv`
- Note: The figure shows why `xsec_sector_complementarity_v2` is a useful near-miss but not a promoted replacement for `base_macro`.

### `30_horizon_a_phase_boundary_selection` - Horizon A phase-boundary selection diagnostics
Selection-rule reliability across the historical panel, the single-family panel, the first interaction branch, and the final Horizon A panel.
- Files: `30_horizon_a_phase_boundary_selection.png`, `30_horizon_a_phase_boundary_selection.svg`
- Tables: `plotting_tables\30_horizon_a_phase_boundary_selection.csv`
- Note: This figure marks the transition away from feature-interaction search and into Phase-2 latent-action review.
