# Research Outputs Index

This file defines the current Ablation Ladder v2 output layout and cleanup guidance.

## Canonical Historical Output

`comparison_outputs`

Meaning:

- Frozen historical Ablation Ladder v2 reference outputs.
- Contains the original v2 feature ladder before the next-cycle candidate-family experiments.
- Should be kept because it is the baseline source used by the final merged analysis.

Recommended naming:

- Keep the directory name `comparison_outputs` for this commit to avoid breaking existing references.
- If a future cleanup commit renames it, use `research_outputs_historical_v2_reference`.

Do not merge this directory into a raw candidate folder. It is the historical reference anchor.

## Raw Next-Cycle Candidate Outputs

These folders should be kept as raw experiment artifacts:

- `research_outputs_next_cycle_xsec_dispersion_correlation_regime`
- `research_outputs_next_cycle_breadth_internal_structure`
- `research_outputs_next_cycle_sector_relative_context`
- `research_outputs_next_cycle_xsec_sector_gated_context`
- `research_outputs_next_cycle_credit_stress_proxies`
- `research_outputs_next_cycle_rates_term_structure_lsc`
- `research_outputs_next_cycle_analyst_or_fund_revision_features`
- `research_outputs_next_cycle_vol_term_or_implied_vol_proxy`
- `research_outputs_next_cycle_rates_credit_vol_risk_state_context`
- `research_outputs_next_cycle_xsec_sector_complementarity_v2`

Meaning:

- Each folder is a single-family candidate-only run.
- The rates/credit/vol folder is the first pre-registered interaction/gating v2 candidate-only run.
- The xsec/sector complementarity v2 folder is the final Horizon A feature-interaction candidate-only run.
- These are the audit trail for the final merged result.
- They should not be collapsed into one raw folder because keeping one folder per experiment makes provenance and reruns easier.

## Canonical Final Merged Output

Current final post-interaction merged output:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2`

Meaning:

- Merged analysis that combines `comparison_outputs`, every completed next-cycle candidate family, the rates/credit/vol risk-state branch, and the final xsec/sector complementarity v2 branch.
- This is the canonical final Horizon A feature-interaction analysis.
- The critical report is `critical_xsec_sector_complementarity_v2_report.md`.

Current post-interaction merged output:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state`

Meaning:

- Merged analysis that combines `comparison_outputs`, every completed next-cycle candidate family, and the first interaction/gating v2 branch.
- This is the superseded analysis after testing only `base_macro_rates_credit_vol_risk_state_context`.
- The critical report is `critical_rates_credit_vol_risk_state_context_report.md`.

Previous all-single-family merged output:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol`

Meaning:

- Final single-family merged analysis that combines `comparison_outputs` with every completed next-cycle candidate family before the interaction/gating v2 branch.
- This remains the canonical pre-interaction analysis.

If this folder is renamed in a future cleanup commit, use:

`research_outputs_merged_history_plus_next_cycle_all_candidates`

For the current commit, the existing name is kept to avoid path churn.

## Superseded Merged Outputs

These folders are intermediate snapshots and are superseded by the canonical final merged output:

- `merged_analysis_history_plus_xsec`
- `merged_analysis_history_plus_xsec_plus_breadth`
- `merged_analysis_history_plus_xsec_plus_breadth_raw_rebuild`
- `merged_analysis_history_plus_xsec_breadth_sector_gated_context`
- `merged_analysis_history_plus_xsec_breadth_sector_gated_credit`
- `merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates`
- `merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst`
- `merged_analysis_history_plus_all_candidates`
- `merged_analysis_history_plus_all_candidates_raw_rebuild`

`merged_analysis_history_plus_all_candidates` is useful only as a historical intermediate snapshot. Despite the name, it only merged the historical outputs with xsec, breadth, and sector. It does not include gated, credit, rates, analyst/revision, or vol proxy. It is therefore not the final all-candidate result.

Cleanup decision:

- These superseded merged folders can be deleted after the canonical final merged folder is committed and backed up.
- Do not delete the raw `research_outputs_next_cycle_*` folders unless the commit intentionally stores only derived summaries.

## Figure Outputs

Current figure directory:

`paper_figures`

New next-cycle figures:

- `21_next_cycle_final_scoreboard`
- `22_next_cycle_candidate_decision_heatmap`
- `23_next_cycle_pairwise_delta_vs_base_macro`
- `24_next_cycle_fold_winner_map`
- `25_next_cycle_selection_rule_degradation`
- `26_next_cycle_benchmark_relative_scatter`
- `27_next_cycle_regime_excess_return_heatmap`
- `28_next_cycle_main_candidate_cumulative_returns`
- `29_horizon_a_interaction_closeout_scoreboard`
- `30_horizon_a_phase_boundary_selection`

Source plotting tables live in:

`paper_figures/plotting_tables`

The generator is:

`generate_next_cycle_final_figures.py`

## Phase-2 Latent-Action Preparation

New notebook runs now persist policy action traces when the notebook runtime returns `df_action`:

- Raw run output: `walk_forward_test_actions.csv`
- Merged output: `walk_forward_test_actions_merged.csv`
- Rebuilt analysis output: `analysis/walk_forward_test_actions.csv`

These action traces are intentionally separate from `walk_forward_daily_test_returns.csv`. Daily returns remain the backtest/statistical surface; actions are the teacher dataset surface for Phase-2 latent-action diagnostics.

The first audit utility lives in the Phase-2 folder:

`..\Latent Actions\latent_action_phase2_tools.py`

It builds:

- `latent_action_teacher_matrix.csv`
- `latent_action_teacher_simple_codes.csv`
- `latent_action_teacher_action_summary.csv`
- `latent_action_teacher_code_counts.csv`
