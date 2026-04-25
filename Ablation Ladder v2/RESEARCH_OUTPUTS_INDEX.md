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

Meaning:

- Each folder is a single-family candidate-only run.
- These are the audit trail for the final merged result.
- They should not be collapsed into one raw folder because keeping one folder per experiment makes provenance and reruns easier.

## Canonical Final Merged Output

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol`

Meaning:

- Final merged analysis that combines `comparison_outputs` with every completed next-cycle candidate family.
- This is the canonical all-candidate analysis for the current commit.

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

Source plotting tables live in:

`paper_figures/plotting_tables`

The generator is:

`generate_next_cycle_final_figures.py`

