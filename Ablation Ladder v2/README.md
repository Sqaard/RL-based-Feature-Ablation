# Ablation Ladder v2

Ablation Ladder v2 is the current Horizon A research package for the Dow 30 PPO feature-ablation work.

The original v2 experiment established a cleaner walk-forward reporting framework around the frozen reference PPO setup. The next-cycle work in this folder extends that baseline with stronger benchmark-relative reporting, single-family candidate experiments, merged final analysis, and updated figures.

## Reference Setup

The reference training setup stayed fixed throughout the next-cycle experiments:

- Algorithm: `PPO`.
- Reward: `custom_reward`.
- Policy: `custom_mlp_policy`.
- Checkpoint selection: `checkpoint_robust_score`.
- Configuration selection diagnostic: `robust_q25_retention`.
- Seeds: `42`, `123`, `999`.
- Costs: `buy_cost_pct = 0.001`, `sell_cost_pct = 0.001`.
- Action setting: long-only box action scaled by `hmax`.

The next-cycle feature work should therefore be interpreted as feature-family evidence, not as an architecture upgrade.

## Before The Next-Cycle Plan

Before `NEXT_EXPERIMENT_PLAN.md`, the stored Ablation Ladder v2 evidence supported four main conclusions.

- `base_macro` was the strongest robust reference feature set. In `comparison_outputs/corrected_walk_forward_summary.csv`, its median test Sharpe was `1.3378`.
- `base_macro_exogenous_plus` did not improve over `base_macro`, so adding generic exogenous/calendar context was not enough.
- `base_macro_hmm` and `base_macro_gru` were retained as negative-control learned-feature branches. GRU was less damaging than HMM, but neither became the main research direction.
- The biggest v2 contribution was methodological: better reporting, daily test exports, selection-rule diagnostics, and regime diagnostics.

The next-cycle plan therefore correctly shifted the project away from architecture escalation and toward benchmark hardening plus one-at-a-time feature-family science.

## What Was Implemented After The Plan

The next-cycle implementation added the missing experiment infrastructure and candidate-family runs:

- Benchmark suite reporting was added and propagated into merged analysis outputs.
- Candidate-only launch/preflight infrastructure was used for one-family-at-a-time experiments.
- New candidate feature families were trained and exported independently.
- Historical v2 outputs were merged with all next-cycle candidates into a single final analysis bundle.
- Per-family critical reports were written for credit, rates, analyst/revision, and vol proxy.
- Final all-candidate figures were generated into `paper_figures`.
- Horizon A closeout figures were added after both interaction/gating branches.
- Phase-2 latent-action preparation now includes test-action trace export and a teacher-action audit utility.

The final canonical merged analysis after all single-family and interaction/gating branches is:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2`

Its main analysis directory is:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2/analysis`

## Final Ranking

The final raw median test-Sharpe ranking is:

| Rank | Feature set | Median test Sharpe | Mean test Sharpe | Median test return pct | Decision |
|---:|---|---:|---:|---:|---|
| 1 | `base_macro` | 1.3378 | 1.2158 | 3.2622 | Keep as primary reference |
| 2 | `base_macro_vol_term_or_implied_vol_proxy` | 1.1405 | 1.1547 | 1.9797 | Keep as top-tier risk-state candidate |
| 3 | `base_macro_xsec_sector_complementarity_v2` | 1.1347 | 1.1643 | 2.7852 | Near-miss interaction branch, not promoted |
| 4 | `base_macro_gru` | 1.1058 | 1.1255 | 2.6674 | Negative control, not promoted |
| 5 | `base_macro_rates_term_structure_lsc` | 1.0974 | 1.2647 | 2.3117 | Keep as top-tier macro candidate |
| 6 | `base_macro_credit_stress_proxies` | 1.0709 | 1.2956 | 1.0683 | Keep as top-tier stress candidate |
| 7 | `base_macro_xsec_dispersion_correlation_regime` | 1.0505 | 1.0702 | 2.6455 | Keep as stable structural candidate |
| 8 | `base` | 0.9435 | 1.0188 | 2.3844 | Historical anchor |
| 9 | `base_macro_analyst_or_fund_revision_features` | 0.8848 | 1.1106 | 1.9805 | Do not promote |
| 10 | `base_macro_sector_relative_context` | 0.8503 | 1.2001 | 2.3244 | Keep as episodic structural candidate |
| 11 | `base_macro_exogenous_plus` | 0.7911 | 1.1509 | 2.0444 | Historical negative result |
| 12 | `base_macro_xsec_sector_gated_context` | 0.6342 | 0.9714 | 1.4635 | Do not promote |
| 13 | `base_macro_hmm` | 0.5466 | 0.9980 | 1.8147 | Negative control |
| 14 | `base_macro_rates_credit_vol_risk_state_context` | 0.5151 | 0.9524 | 0.8039 | Reject interaction branch |
| 15 | `base_macro_breadth_internal_structure` | 0.4827 | 1.0650 | 1.6654 | Low-priority diagnostic only |

The primary conclusion did not change: `base_macro` remains the reference setup. None of the new single-family or interaction/gating candidates justifies replacing it.

## Critical Interpretation

The most important result is not a single new winner. The result is that different candidate families contribute different partial signals:

- `vol_term_or_implied_vol_proxy` is the best new family by median test Sharpe and has a relatively strong benchmark-relative profile, but it is not a statistically reliable improvement over `base_macro`.
- `rates_term_structure_lsc` is balanced and economically interpretable, with strong mean Sharpe and respectable median return.
- `credit_stress_proxies` has the best mean Sharpe, but weak median return, which points to episodic stress-regime value rather than stable return dominance.
- `xsec_dispersion_correlation_regime` remains the cleanest internally derived structural family and has strong retention/generalization diagnostics.
- `sector_relative_context` is episodic: weak median ranking but useful fold wins and high positive-Sharpe frequency.
- `analyst_or_fund_revision_features` should not be promoted because it weakened benchmark-relative and selection-rule evidence.
- `xsec_sector_gated_context` should not be extended mechanically; the first gated design underperformed.

## Interaction And Gating v2 Plan And Outcome

The final Horizon A branch tested interaction/gating ideas rather than another standalone single-family run.

Recommended interaction/gating v2 shortlist:

- Primary interaction input set: `rates_term_structure_lsc`, `credit_stress_proxies`, `vol_term_or_implied_vol_proxy`.
- Secondary structural input set: `xsec_dispersion_correlation_regime`, `sector_relative_context`.
- Optional diagnostic-only gate input: `breadth_internal_structure`, only if the gate is pre-registered and narrow.

Do not include in the first interaction/gating v2 design:

- `analyst_or_fund_revision_features`.
- `xsec_sector_gated_context` as implemented in v1 of the gate.
- `base_macro_hmm` or `base_macro_gru`, except as negative controls.
- `base_macro_exogenous_plus`.

First interaction branch:

`base_macro_rates_credit_vol_risk_state_context`

Second interaction branch:

`base_macro_xsec_sector_complementarity_v2`

The second branch was redesigned rather than copied from `base_macro_xsec_sector_gated_context`, because the first gated implementation was not validated.

## Final Interaction/Gating Stage

The final planned Horizon A feature-interaction branch has now been run:

- Candidate family: `xsec_sector_complementarity_v2`.
- Feature set: `base_macro_xsec_sector_complementarity_v2`.
- Raw output folder: `research_outputs_next_cycle_xsec_sector_complementarity_v2`.
- Final merged bundle: `merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2`.

Decision: do not promote over `base_macro`.

The branch ranked third of fifteen feature sets by median test Sharpe and clearly improved on the failed `xsec_sector_gated_context`, but it did not beat `base_macro`, did not improve the primary benchmark-relative bar, and was never the actual fold-level test winner.

Plan and results:

- `XSEC_SECTOR_COMPLEMENTARITY_V2_PLAN.md`
- `XSEC_SECTOR_COMPLEMENTARITY_V2_RESULTS.md`
- `merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2/critical_xsec_sector_complementarity_v2_report.md`

## Interaction/Gating v2 Result

The first interaction/gating v2 branch has now been run and merged:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state`

Decision: reject `base_macro_rates_credit_vol_risk_state_context`.

The branch ranked 13th of 14 feature sets by median test Sharpe, was never the actual fold-level test winner, and did not improve benchmark-relative evidence versus `base_macro`.

Summary results are in `INTERACTION_GATING_V2_RESULTS.md`. The full critical report is in:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state/critical_rates_credit_vol_risk_state_context_report.md`

Horizon A feature-interaction search is closed here. The next project stage is Phase-2 review, with latent actions ahead of generic SSL/state compression.

## Output Layout

Important directories:

- `comparison_outputs`: frozen historical Ablation Ladder v2 reference outputs.
- `research_outputs_next_cycle_*`: raw candidate-only outputs for each next-cycle feature family.
- `merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2`: canonical final merged analysis after both interaction/gating v2 branches.
- `paper_figures`: old v2 figures plus final next-cycle figures.
- `paper_figures/plotting_tables`: source CSV tables for figures.

Detailed cleanup and naming guidance is in `RESEARCH_OUTPUTS_INDEX.md`.

## Final Reports And Figures

Core report:

- `NEXT_CYCLE_FINAL_RANKING.md`

Per-family critical conclusions are integrated into:

- `NEXT_CYCLE_FINAL_RANKING.md`
- `merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol/critical_vol_proxy_comparison_report.md`

Earlier per-family report files may exist locally in superseded intermediate merge folders, but those folders are not required for the final committed analysis.

Updated final figures:

- `paper_figures/21_next_cycle_final_scoreboard.png`
- `paper_figures/22_next_cycle_candidate_decision_heatmap.png`
- `paper_figures/23_next_cycle_pairwise_delta_vs_base_macro.png`
- `paper_figures/24_next_cycle_fold_winner_map.png`
- `paper_figures/25_next_cycle_selection_rule_degradation.png`
- `paper_figures/26_next_cycle_benchmark_relative_scatter.png`
- `paper_figures/27_next_cycle_regime_excess_return_heatmap.png`
- `paper_figures/28_next_cycle_main_candidate_cumulative_returns.png`
- `paper_figures/29_horizon_a_interaction_closeout_scoreboard.png`
- `paper_figures/30_horizon_a_phase_boundary_selection.png`

The figure generator is:

`generate_next_cycle_final_figures.py`

## Phase-2 Entry

The next stage is not another Horizon A feature stack. It starts with latent-action staging:

- rerun or continue from a teacher policy with `walk_forward_test_actions.csv` enabled,
- audit the teacher action trace with `..\Latent Actions\latent_action_phase2_tools.py`,
- test simple discretized action codes before tokenizer, behavior cloning, or PPO fine-tuning,
- keep the same benchmark-relative discipline as Horizon A.

Detailed plan:

- `HORIZON_A_CLOSEOUT.md`
- `..\Latent Actions\LATENT_ACTIONS_PHASE2_PLAN.md`

## Commit Scope

This folder is ready to be committed as an extension of the prior Ablation Ladder v2 work once the local checkout is in a real git repository.

The commit should be framed as:

`Add next-cycle Ablation Ladder v2 candidate analysis and final ranking`
