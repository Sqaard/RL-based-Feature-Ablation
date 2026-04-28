# Horizon A Closeout

Status: closed.

Horizon A feature-interaction search should not continue with another feature stack under the frozen PPO setup.

## Decision

Keep `base_macro` as the primary reference.

Do not promote either interaction/gating branch:

- `base_macro_rates_credit_vol_risk_state_context`
- `base_macro_xsec_sector_complementarity_v2`

The second branch is a useful near-miss, not a replacement. It improved on the failed xsec-sector gate and ranked near the top by median test Sharpe, but it did not clear the pre-registered promotion bar.

## Evidence

Final canonical bundle:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2`

Key final metrics:

| Feature set | Median test Sharpe | Median excess Sharpe vs primary benchmark | Actual winner folds | Decision |
|---|---:|---:|---:|---|
| `base_macro` | 1.3378 | -0.2144 | 2 | Primary reference |
| `base_macro_vol_term_or_implied_vol_proxy` | 1.1405 | -0.2135 | 1 | Retain as diagnostic/top-tier family |
| `base_macro_xsec_sector_complementarity_v2` | 1.1347 | -0.2232 | 0 | Near-miss, do not promote |
| `base_macro_rates_term_structure_lsc` | 1.0974 | -0.2336 | 2 | Retain as diagnostic/top-tier family |
| `base_macro_credit_stress_proxies` | 1.0709 | -0.1903 | 2 | Retain as episodic stress family |
| `base_macro_rates_credit_vol_risk_state_context` | 0.5151 | -0.2919 | 0 | Reject |

Selection-rule diagnostics worsened as the panel expanded:

| Panel | Robust retention selected median Sharpe | Robust retention median regret | Robust retention winner-match rate |
|---|---:|---:|---:|
| Historical v2 | 1.3652 | 0.0363 | 0.4286 |
| Single-family panel | 1.2102 | 0.3721 | 0.1429 |
| + Risk-state interaction | 1.2102 | 0.3721 | 0.1429 |
| Final Horizon A | 1.0281 | 0.5177 | 0.1429 |

This means the bottleneck is no longer "find one more exogenous feature stack." Larger feature panels are making selection harder without producing a benchmark-relative OOS winner.

## Updated Figures

The closeout evidence is now reflected in:

- `paper_figures/21_next_cycle_final_scoreboard.png`
- `paper_figures/22_next_cycle_candidate_decision_heatmap.png`
- `paper_figures/27_next_cycle_regime_excess_return_heatmap.png`
- `paper_figures/28_next_cycle_main_candidate_cumulative_returns.png`
- `paper_figures/29_horizon_a_interaction_closeout_scoreboard.png`
- `paper_figures/30_horizon_a_phase_boundary_selection.png`

## Phase Boundary

Next stage: Phase-2 latent-action review.

Do not start generic SSL/state-compression first. The current evidence says the observation feature search has reached diminishing returns under this frozen PPO setup. The nearer intervention is action-space regularization and action modeling:

1. export teacher action traces,
2. audit simple discretized action codes,
3. build a teacher action dataset,
4. only then test tokenizer, behavior cloning warm-start, and PPO fine-tuning.

The concrete Phase-2 entry plan is in:

`..\Latent Actions\LATENT_ACTIONS_PHASE2_PLAN.md`
