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


## Final Ranking

The final raw median test-Sharpe ranking is:

| Rank | Feature set | Median test Sharpe | Mean test Sharpe | Median test return pct | Decision |
|---:|---|---:|---:|---:|---|
| 1 | `base_macro` | 1.3378 | 1.2158 | 3.2622 | Keep as primary reference |
| 2 | `base_macro_vol_term_or_implied_vol_proxy` | 1.1405 | 1.1547 | 1.9797 | Keep as top-tier risk-state candidate |
| 3 | `base_macro_gru` | 1.1058 | 1.1255 | 2.6674 | Negative control, not promoted |
| 4 | `base_macro_rates_term_structure_lsc` | 1.0974 | 1.2647 | 2.3117 | Keep as top-tier macro candidate |
| 5 | `base_macro_credit_stress_proxies` | 1.0709 | 1.2956 | 1.0683 | Keep as top-tier stress candidate |
| 6 | `base_macro_xsec_dispersion_correlation_regime` | 1.0505 | 1.0702 | 2.6455 | Keep as stable structural candidate |
| 7 | `base` | 0.9435 | 1.0188 | 2.3844 | Historical anchor |
| 8 | `base_macro_analyst_or_fund_revision_features` | 0.8848 | 1.1106 | 1.9805 | Do not promote |
| 9 | `base_macro_sector_relative_context` | 0.8503 | 1.2001 | 2.3244 | Keep as episodic structural candidate |
| 10 | `base_macro_exogenous_plus` | 0.7911 | 1.1509 | 2.0444 | Historical negative result |
| 11 | `base_macro_xsec_sector_gated_context` | 0.6342 | 0.9714 | 1.4635 | Do not promote |
| 12 | `base_macro_hmm` | 0.5466 | 0.9980 | 1.8147 | Negative control |
| 13 | `base_macro_breadth_internal_structure` | 0.4827 | 1.0650 | 1.6653 | Low-priority diagnostic only |

The primary conclusion did not change: `base_macro` remains the reference setup. None of the new single-family candidates justifies replacing it.

## Critical Interpretation

The most important result is not a single new winner. The result is that different candidate families contribute different partial signals:

- `vol_term_or_implied_vol_proxy` is the best new family by median test Sharpe and has a relatively strong benchmark-relative profile, but it is not a statistically reliable improvement over `base_macro`.
- `rates_term_structure_lsc` is balanced and economically interpretable, with strong mean Sharpe and respectable median return.
- `credit_stress_proxies` has the best mean Sharpe, but weak median return, which points to episodic stress-regime value rather than stable return dominance.
- `xsec_dispersion_correlation_regime` remains the cleanest internally derived structural family and has strong retention/generalization diagnostics.
- `sector_relative_context` is episodic: weak median ranking but useful fold wins and high positive-Sharpe frequency.
- `analyst_or_fund_revision_features` should not be promoted because it weakened benchmark-relative and selection-rule evidence.
- `xsec_sector_gated_context` should not be extended mechanically; the first gated design underperformed.

## Interaction And Gating v2 Decision

The next experiment should not be another standalone run of the same families.

Recommended interaction/gating v2 shortlist:

- Primary interaction input set: `rates_term_structure_lsc`, `credit_stress_proxies`, `vol_term_or_implied_vol_proxy`.
- Secondary structural input set: `xsec_dispersion_correlation_regime`, `sector_relative_context`.
- Optional diagnostic-only gate input: `breadth_internal_structure`, only if the gate is pre-registered and narrow.

Do not include in the first interaction/gating v2 design:

- `analyst_or_fund_revision_features`.
- `xsec_sector_gated_context` as implemented in v1 of the gate.
- `base_macro_hmm` or `base_macro_gru`, except as negative controls.
- `base_macro_exogenous_plus`.

Recommended first interaction branch:

`base_macro_rates_credit_vol_risk_state_context`

Recommended second interaction branch, only after the first branch is analyzed:

`base_macro_xsec_sector_complementarity_v2`

The second branch should be redesigned rather than copied from `base_macro_xsec_sector_gated_context`, because the first gated implementation was not validated.

## Output Layout

Important directories:

- `comparison_outputs`: frozen historical Ablation Ladder v2 reference outputs.
- `research_outputs_next_cycle_*`: raw candidate-only outputs for each next-cycle feature family.
- `merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol`: canonical final merged analysis with historical and all next-cycle candidates.
- `paper_figures`: old v2 figures plus final next-cycle figures.
- `paper_figures/plotting_tables`: source CSV tables for figures.

Detailed cleanup and naming guidance is in `RESEARCH_OUTPUTS_INDEX.md`.

