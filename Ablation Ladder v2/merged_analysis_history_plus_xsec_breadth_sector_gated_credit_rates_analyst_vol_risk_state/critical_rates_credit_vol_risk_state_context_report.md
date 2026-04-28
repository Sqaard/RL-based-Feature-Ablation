# Critical Comparison Report: Rates/Credit/Vol Risk-State Context

## Scope

This report analyzes the first interaction/gating v2 branch:

`base_macro_rates_credit_vol_risk_state_context`

It compares the new candidate against the previous canonical all-candidate bundle:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol`

The rebuilt merged bundle is:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state`

Merge integrity:

- Raw run rows: `588`.
- Unique run keys: `588`.
- Feature sets: `14`.
- Folds: `14`.
- Seeds per feature set: `3`.
- Rebuild warnings: none.

## Executive Decision

Reject `base_macro_rates_credit_vol_risk_state_context` as a promotable branch.

The branch fails the pre-registered kill rules:

- It does not beat `base_macro` on median test Sharpe.
- It does not improve benchmark-relative excess Sharpe.
- It is never the actual fold-level test winner.
- It is selected by validation rules only in losing folds.
- It weakens the already fragile case for broad feature stacking.

This is a negative result for the first interaction/gating v2 design. It does not justify SSL, latent actions, or an architecture change.

## Main Ranking

| Rank | Feature set | Median test Sharpe | Mean test Sharpe | Median test return pct | Median excess Sharpe vs primary benchmark |
|---:|---|---:|---:|---:|---:|
| 1 | `base_macro` | 1.3378 | 1.2158 | 3.2622 | -0.2144 |
| 2 | `base_macro_vol_term_or_implied_vol_proxy` | 1.1405 | 1.1547 | 1.9797 | -0.2135 |
| 3 | `base_macro_gru` | 1.1058 | 1.1255 | 2.6674 | -0.1587 |
| 4 | `base_macro_rates_term_structure_lsc` | 1.0974 | 1.2647 | 2.3117 | -0.2336 |
| 5 | `base_macro_credit_stress_proxies` | 1.0709 | 1.2956 | 1.0683 | -0.1903 |
| 6 | `base_macro_xsec_dispersion_correlation_regime` | 1.0505 | 1.0702 | 2.6455 | -0.2463 |
| 7 | `base` | 0.9435 | 1.0188 | 2.3844 | -0.2266 |
| 8 | `base_macro_analyst_or_fund_revision_features` | 0.8848 | 1.1106 | 1.9805 | -0.3564 |
| 9 | `base_macro_sector_relative_context` | 0.8503 | 1.2001 | 2.3244 | -0.1661 |
| 10 | `base_macro_exogenous_plus` | 0.7911 | 1.1509 | 2.0444 | -0.2845 |
| 11 | `base_macro_xsec_sector_gated_context` | 0.6342 | 0.9714 | 1.4635 | -0.2856 |
| 12 | `base_macro_hmm` | 0.5466 | 0.9980 | 1.8147 | -0.5346 |
| 13 | `base_macro_rates_credit_vol_risk_state_context` | 0.5151 | 0.9524 | 0.8039 | -0.2919 |
| 14 | `base_macro_breadth_internal_structure` | 0.4827 | 1.0650 | 1.6654 | -0.2119 |

The new branch ranks near the bottom by median Sharpe and has the weakest median return among the main retained macro/risk candidates.

## Benchmark-Relative View

Against the primary benchmark, `dow30_equal_weight_rebalance_matched`, the risk-state branch has:

- Median excess return pct: `-0.9364`.
- Median excess Sharpe: `-0.2919`.
- Return outperform rate: `0.4048`.
- Sharpe outperform rate: `0.3571`.

This is worse than `base_macro` on median excess Sharpe and worse than the standalone retained rates, credit, and vol proxy branches on the main promotion criteria.

## Pairwise Evidence

Risk-state branch versus key references:

| Reference | Mean delta Sharpe, risk minus reference | Median delta Sharpe | Sharpe win rate | p-value |
|---|---:|---:|---:|---:|
| `base_macro` | -0.2634 | -0.2132 | 0.3810 | 0.0916 |
| `base_macro_rates_term_structure_lsc` | -0.3123 | -0.1017 | 0.3333 | 0.0241 |
| `base_macro_credit_stress_proxies` | -0.3432 | -0.2458 | 0.4048 | 0.0027 |
| `base_macro_vol_term_or_implied_vol_proxy` | -0.2023 | -0.1926 | 0.3333 | 0.1256 |
| `base_macro_xsec_dispersion_correlation_regime` | -0.1178 | 0.0199 | 0.5238 | 0.3505 |
| `base_macro_sector_relative_context` | -0.2477 | -0.2790 | 0.3810 | 0.0283 |
| `base_macro_xsec_sector_gated_context` | -0.0190 | 0.0219 | 0.5000 | 0.8978 |

Interpretation:

- The new branch is directionally worse than `base_macro`.
- It is materially worse than the standalone rates and credit families.
- The only close comparison is the already rejected `xsec_sector_gated_context`.
- The result argues against this specific risk-state interaction design, not against the existence of all possible interactions.

## Selection Layer

Adding the branch did not improve selection reliability.

| Selection rule | Selected median test Sharpe | Match rate | Median regret |
|---|---:|---:|---:|
| `robust_q25_retention` | 1.2102 | 0.1429 | 0.3721 |
| `sharpe_only` | 0.9562 | 0.1429 | 0.4734 |
| `robust_q25` | 0.7036 | 0.0714 | 0.5988 |

The risk-state branch was selected in five rule/fold rows:

- fold 02 under `robust_q25`,
- fold 12 under all three rules,
- fold 13 under `sharpe_only`.

It was wrong every time. It was never the actual test winner in any fold.

## Regime Diagnostics

The regime breakdown is not promotable evidence.

| Regime | Folds | Median days | Median daily return | Median excess return vs benchmark |
|---|---:|---:|---:|---:|
| `bear_high_vol` | 6 | 4.0 | 0.000645 | -0.000541 |
| `bear_low_vol` | 3 | 1.0 | 0.000145 | -0.000062 |
| `bull_high_vol` | 3 | 1.0 | 0.004385 | 0.000905 |
| `bull_low_vol` | 6 | 3.5 | 0.001408 | -0.000105 |
| `unknown` | 14 | 59.0 | 0.000425 | -0.000204 |

The only positive excess regime is `bull_high_vol`, but the median regime window is one day. That is diagnostic noise, not a basis for promotion. The branch is negative versus benchmark in the larger `unknown` bucket and in `bear_high_vol`, where a risk-state branch should have been most useful.

## Final Assessment

The first interaction/gating v2 branch should be killed.

Do not:

- promote it over `base_macro`,
- run a 5-seed extension for this branch,
- use it as evidence for SSL/state compression,
- use it as evidence for latent-action PPO,
- broaden into an all-family stack.

Keep:

- `base_macro` as the frozen reference.
- Standalone `vol_term_or_implied_vol_proxy`, `rates_term_structure_lsc`, and `credit_stress_proxies` as useful diagnostic families, but not as a combined risk-state stack.
- `xsec_dispersion_correlation_regime` and `sector_relative_context` as possible inputs for a second, redesigned structural complementarity test only if it is narrow and pre-registered.

