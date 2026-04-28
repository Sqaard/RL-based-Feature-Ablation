# Critical Comparison Report: Xsec/Sector Complementarity v2

## Scope

This report analyzes the final planned Horizon A feature-interaction branch:

`base_macro_xsec_sector_complementarity_v2`

It compares the new candidate against the prior post-interaction bundle:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state`

The rebuilt final merged bundle is:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2`

Merge integrity:

- Raw run rows: `630`.
- Unique run keys: `630`.
- Feature sets: `15`.
- Folds: `14`.
- Seeds per feature set: `3`.
- Rebuild warnings: none.

## Executive Decision

Do not promote `base_macro_xsec_sector_complementarity_v2` over `base_macro`.

This is the best interaction/gating branch tested so far, but it still fails the pre-registered promotion bar:

- It ranks third by median test Sharpe, behind `base_macro` and `base_macro_vol_term_or_implied_vol_proxy`.
- It does not beat `base_macro` on median test Sharpe.
- It does not improve primary benchmark-relative excess Sharpe versus `base_macro`.
- It is never the actual fold-level test winner.
- It is selected by validation rules only in losing folds.
- Adding it worsens selection-rule median regret.

This should close the Horizon A feature-interaction search unless the project explicitly relaxes the kill rules. The evidence does not justify another feature stack.

## Main Ranking

| Rank | Feature set | Median test Sharpe | Mean test Sharpe | Median test return pct | Median excess Sharpe vs primary benchmark |
|---:|---|---:|---:|---:|---:|
| 1 | `base_macro` | 1.3378 | 1.2158 | 3.2622 | -0.2144 |
| 2 | `base_macro_vol_term_or_implied_vol_proxy` | 1.1405 | 1.1547 | 1.9797 | -0.2135 |
| 3 | `base_macro_xsec_sector_complementarity_v2` | 1.1347 | 1.1643 | 2.7852 | -0.2232 |
| 4 | `base_macro_gru` | 1.1058 | 1.1255 | 2.6674 | -0.1587 |
| 5 | `base_macro_rates_term_structure_lsc` | 1.0974 | 1.2647 | 2.3117 | -0.2336 |
| 6 | `base_macro_credit_stress_proxies` | 1.0709 | 1.2956 | 1.0683 | -0.1903 |
| 7 | `base_macro_xsec_dispersion_correlation_regime` | 1.0505 | 1.0702 | 2.6455 | -0.2463 |
| 8 | `base` | 0.9435 | 1.0188 | 2.3844 | -0.2266 |
| 9 | `base_macro_analyst_or_fund_revision_features` | 0.8848 | 1.1106 | 1.9805 | -0.3564 |
| 10 | `base_macro_sector_relative_context` | 0.8503 | 1.2001 | 2.3244 | -0.1661 |
| 11 | `base_macro_exogenous_plus` | 0.7911 | 1.1509 | 2.0444 | -0.2845 |
| 12 | `base_macro_xsec_sector_gated_context` | 0.6342 | 0.9714 | 1.4635 | -0.2856 |
| 13 | `base_macro_hmm` | 0.5466 | 0.9980 | 1.8147 | -0.5346 |
| 14 | `base_macro_rates_credit_vol_risk_state_context` | 0.5151 | 0.9524 | 0.8039 | -0.2919 |
| 15 | `base_macro_breadth_internal_structure` | 0.4827 | 1.0650 | 1.6654 | -0.2119 |

The v2 complementarity branch is a large improvement over the failed `xsec_sector_gated_context`, but it still does not clear the reference baseline.

## Benchmark-Relative View

Against the primary benchmark, `dow30_equal_weight_rebalance_matched`, the branch has:

- Median excess return pct: `-0.7169`.
- Median excess Sharpe: `-0.2232`.
- Return outperform rate: `0.3571`.
- Sharpe outperform rate: `0.3571`.

This is slightly worse than `base_macro` on median excess Sharpe (`-0.2232` vs `-0.2144`) and worse than `base_macro_vol_term_or_implied_vol_proxy` on the same field (`-0.2135`).

The branch does beat the trend-filter overlay median, but that is not enough because the primary benchmark and frozen `base_macro` reference remain the promotion bar.

## Pairwise Evidence

Xsec/sector complementarity v2 versus key references:

| Reference | Mean delta Sharpe, v2 minus reference | Median delta Sharpe | Sharpe win rate | p-value |
|---|---:|---:|---:|---:|
| `base_macro` | -0.0516 | -0.1454 | 0.4286 | 0.7235 |
| `base_macro_vol_term_or_implied_vol_proxy` | 0.0095 | -0.0946 | 0.3810 | 0.9327 |
| `base_macro_xsec_dispersion_correlation_regime` | 0.0940 | -0.0580 | 0.4762 | 0.4962 |
| `base_macro_sector_relative_context` | -0.0358 | -0.0856 | 0.3810 | 0.7434 |
| `base_macro_xsec_sector_gated_context` | 0.1928 | 0.1792 | 0.6905 | 0.1319 |
| `base_macro_rates_credit_vol_risk_state_context` | 0.2119 | 0.1451 | 0.5238 | 0.1703 |
| `base_macro_rates_term_structure_lsc` | -0.1004 | -0.1194 | 0.4286 | 0.4129 |
| `base_macro_credit_stress_proxies` | -0.1313 | -0.1595 | 0.4286 | 0.2362 |

Interpretation:

- The new branch is much better than the old xsec-sector gate.
- It is better than the rejected rates/credit/vol interaction stack.
- It is statistically indistinguishable from the top single-family candidates.
- It does not show pairwise superiority over `base_macro`.

## Selection Layer

Adding the branch worsened selection reliability.

| Panel | Rule | Selected median test Sharpe | Match rate | Median regret |
|---|---|---:|---:|---:|
| Before xsec/sector v2 | `robust_q25_retention` | 1.2102 | 0.1429 | 0.3721 |
| Final panel | `robust_q25_retention` | 1.0281 | 0.1429 | 0.5177 |
| Before xsec/sector v2 | `sharpe_only` | 0.9562 | 0.1429 | 0.4734 |
| Final panel | `sharpe_only` | 0.7741 | 0.1429 | 0.5059 |
| Before xsec/sector v2 | `robust_q25` | 0.7036 | 0.0714 | 0.5988 |
| Final panel | `robust_q25` | 0.6695 | 0.0714 | 0.7580 |

The new branch was selected in six rule/fold rows:

- fold 01 under `robust_q25_retention`,
- fold 04 under `robust_q25` and `robust_q25_retention`,
- fold 10 under all three rules.

It was wrong every time. It was never the actual test winner in any fold.

Actual fold-level winners in the final panel did not include `base_macro_xsec_sector_complementarity_v2`.

## Regime Diagnostics

Regime evidence is mixed and not promotable.

| Regime | Folds | Median days | Median daily return | Median excess return vs benchmark |
|---|---:|---:|---:|---:|
| `bear_high_vol` | 6 | 4.0 | 0.001921 | -0.000010 |
| `bear_low_vol` | 3 | 1.0 | 0.000837 | -0.001106 |
| `bull_high_vol` | 3 | 1.0 | 0.003088 | 0.000486 |
| `bull_low_vol` | 6 | 3.5 | -0.000446 | -0.000796 |
| `unknown` | 14 | 59.0 | 0.000511 | -0.000213 |

The labeled regime windows are short. The large `unknown` bucket remains benchmark-negative, so regime diagnostics do not rescue the branch.

## Final Assessment

`base_macro_xsec_sector_complementarity_v2` is a useful negative/near-miss result:

- It validates that redesigning the xsec/sector interaction was better than copying the failed gate.
- It shows that narrow complementarity features can be competitive.
- It does not show enough evidence to replace `base_macro`.

Decision:

- Keep `base_macro` as the primary reference.
- Do not run a broad all-family stack.
- Do not run another Horizon A feature-interaction variant.
- Do not run a 5-seed extension unless the project explicitly changes the promotion bar.
- Move to Phase-2 review.

Recommended Phase-2 ordering:

1. Latent-action branch, starting with simple discretization and teacher-action dataset.
2. Behavior cloning warm start only after the discretization baseline is inspected.
3. PPO fine-tuning only after the staged action-space baseline is defensible.
4. SSL/domain-invariance work remains later than latent actions unless there is new evidence that state representation, not action regularization or benchmark discipline, is the bottleneck.

