# Critical Comparison Report: Vol Term Or Implied Vol Proxy

## Scope

This report compares `base_macro_vol_term_or_implied_vol_proxy` against the historical Horizon A panel and the previously tested next-cycle feature families:

- Historical/reference families from `comparison_outputs`.
- `base_macro_xsec_dispersion_correlation_regime`.
- `base_macro_breadth_internal_structure`.
- `base_macro_sector_relative_context`.
- `base_macro_xsec_sector_gated_context`.
- `base_macro_credit_stress_proxies`.
- `base_macro_rates_term_structure_lsc`.
- `base_macro_analyst_or_fund_revision_features`.
- `base_macro_vol_term_or_implied_vol_proxy`.

The merged analysis was rebuilt into:

`Ablation Ladder v2/merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol`

The lag-clean dataset used for the merged rebuild was:

`processed_final_fixed_external_lagclean_full.csv`

## Warning Review

During the notebook run, `empyrical.stats` emitted:

`RuntimeWarning: invalid value encountered in scalar divide`

The warning comes from a tail-ratio style calculation where the denominator can become zero or invalid for a short or degenerate return series.

Exported artifact checks:

- `unique_run_level_results.csv` has no NaN or infinite numeric values.
- `corrected_walk_forward_summary.csv` has no NaN or infinite numeric values.
- `corrected_walk_forward_summary_with_primary_benchmark.csv` has no NaN or infinite numeric values.
- `benchmark_run_level_metrics.csv` has no NaN or infinite numeric values.
- Exported test daily returns did not show zero or near-zero 5th-percentile denominators when checked by run.
- Regime-level `sharpe` contains NaN values for short regime windows, which is expected and is explicitly handled through `insufficient_days_for_sharpe`.

Decision: the warning does not invalidate the primary run-level, summary, or benchmark-relative artifacts. It should still be tracked as a hygiene issue, but it is not a reason to discard the experiment.

## Executive Decision

`base_macro_vol_term_or_implied_vol_proxy` should be retained as a top-tier diagnostic candidate, but it should not replace `base_macro` as the primary reference family.

The vol proxy family is strong by median Sharpe and benchmark-relative metrics. It ranks second by median Sharpe, directly behind `base_macro`, and has one of the better primary-benchmark excess Sharpe profiles. However, it is not strong by mean Sharpe, has only moderate median return, and does not show statistically reliable pairwise superiority over the main contenders.

## Main Summary

| Feature set | Median test Sharpe | Mean test Sharpe | Median test return pct | Positive Sharpe rate |
|---|---:|---:|---:|---:|
| `base_macro` | 1.337815 | 1.215812 | 3.262182 | 0.619048 |
| `base_macro_vol_term_or_implied_vol_proxy` | 1.140471 | 1.154709 | 1.979735 | 0.595238 |
| `base_macro_gru` | 1.105760 | 1.125519 | 2.667408 | 0.595238 |
| `base_macro_rates_term_structure_lsc` | 1.097406 | 1.264703 | 2.311667 | 0.642857 |
| `base_macro_credit_stress_proxies` | 1.070871 | 1.295586 | 1.068273 | 0.595238 |
| `base_macro_xsec_dispersion_correlation_regime` | 1.050470 | 1.070218 | 2.645473 | 0.523810 |
| `base` | 0.943522 | 1.018800 | 2.384394 | 0.571429 |
| `base_macro_analyst_or_fund_revision_features` | 0.884806 | 1.110608 | 1.980467 | 0.571429 |
| `base_macro_sector_relative_context` | 0.850349 | 1.200099 | 2.324392 | 0.666667 |
| `base_macro_exogenous_plus` | 0.791133 | 1.150858 | 2.044412 | 0.595238 |
| `base_macro_xsec_sector_gated_context` | 0.634204 | 0.971413 | 1.463510 | 0.595238 |
| `base_macro_hmm` | 0.546588 | 0.998040 | 1.814671 | 0.571429 |
| `base_macro_breadth_internal_structure` | 0.482745 | 1.064975 | 1.665350 | 0.619048 |

Key interpretation:

- Vol proxy is second by median test Sharpe.
- Vol proxy is not top-tier by mean test Sharpe; it ranks behind credit, rates, `base_macro`, and sector.
- Its median return is moderate, below `base_macro`, GRU, xsec, base, sector, and rates.
- The family looks more like a risk-shaping or defensive signal than a return-dominant standalone family.

## Benchmark-Relative View

| Feature set | Median excess return pct | Median excess Sharpe | Return outperform rate | Sharpe outperform rate |
|---|---:|---:|---:|---:|
| `base_macro_gru` | -0.082795 | -0.158732 | 0.500000 | 0.285714 |
| `base_macro_sector_relative_context` | -0.348832 | -0.166098 | 0.380952 | 0.357143 |
| `base_macro_credit_stress_proxies` | -0.490346 | -0.190267 | 0.476190 | 0.428571 |
| `base_macro_breadth_internal_structure` | -1.091131 | -0.211936 | 0.428571 | 0.380952 |
| `base_macro_vol_term_or_implied_vol_proxy` | -0.197218 | -0.213458 | 0.452381 | 0.452381 |
| `base_macro` | -0.791446 | -0.214426 | 0.428571 | 0.428571 |
| `base` | -1.030975 | -0.226635 | 0.452381 | 0.380952 |
| `base_macro_rates_term_structure_lsc` | -0.670529 | -0.233650 | 0.380952 | 0.428571 |
| `base_macro_xsec_dispersion_correlation_regime` | -0.480941 | -0.246297 | 0.428571 | 0.357143 |
| `base_macro_exogenous_plus` | -0.503158 | -0.284472 | 0.428571 | 0.380952 |
| `base_macro_xsec_sector_gated_context` | -1.026479 | -0.285585 | 0.428571 | 0.333333 |
| `base_macro_analyst_or_fund_revision_features` | -1.427168 | -0.356440 | 0.333333 | 0.309524 |
| `base_macro_hmm` | -1.566051 | -0.534555 | 0.333333 | 0.309524 |

The benchmark-relative result is one of the strongest arguments for keeping vol proxy. It is close to `base_macro` on excess Sharpe and better than `base_macro` on median excess return and benchmark outperform rates.

This is still not a promotion signal because median excess Sharpe remains negative.

## Pairwise Evidence

| Reference | Mean delta Sharpe, vol minus reference | Median delta Sharpe | Sharpe win rate | Mean delta return pct | Return win rate |
|---|---:|---:|---:|---:|---:|
| `base_macro` | -0.061103 | -0.013772 | 0.500000 | -0.071797 | 0.500000 |
| `base_macro_rates_term_structure_lsc` | -0.109994 | 0.044643 | 0.571429 | -0.123733 | 0.523810 |
| `base_macro_credit_stress_proxies` | -0.140877 | -0.084070 | 0.476190 | 0.219852 | 0.452381 |
| `base_macro_analyst_or_fund_revision_features` | 0.044101 | 0.146611 | 0.571429 | 0.580116 | 0.595238 |
| `base_macro_sector_relative_context` | -0.045390 | 0.087189 | 0.571429 | -0.211234 | 0.452381 |
| `base_macro_xsec_dispersion_correlation_regime` | 0.084491 | 0.096198 | 0.523810 | 0.280086 | 0.500000 |
| `base_macro_xsec_sector_gated_context` | 0.183296 | 0.217577 | 0.690476 | 0.581708 | 0.666667 |
| `base_macro_breadth_internal_structure` | 0.089734 | 0.231933 | 0.571429 | 0.589784 | 0.523810 |
| `base_macro_gru` | 0.029190 | 0.009244 | 0.500000 | 0.052006 | 0.404762 |
| `base_macro_hmm` | 0.156669 | 0.177753 | 0.619048 | 0.700588 | 0.571429 |
| `base_macro_exogenous_plus` | 0.003851 | 0.037231 | 0.547619 | 0.062656 | 0.452381 |
| `base` | 0.135909 | 0.165473 | 0.595238 | 0.417878 | 0.571429 |

Permutation results involving vol proxy:

| Comparison | Mean Sharpe left | Mean Sharpe right | Observed diff | p-value | Interpretation |
|---|---:|---:|---:|---:|---|
| `base` vs vol | 1.018800 | 1.154709 | -0.135909 | 0.346500 | No robust evidence. |
| `base_macro` vs vol | 1.215812 | 1.154709 | 0.061103 | 0.696200 | No robust evidence; direction favors `base_macro`. |
| `base_macro_exogenous_plus` vs vol | 1.150858 | 1.154709 | -0.003851 | 0.972200 | Indistinguishable. |
| `base_macro_hmm` vs vol | 0.998040 | 1.154709 | -0.156669 | 0.124000 | Weak evidence only. |
| `base_macro_gru` vs vol | 1.125519 | 1.154709 | -0.029190 | 0.773900 | No robust evidence. |
| `xsec_dispersion_correlation_regime` vs vol | 1.070218 | 1.154709 | -0.084491 | 0.507600 | No robust evidence. |
| `breadth_internal_structure` vs vol | 1.064975 | 1.154709 | -0.089734 | 0.541800 | No robust evidence. |
| `sector_relative_context` vs vol | 1.200099 | 1.154709 | 0.045390 | 0.663300 | No robust evidence. |
| `xsec_sector_gated_context` vs vol | 0.971413 | 1.154709 | -0.183296 | 0.142200 | Weak evidence only. |
| `credit_stress_proxies` vs vol | 1.295586 | 1.154709 | 0.140877 | 0.303000 | No robust evidence; direction favors credit. |
| `rates_term_structure_lsc` vs vol | 1.264703 | 1.154709 | 0.109994 | 0.461700 | No robust evidence; direction favors rates. |
| `analyst_or_fund_revision_features` vs vol | 1.110608 | 1.154709 | -0.044101 | 0.751000 | No robust evidence. |

Critical reading:

- Vol proxy is competitive, but the pairwise evidence does not justify promotion.
- It is practically better than analyst/revision and the first gated family.
- It does not beat rates, credit, sector, or `base_macro` on a defensible statistical basis.

## Selection Rule Effects

| Selection rule | Folds | Selected median test Sharpe | Actual winner median test Sharpe | Match rate | Median regret |
|---|---:|---:|---:|---:|---:|
| `robust_q25_retention` | 14 | 1.210195 | 1.472325 | 0.142857 | 0.372119 |
| `sharpe_only` | 14 | 0.956192 | 1.472325 | 0.142857 | 0.477859 |
| `robust_q25` | 14 | 0.540890 | 1.472325 | 0.071429 | 0.723862 |

Adding vol proxy did not fix the selection layer. `robust_q25_retention` remains weak and its median regret worsened relative to the analyst-merged panel.

Vol proxy was selected in:

- Fold 01 under all three rules, incorrectly; credit was the actual winner.
- Fold 09 under all three rules, correctly, but the actual winning Sharpe was still negative.
- Fold 13 under `robust_q25` and `robust_q25_retention`, incorrectly; sector was the actual winner.

Vol proxy was the actual test winner only in:

- Fold 09.

This is not a strong fold-coverage result. The family is useful as a signal, but it is not robustly selectable.

## Regime Breakdown

| Regime | Runs | Folds | Median days | Median daily return | Median max drawdown | Median hit rate | Median excess return vs benchmark |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bear_high_vol` | 18 | 6 | 4.0 | 0.002577 | -0.013613 | 0.550000 | 0.000250 |
| `bear_low_vol` | 9 | 3 | 1.0 | 0.000274 | 0.000000 | 0.666667 | 0.000067 |
| `bull_high_vol` | 9 | 3 | 1.0 | 0.002484 | 0.000000 | 0.666667 | -0.000089 |
| `bull_low_vol` | 18 | 6 | 3.5 | 0.001401 | -0.002842 | 0.666667 | 0.000057 |
| `unknown` | 42 | 14 | 59.0 | 0.000307 | -0.053399 | 0.533898 | -0.000016 |

The regime table is directionally coherent with the feature hypothesis. Vol proxy has positive median excess return in several labeled regimes, especially `bear_high_vol`. However, the labeled regime windows are short, so this remains diagnostic rather than conclusive.

## Final Assessment

Keep:

- `base_macro` as the primary reference family.
- `base_macro_rates_term_structure_lsc` and `base_macro_credit_stress_proxies` as leading external macro candidates.
- `base_macro_vol_term_or_implied_vol_proxy` as a top-tier diagnostic/risk-state candidate.
- `base_macro_xsec_dispersion_correlation_regime` and `base_macro_sector_relative_context` as useful secondary candidates.

Do not promote:

- `base_macro_vol_term_or_implied_vol_proxy` to primary status.
- `base_macro_analyst_or_fund_revision_features`.
- `base_macro_xsec_sector_gated_context`.

Recommended treatment:

1. Keep vol proxy in the candidate pool for later interaction design.
2. Do not run another standalone vol-only experiment unless the feature construction changes.
3. Treat the empyrical tail-ratio warning as a reporting hygiene issue, not a blocking validity issue.
4. If interaction design is attempted later, combine vol proxy with rates or credit only after all single-family candidates have been merged and fold-level complementarity has been audited.

