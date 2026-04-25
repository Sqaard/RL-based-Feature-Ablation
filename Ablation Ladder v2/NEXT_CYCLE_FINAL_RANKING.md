# Next-Cycle Final Ranking And Interaction Decision

## Purpose

This report closes the next-cycle Horizon A feature-family round. It summarizes what changed after `NEXT_EXPERIMENT_PLAN.md`, ranks all candidate families against the historical Ablation Ladder v2 baseline, and defines the next interaction/gating v2 shortlist.

## Pre-Plan Baseline

Before the next-cycle experiments, the repository evidence was:

- `base_macro` was the strongest reference family in Ablation Ladder v2.
- `base_macro_exogenous_plus` was a negative result: generic calendar/event context did not beat compact macro context.
- `base_macro_hmm` and `base_macro_gru` were negative controls rather than mainline improvements.
- The v2 contribution was mostly methodological: better selection diagnostics, daily exports, and regime reporting.
- Benchmark-relative interpretation was still not strong enough for final OOS claims.

This is why the next-cycle plan prioritized benchmark hardening and one-at-a-time feature-family experiments instead of architecture changes.

## Implemented Next-Cycle Work

The following next-cycle components were implemented and used:

- Benchmark suite reporting and benchmark-relative summaries.
- Candidate-only launch/preflight workflow.
- Lag-clean external dataset support for rates, credit, vol proxy, and analyst/revision proxy fields.
- Single-family candidate experiments for xsec, breadth, sector, gated xsec/sector, credit, rates, analyst/revision, and vol proxy.
- Final merged analysis with historical v2 outputs plus all next-cycle candidates.
- Final paper figures and plotting tables.

Canonical final merged bundle:

`Ablation Ladder v2/merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol`

## Raw Ranking

| Rank | Feature set | Median test Sharpe | Mean test Sharpe | Median test return pct | Median excess Sharpe vs primary benchmark |
|---:|---|---:|---:|---:|---:|
| 1 | `base_macro` | 1.3378 | 1.2158 | 3.2622 | -0.2144 |
| 2 | `base_macro_vol_term_or_implied_vol_proxy` | 1.1405 | 1.1547 | 1.9797 | -0.2135 |
| 3 | `base_macro_gru` | 1.1058 | 1.1255 | 2.6674 | -0.1587 |
| 4 | `base_macro_rates_term_structure_lsc` | 1.0974 | 1.2647 | 2.3117 | -0.2337 |
| 5 | `base_macro_credit_stress_proxies` | 1.0709 | 1.2956 | 1.0683 | -0.1903 |
| 6 | `base_macro_xsec_dispersion_correlation_regime` | 1.0505 | 1.0702 | 2.6455 | -0.2463 |
| 7 | `base` | 0.9435 | 1.0188 | 2.3844 | -0.2266 |
| 8 | `base_macro_analyst_or_fund_revision_features` | 0.8848 | 1.1106 | 1.9805 | -0.3564 |
| 9 | `base_macro_sector_relative_context` | 0.8503 | 1.2001 | 2.3244 | -0.1661 |
| 10 | `base_macro_exogenous_plus` | 0.7911 | 1.1509 | 2.0444 | -0.2845 |
| 11 | `base_macro_xsec_sector_gated_context` | 0.6342 | 0.9714 | 1.4635 | -0.2856 |
| 12 | `base_macro_hmm` | 0.5466 | 0.9980 | 1.8147 | -0.5346 |
| 13 | `base_macro_breadth_internal_structure` | 0.4827 | 1.0650 | 1.6653 | -0.2119 |

Raw ranking alone is not sufficient for promotion because no candidate clears the benchmark-relative and pairwise evidence bar.

## Decision Ranking

| Tier | Feature families | Decision |
|---|---|---|
| Primary reference | `base_macro` | Keep as frozen reference. |
| Top-tier candidates | `vol_term_or_implied_vol_proxy`, `rates_term_structure_lsc`, `credit_stress_proxies` | Use as primary interaction/gating v2 inputs. |
| Structural candidates | `xsec_dispersion_correlation_regime`, `sector_relative_context` | Keep for complementarity and diagnostics. |
| Low-priority diagnostic | `breadth_internal_structure` | Do not run standalone again; use only as a narrow gate if pre-registered. |
| Do not promote | `analyst_or_fund_revision_features`, `xsec_sector_gated_context`, `exogenous_plus`, `hmm`, `gru` | Exclude from first interaction/gating v2 except as controls. |

## Why `base_macro` Still Wins

`base_macro` remains the best reference because it has the best median test Sharpe and best median test return. Several new families are competitive, but none shows robust pairwise superiority over `base_macro`.

The benchmark-relative view also prevents overclaiming. All families still have negative median excess Sharpe versus the primary equal-weight benchmark. Some candidates improve parts of the benchmark-relative profile, but none clears the passive benchmark bar decisively.

## Candidate Family Conclusions

### `vol_term_or_implied_vol_proxy`

Decision: keep as top-tier diagnostic/risk-state input.

Vol proxy is the best new family by median test Sharpe and has a relatively strong benchmark-relative profile. It should be treated as a risk-state feature rather than a standalone return engine.

### `rates_term_structure_lsc`

Decision: keep as top-tier macro input.

Rates is balanced: strong mean Sharpe, respectable median Sharpe, respectable median return, and clear economic interpretability. It is a better candidate for interaction design than another standalone repeat.

### `credit_stress_proxies`

Decision: keep as top-tier stress input.

Credit has the best mean Sharpe but weak median return. This points to episodic stress-regime value. It is useful for interaction/gating, not for standalone promotion.

### `xsec_dispersion_correlation_regime`

Decision: keep as structural diagnostic input.

Xsec is internally derived and stable. It has strong retention/generalization characteristics, but it did not beat `base_macro`.

### `sector_relative_context`

Decision: keep as episodic structural input.

Sector has weak median Sharpe but useful fold-level wins and high positive-Sharpe frequency. It should be treated as complementary, not as a stable replacement.

### `analyst_or_fund_revision_features`

Decision: do not promote.

Analyst/revision did not improve the main evidence and worsened selection-rule reliability. It can stay as a low-priority diagnostic only after a separate sparsity/staleness audit.

### `xsec_sector_gated_context`

Decision: do not extend mechanically.

The first gated implementation underperformed. Any future xsec/sector interaction must be redesigned and tested as a new hypothesis.

## Selection Layer Result

Selection reliability degraded as the candidate panel expanded.

| Panel | Best rule | Winner match rate | Median regret |
|---|---|---:|---:|
| Historical v2 panel | `robust_q25_retention` | 0.4286 | 0.0363 |
| Final all-candidate panel | `robust_q25_retention` | 0.1429 | 0.3721 |

This is important. The project should not rely on validation-to-test family selection to choose the final candidate. Interaction/gating v2 should be pre-registered, narrow, and evaluated against the frozen `base_macro` reference.

## Interaction/Gating v2 Recommendation

Recommended first branch:

`base_macro_rates_credit_vol_risk_state_context`

Rationale:

- Rates captures policy/discount-rate state.
- Credit captures stress and funding/risk-appetite state.
- Vol proxy captures risk-state and fear persistence.
- These signals are economically coherent and complementary.

Recommended second branch:

`base_macro_xsec_sector_complementarity_v2`

Rationale:

- Xsec is the cleaner stable internal-market structure signal.
- Sector is more episodic but has fold-level winner evidence.
- The prior xsec/sector gate failed, so the second branch must be redesigned rather than copied.

Do not run a broad all-family stack. The next step should be one or two pre-registered interaction designs with strict kill rules.

## Kill Rules For Interaction/Gating v2

Reject an interaction/gating v2 branch if:

- it does not beat `base_macro` on median test Sharpe,
- it does not improve benchmark-relative excess Sharpe,
- it only improves validation and not test,
- its fold wins are concentrated in one narrow regime without improving aggregate robustness,
- or it worsens selection-rule reliability further.

