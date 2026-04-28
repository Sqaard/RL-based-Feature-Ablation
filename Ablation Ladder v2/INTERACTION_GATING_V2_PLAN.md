# Interaction/Gating v2 Plan

## Current Stage

The immediate next branch is:

`base_macro_rates_credit_vol_risk_state_context`

This branch implements the first interaction/gating v2 recommendation from `NEXT_CYCLE_FINAL_RANKING.md`. It is not an SSL branch, not a latent-action branch, and not an architecture change.

## Hypothesis

Rates, credit, and implied-volatility proxy signals may be complementary risk-state inputs:

- rates features encode discount-rate and policy pressure,
- credit features encode funding stress and risk appetite,
- vol proxy features encode persistence of fear and implied-volatility stress.

The branch should only survive if this complementarity improves frozen Horizon A out-of-sample evidence versus `base_macro` and the benchmark suite.

## Feature Construction

The implemented candidate family is:

`rates_credit_vol_risk_state_context`

Its feature set is:

`base_macro_rates_credit_vol_risk_state_context`

It includes the existing lag-clean single-family inputs:

- `rates_term_structure_lsc`,
- `credit_stress_proxies`,
- `vol_term_or_implied_vol_proxy`,

plus eight bounded interaction gates:

- `risk_state_rates_credit_stress_gate`,
- `risk_state_rates_vol_stress_gate`,
- `risk_state_credit_vol_stress_gate`,
- `risk_state_curve_inversion_credit_gate`,
- `risk_state_curve_inversion_vol_gate`,
- `risk_state_vol_backwardation_credit_gate`,
- `risk_state_policy_credit_vol_composite`,
- `risk_state_discount_stress_alignment`.

All interaction inputs are derived from already lag-clean columns in `processed_final_fixed_external_lagclean_full.csv`.

## Launch Config

Use:

`configs/next_cycle_candidate_only_rates_credit_vol_risk_state_context.yaml`

Expected dataset:

`processed_final_fixed_external_lagclean_full.csv`

Recommended output directory:

`Ablation Ladder v2/research_outputs_next_cycle_rates_credit_vol_risk_state_context`

Preflight command:

```powershell
python -m dow30_next_cycle_launch preflight-launch `
  --config ..\configs\next_cycle_candidate_only_rates_credit_vol_risk_state_context.yaml `
  --dataset ..\processed_final_fixed_external_lagclean_full.csv `
  --output-dir .\research_outputs_next_cycle_rates_credit_vol_risk_state_context
```

The launch should remain candidate-only and then be merged with the historical and completed next-cycle outputs.

## Kill Rules

Reject this branch if it does not improve:

- median test Sharpe versus `base_macro`,
- benchmark-relative excess Sharpe,
- test evidence rather than validation-only evidence,
- fold-level robustness without relying on one narrow regime,
- selection-rule reliability after the merged rebuild.

## Deferred Branches

The second interaction branch remains deferred until this branch is analyzed:

`base_macro_xsec_sector_complementarity_v2`

SSL/state-compression and latent actions remain Phase-2 future work.
