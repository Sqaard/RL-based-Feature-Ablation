# Xsec/Sector Complementarity v2 Results

## Branch

Candidate family:

`xsec_sector_complementarity_v2`

Feature set:

`base_macro_xsec_sector_complementarity_v2`

Final merged bundle:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2`

Critical report:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2/critical_xsec_sector_complementarity_v2_report.md`

## Decision

Do not promote over `base_macro`.

The branch is the strongest interaction/gating design tested, but it does not pass the pre-registered promotion bar.

Key metrics:

- Rank by median test Sharpe: `3 / 15`.
- Median test Sharpe: `1.1347`.
- Mean test Sharpe: `1.1643`.
- Median test return pct: `2.7852`.
- Median excess Sharpe versus primary benchmark: `-0.2232`.
- Actual fold winner count: `0`.

Compared with `base_macro`:

- Mean Sharpe delta: `-0.0516`.
- Median Sharpe delta: `-0.1454`.
- Sharpe win rate: `0.4286`.
- Permutation p-value: `0.7235`.

## Interpretation

This is a near-miss, not a winner.

The branch materially improved on the failed `xsec_sector_gated_context` design, but it still did not beat the frozen `base_macro` reference or the primary benchmark-relative bar. It was selected by validation rules in several folds, but those selections were all wrong out of sample.

## Phase Boundary

Horizon A feature-interaction search should close here.

Recommended next step:

`LATENT_ACTIONS_FUTURE_WORK.md`

Start only with the staged latent-action baseline:

1. simple discretization baseline,
2. teacher action dataset,
3. tokenizer / behavior-cloning warm start,
4. PPO fine-tuning,
5. only then residual or hierarchical VQ variants.

SSL/state-compression should remain deferred unless a separate Phase-2 review shows that state representation, rather than action regularization and selection robustness, is the limiting factor.
