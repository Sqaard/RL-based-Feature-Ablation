# Interaction/Gating v2 Results

## First Branch Tested

Tested branch:

`base_macro_rates_credit_vol_risk_state_context`

Candidate family:

`rates_credit_vol_risk_state_context`

Merged analysis bundle:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state`

Critical report:

`merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state/critical_rates_credit_vol_risk_state_context_report.md`

## Decision

Reject the first interaction/gating v2 branch.

It ranked 13th of 14 feature sets by median test Sharpe:

- Median test Sharpe: `0.5151`.
- Mean test Sharpe: `0.9524`.
- Median test return pct: `0.8039`.
- Median excess Sharpe versus primary benchmark: `-0.2919`.
- Actual fold winner count: `0`.

The branch failed the pre-registered kill rules and should not be promoted, extended to five seeds, or used as a reason to start SSL/state-compression or latent-action PPO.

## Interpretation

The negative result is specific to this rates/credit/vol risk-state stack. It shows that the retained single-family macro/risk signals do not combine mechanically into a better policy input under the frozen Horizon A PPO setup.

`base_macro` remains the reference family. The next possible interaction experiment, if any, should be the already planned but redesigned structural branch:

`base_macro_xsec_sector_complementarity_v2`

That branch should stay narrow and pre-registered because the previous `xsec_sector_gated_context` branch also underperformed.

## Final Branch Result

The final Horizon A feature-interaction branch has now been run:

- candidate family: `xsec_sector_complementarity_v2`,
- feature set: `base_macro_xsec_sector_complementarity_v2`,
- merged bundle: `merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2`,
- critical report: `merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2/critical_xsec_sector_complementarity_v2_report.md`.

Decision: do not promote over `base_macro`.

It ranked third of fifteen feature sets by median test Sharpe, but it did not beat `base_macro`, did not improve primary benchmark-relative excess Sharpe, and was never the actual fold-level test winner.

Horizon A feature-interaction search should close here. Move to Phase-2 review, with latent actions ahead of generic SSL/state compression.
