# Xsec/Sector Complementarity v2 Plan

## Purpose

This is the final planned Horizon A feature-interaction experiment before Phase-2 review.

Candidate family:

`xsec_sector_complementarity_v2`

Feature set:

`base_macro_xsec_sector_complementarity_v2`

The branch is intentionally narrow. It is not a repeat of `xsec_sector_gated_context`, and it is not an architecture, SSL, or latent-action experiment.

## Hypothesis

Cross-sectional dispersion/correlation may identify when sector-relative context is useful.

The v2 design tests complementarity in three ways:

- stock-picking regime strength from dispersion minus correlation,
- sector leadership concentration and stock-vs-sector residual strength,
- mismatch between high correlation pressure and concentrated sector leadership.

## Feature Construction

The branch includes:

- existing `xsec_dispersion_correlation_regime` features,
- existing `sector_relative_context` features,
- eight v2 complementarity features:

| Feature | Intent |
|---|---|
| `xsec_sector_v2_stockpick_regime_strength` | Positive stock-picking regime strength from dispersion minus correlation. |
| `xsec_sector_v2_leadership_concentration` | Absolute distance of sector leadership rank from the middle of the sector pack. |
| `xsec_sector_v2_stockpick_leadership_strength` | Whether stock-picking regimes coincide with strong sector leadership/laggard concentration. |
| `xsec_sector_v2_stockpick_residual_strength` | Whether stock-picking regimes coincide with strong stock-vs-sector residual moves. |
| `xsec_sector_v2_corr_leadership_mismatch` | Penalty-style mismatch when high correlation pressure coexists with concentrated sector leadership. |
| `xsec_sector_v2_sector_stock_confirmation` | Directional confirmation between sector momentum and stock-vs-sector residual context. |
| `xsec_sector_v2_rotation_pressure` | Sector-vs-market strength interacting with stock residual strength versus leadership concentration. |
| `xsec_sector_v2_complementarity_score` | Compact summary of stock-picking complementarity net of correlation mismatch. |

All features are derived causally from lagged/rolling panel data.

## Launch Config

Use:

`configs/next_cycle_candidate_only_xsec_sector_complementarity_v2.yaml`

Recommended dataset:

`processed_final_fixed_external_lagclean_full.csv`

The branch does not require external columns, so `processed_final_fixed.csv` is also acceptable if the notebook environment uses the original processed dataset.

Recommended output directory:

`Ablation Ladder v2/research_outputs_next_cycle_xsec_sector_complementarity_v2`

Preflight command:

```powershell
python -m dow30_next_cycle_launch preflight-launch `
  --config ..\configs\next_cycle_candidate_only_xsec_sector_complementarity_v2.yaml `
  --dataset ..\processed_final_fixed_external_lagclean_full.csv `
  --output-dir .\research_outputs_next_cycle_xsec_sector_complementarity_v2
```

## Kill Rules

Reject this branch unless it improves:

- median test Sharpe versus `base_macro`,
- benchmark-relative excess Sharpe,
- fold-level actual winner coverage,
- test evidence rather than validation-only selection,
- selection-rule reliability after merged reporting.

If this branch fails or is ambiguous, close Horizon A feature-interaction search and move to Phase-2 review. Latent actions are the nearer Phase-2 branch than generic SSL/state compression.
