# Ablation Ladder v2

This folder contains the current walk-forward experiment stage.

## Goal
To extend the v1 baseline with:
- stability-driven selection,
- improved reporting at unique run level,
- daily test export,
- exogenous regime diagnostics,
- controlled feature ladder comparisons.

## Expected structure
- `Experiments_Ablation_Ladder_v2.ipynb`
- `comparison_outputs/`
- `paper_figures/`

## Intended interpretation
This stage should answer a stricter question than v1:

> Which feature families and selection rules remain the most robust across walk-forward windows and regimes?

The comparison to v1 should focus on:
- run-level robustness,
- selection stability,
- fold-level consistency,
- regime-aware behavior.
