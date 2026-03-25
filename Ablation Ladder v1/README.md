# Ablation Ladder v1

This folder contains the previous walk-forward / Horizon A baseline.

## Goal
To move beyond a single validation/test split and evaluate feature families under a more robust walk-forward setup.

## What this experiment established
This legacy baseline led to an important shift in the project:

- macro / exogenous features were more robust than HMM/GRU-style learned features,
- HMM and GRU are therefore currently treated as negative controls,
- the project focus moved from “more latent complexity” to “better OOS robustness and methodology”.
