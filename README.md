# RL for Financial Time-Series Forecasting on the Dow 30

This repository contains a research pipeline for **reinforcement learning in portfolio trading** on **Dow 30** equities. Its primary objective **scientifically credible out-of-sample evaluation** of PPO-based portfolio policies under regime shift.

## Project Scope

The project is organized into three research stages:

1. **Configuration Comparison**  
   A controlled comparison of PPO configurations across reward functions, policy-network variants, and training-stability settings.

2. **Ablation Ladder v1**  
   The first walk-forward baseline focused on feature-family comparison under a more realistic OOS protocol.

3. **Ablation Ladder v2**  
   The current Horizon A experiment, built around a fixed reference PPO agent, robust model selection, controlled feature ablations, and regime-aware reporting.

## Research Question

The central question of the project is:

> Can a PPO-based trading agent learn a portfolio policy that generalizes across changing market regimes, rather than only fitting a single validation window?

A related question is:

> Which feature families improve robustness out of sample, and which mainly add complexity?

## Data and Features

The base universe is the **Dow 30**, enriched with:
- technical indicators,
- WRDS-based firm-level variables,
- macro / market context,
- HMM-derived regime features,
- GRU-based short-horizon forecasts,
- calendar/event features.

The earlier configuration-comparison stage used a high-dimensional state representation (approximately 629 features).  
The later ablation stages replaced this with a controlled feature ladder of smaller, interpretable feature sets.

## Experimental Design

### Configuration Comparison
A fixed chronological split was used:

- **Train:** 2010-01-01 → 2021-10-01  
- **Validation:** 2021-10-01 → 2022-01-03  
- **Frozen test:** 2022-01-03 → 2023-03-01  

This stage compared PPO configurations under a single split and identified a stable reference setup.

### Ablation Ladder / Horizon A
Later experiments used **walk-forward evaluation** with repeated train / validation / test windows and multi-seed assessment.

The current reference setup uses:
- PPO
- custom reward
- custom MLP policy
- robust checkpoint selection
- robust configuration selection
- controlled feature-family ablations

The current feature ladder includes:
- `base`
- `base_macro`
- `base_macro_exogenous_plus`
- `base_macro_hmm` *(negative control)*
- `base_macro_gru` *(negative control)*

## Main Findings

### Configuration Comparison
The early comparison stage showed that:
- several PPO configurations achieved strong validation Sharpe,
- none of the tested RL models beat passive benchmarks on the frozen test,
- the main bottleneck was therefore **out-of-sample generalization**, not simply reward design or network size.

A useful distinction is that:
- `zhang_custom` was the strongest **validation** candidate in the earlier focused comparison,
- `custom_custom` later became the strongest **RL configuration overall** in the broader six-configuration comparison under the fixed stable setup,
- but even this configuration did **not** outperform passive benchmarks on the frozen test.

### Early Stopping and Dropout
A dedicated stability study showed that:
- the original early stopping mode was too aggressive,
- **relaxed** early stopping was more reliable,
- **dropout = 0.1** was preferred over no dropout for the custom policy in that setting.

These settings became the default stable training setup for later experiments.

### Ablation Ladder v1
The first walk-forward baseline established the main feature-level direction of the project:
- **macro/exogenous context was more robust than HMM/GRU-style learned features**,
- HMM and GRU did not provide convincing evidence of stable OOS improvement,
- both are therefore currently treated as **negative controls** rather than primary solution paths.

### Ablation Ladder v2
The current Horizon A result sharpened this conclusion:
- **`base_macro` remained the strongest robust feature set**,
- the new causal calendar/event layer did **not** improve performance beyond the macro baseline,
- `base_macro_gru` performed better than `base_macro_hmm`, but still did not exceed `base_macro`,
- robust selection rules performed better than Sharpe-only selection in fold-level decision quality.

## Current Interpretation

At this stage, the project does **not** claim that RL has already achieved reliable superiority over passive benchmarks.

Instead, the repository should be read as a transparent research record showing:
- what looked promising on validation,
- what failed on frozen or walk-forward test windows,
- which feature families appear more robust,
- and how the experimental protocol has been redesigned to improve OOS credibility.

The current interpretation is:

1. **generalization remains the central challenge**;
2. **macro context is currently the strongest robust signal**;
3. **HMM/GRU-style learned extensions have not justified their added complexity**;
4. **robust selection protocol matters more than peak validation Sharpe**;
5. the most promising path forward lies in **feature science and evaluation design**, not in immediate architectural escalation.

## Repository Structure

```text
RL_for_financial_time_series_forecasting/
│
├── README.md
├── Preprocessing/
├── Configuration Comparison/
├── Ablation Ladder v1/
└── Ablation Ladder v2/
