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

## Key Results

### 1) Configuration Comparison (single frozen test split)

**Meaning of columns**
- **Val Sharpe**: average validation Sharpe across seeds. Higher is better.
- **Test Sharpe**: average frozen-test Sharpe across seeds. Higher is better.
- **Annual Return**: average test annual return. Higher is better.
- **MDD**: average maximum drawdown on test. Closer to 0 is better.

| Model | Val Sharpe (mean) | Test Sharpe (mean) | Test Annual Return (mean) | Test MDD (mean) |
|---|---:|---:|---:|---:|
| custom_custom | 2.15 | -0.82 | -12.17% | -0.183 |
| finrl_custom | 1.86 | -0.92 | -13.38% | -0.195 |
| zhang_custom | 1.69 | -0.98 | -14.38% | -0.200 |
| zhang_finrl | 2.61 | -1.42 | -1.90% | -0.023 |
| custom_finrl | 2.79 | -1.53 | -6.10% | -0.074 |
| finrl_finrl | 1.90 | -2.06 | -5.16% | -0.060 |
| Buy & Hold (equal-weight) | — | **-0.33** | **-7.50%** | -0.208 |
| DJI baseline | — | -0.42 | -9.40% | -0.219 |

**Takeaway:** validation looked strong, but **none of the RL models beat passive baselines on the frozen test**.

---

### 2) Ablation Ladder v2 (walk-forward, 42 runs per feature set)

**Meaning of columns**
- **Val Sharpe**: median validation Sharpe across walk-forward runs.
- **Test Sharpe**: median test Sharpe across walk-forward runs.
- **Test Return**: median test return across walk-forward runs.
- This stage is more important than the single-split stage because it uses repeated walk-forward evaluation.

| Feature Set | Runs | Val Sharpe (median) | Test Sharpe (median) | Test Return % (median) |
|---|---:|---:|---:|---:|
| **base_macro** | 42 | 2.585 | **1.338** | **3.26%** |
| base_macro_gru | 42 | 2.575 | 1.106 | 2.67% |
| base | 42 | 2.480 | 0.944 | 2.38% |
| base_macro_exogenous_plus | 42 | 2.508 | 0.791 | 2.04% |
| base_macro_hmm | 42 | 2.377 | 0.547 | 1.81% |

**Takeaway:** the best robust feature set in v2 is **base_macro**.  
The new calendar/event exogenous layer did **not** beat the macro baseline, and HMM remained the weakest branch.

---

### 3) Selection Rule Comparison in Ablation Ladder v2

**Meaning of columns**
- **Selected Test Sharpe**: median test Sharpe of the feature sets chosen by the rule.
- **Winner Match Rate**: how often the rule picked the actual test winner for that fold.
- **Regret**: how far the selected model was from the real fold winner. Lower is better.

| Selection Rule | Folds | Selected Test Sharpe (median) | Winner Match Rate | Median Regret |
|---|---:|---:|---:|---:|
| **robust_q25_retention** | 14 | **1.365** | **42.9%** | **0.036** |
| robust_q25 | 14 | 0.871 | 28.6% | 0.234 |
| sharpe_only | 14 | 0.728 | 7.1% | 0.369 |

**Takeaway:** robust selection worked much better than simple Sharpe-only selection.

---

## Short Interpretation

The project currently supports three conclusions:

1. **Generalization is the main challenge**: strong validation results did not carry over to the frozen test in the earlier configuration-comparison stage.
2. **Macro context is the strongest robust signal** in the walk-forward setting.
3. **Selection protocol matters**: robust selection rules were more reliable than Sharpe-only selection.

So the current direction of the project is not “make the network bigger”, but:
- improve walk-forward methodology,
- use stability-driven selection,
- and test feature families that are more economically meaningful and robust out of sample.

## Repository Structure

```text
RL_for_financial_time_series_forecasting/
│
├── README.md
├── Preprocessing/
├── Configuration Comparison/
├── Ablation Ladder v1/
└── Ablation Ladder v2/
