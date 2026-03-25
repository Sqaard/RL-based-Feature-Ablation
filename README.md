# RL for Financial Time-Series Forecasting on the Dow 30

This repository contains a research pipeline for **reinforcement learning in portfolio trading** on **Dow 30** stocks.  
The project is organized as a sequence of research stages.

The current repository documents two main lines of work:

1. **Configuration Comparison**  
   A controlled comparison of PPO configurations across:
   - reward design (**FinRL / Zhang / Custom quadratic utility**),
   - policy network (**FinRL policy / Custom policy**),
   - training stability settings (**early stopping / dropout / multiple seeds**).

2. **Ablation Ladder / Horizon A**  
   A later experiment series focused on **out-of-sample robustness**, **walk-forward evaluation**, and **feature-family comparison**, where macro/exogenous features emerged as more robust than HMM/GRU-style learned features.

---

## Research Goal

The main goal is not deployment, but **scientifically credible out-of-sample evaluation** of RL-based portfolio policies.

The core question is:

> Can a PPO-based trading agent learn a portfolio policy that generalizes across changing market regimes, rather than only fitting a single validation window?

---

## Data and State Construction

The base universe is **Dow 30**, enriched with additional market and firm-level information.

The preprocessing pipeline includes:
- price-based and technical indicators,
- WRDS-enriched features,
- macro / market context,
- causal HMM-based regime features,
- GRU-based short-horizon forecasts,
- state assembly for RL training.

For the earlier configuration-comparison stage, the state dimensionality was approximately **629 features**.

---

## Experimental Design

### Fixed chronological split used in the configuration-comparison stage
- **Train:** 2010-01-01 → 2021-10-01
- **Validation:** 2021-10-01 → 2022-01-03
- **Frozen test:** 2022-01-03 → 2023-03-01

### RL algorithm
- **PPO**
- Implemented with **FinRL / Stable-Baselines3**

### Compared modeling choices
- **Reward functions:** FinRL, Zhang, Custom quadratic utility
- **Policy networks:** FinRL policy, Custom policy
- **Training stability options:** early stopping, dropout, multiple seeds

---

## Main Findings So Far

### 1) Configuration Comparison
The earlier configuration-comparison experiments showed a consistent pattern:

- many RL configurations achieved **strong validation Sharpe**,
- but **none of the tested RL models beat passive benchmarks on the frozen test**,
- all tested RL configurations had **negative mean test Sharpe**,
- therefore the main bottleneck is **out-of-sample generalization**, not simply reward complexity or network size.

A useful nuance is that the project had **two different “best model” moments**:
- in the **earlier primary comparison**, `zhang_custom` was used as the strongest validation candidate for the dedicated early-stopping / dropout study;
- in the **later broader six-configuration comparison under fixed stable training settings**, `custom_custom` was the strongest RL configuration overall by mean test Sharpe among the six RL variants, although it still did **not** beat passive baselines.

This distinction matters and should be reflected in the text to avoid overstating a single “winner”.

### 2) Early Stopping and Dropout
The dedicated training-stability study showed:

- the original **current** early stopping mode was too aggressive,
- **relaxed** early stopping was more reliable,
- **dropout = 0.1** was preferred over no dropout for the custom policy in that setup.

These settings were then used as the default stable configuration for the broader comparison stage.

### 3) Horizon A / Walk-Forward Legacy Baseline
In the later walk-forward baseline:

- the corrected report had **300 raw rows**, **210 unique run-level rows**, and **90 regime-expanded rows**,
- the central finding was that **macro/exogenous features were more robust than HMM/GRU-style learned features**,
- HMM and GRU are therefore currently treated as **negative controls** rather than the main route to OOS improvement,

This means the project’s current direction is shifting away from “more complex latent features” and toward:
- stronger walk-forward methodology,
- stability-driven model selection,
- exogenous / economically motivated feature engineering,
- cleaner regime-aware diagnostics.

---

## Repository Structure

```text
RL_for_financial_time_series_forecasting/
│
├── README.md
│   Root project overview and research roadmap.
│
├── Preprocessing/
│   Data construction and feature engineering.
│   ├── Data_preprocessing.ipynb
│   ├── dow_30_fundamental_wrds.csv
│   └── README.md
│
├── Configuration Comparison/
│   Earlier single-split experiment stage:
│   reward × policy comparison + early stopping / dropout study.
│   ├── Experiments_Config_Comparison.ipynb
│   ├── comparison_outputs/
│   ├── paper_figures/
│   └── README.md
│
├── Ablation Ladder v1/
│   Previous walk-forward / Horizon A baseline.
│   ├── Experiments_Ablation_Ladder_v1.ipynb
│   ├── comparison_outputs/
│   ├── paper_figures/
│   └── README.md
│
└── Ablation Ladder v2/
    Current walk-forward experiment stage.
    ├── Experiments_Ablation_Ladder_v2.ipynb
    ├── comparison_outputs/
    ├── paper_figures/
    └── README.md
```

---

## Current Research Direction

The most important current research question is:

> How can the project achieve more reliable out-of-sample performance under regime shift?

Current priorities:
1. stronger walk-forward evaluation,
2. stability-driven model selection,
3. cleaner regime diagnostics,
4. exogenous / macro-style feature engineering,
5. controlled ablations against HMM/GRU negative controls.

---

## Tech Stack

- Python
- PyTorch
- FinRL
- Stable-Baselines3
- pandas / NumPy
- hmmlearn
- yfinance

---

## Status

This repository is in the **research stage**.  
It should be read as a transparent record of experimental progress:
- what looked promising on validation,
- what failed on frozen test,
- and how the methodology is being redesigned to improve robustness.
