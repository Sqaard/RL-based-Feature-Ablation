# RL for Financial Time-Series Forecasting

This repository contains a research pipeline for **reinforcement learning in portfolio trading** on **Dow 30** stocks.  
The current version focuses on a **research-clean experimental setup**: feature engineering, market regime detection, reward design, policy architecture comparison, validation/test separation, and benchmark evaluation against **DJI** and **Buy-and-Hold**.

## Project Scope

The project studies whether a trading agent can be improved by combining:

- **technical indicators**
- **WRDS-based market features**
- **GRU-based short-horizon forecasts**
- **market regime features from HMM**
- **custom reward functions**
- **custom policy networks**

The main RL algorithm used in the current stage is **PPO** within the **FinRL** framework.

## What Is Implemented Now

### Data Preprocessing

The dataset is built on **Dow 30** constituents and enriched with additional features from **WRDS**.

The preprocessing pipeline currently includes:

- standard market and technical features
- **GRU forecasts for 1–5 days ahead**
- **macro and regime features**
- **causal HMM-based market regime detection**
- final state construction for RL training

The HMM block is trained only on the training interval and applied in a **past-only** manner to avoid information leakage.

### RL Environments

Several trading environments are compared:

- **FinRL default reward**
- **Zhang reward**
- **custom quadratic-utility reward**

The custom reward is designed to penalize overly aggressive behavior and better reflect risk-sensitive portfolio management.

### Policy Architectures

Two policy families are compared:

- **default FinRL policy**
- **custom policy network** with a larger feature extractor and optional dropout

The custom architecture is used because the state space is high-dimensional, and a very small default network may compress informative structure too aggressively.

### Experimental Protocol

The experiments use a strict time-based split:

- **Train:** `2010-01-01` → `2021-10-01`
- **Validation:** `2021-10-01` → `2022-01-03`
- **Frozen test:** `2022-01-03` → `2023-03-01`

This setup separates:

- model fitting
- model selection
- final out-of-sample evaluation

To improve training stability, the project also studies:

- **early stopping**
- **dropout**
- **multiple random seeds**

## Current Findings

The current results show a clear pattern:

- several RL configurations perform well on **validation**
- however, **none of the tested RL models outperform DJI or Buy-and-Hold on the frozen test**
- this indicates that the main challenge is **out-of-sample generalization**, not just reward design or network size

Additional ablation experiments on the strongest validation model (`zhang_custom`) showed that:

- the original **current early stopping** setting was too aggressive
- a more relaxed stopping rule works better
- **dropout = 0.1** is preferable to no dropout in the tested setup

These settings were then used for the broader comparison across the remaining model configurations.

## Main Outputs

The repository currently contains:

- merged summary tables for all tested configurations
- benchmark comparison against **DJI** and **Buy-and-Hold**
- long-format daily portfolio curves for plotting
- aggregated figures for validation/test comparison

## Tech Stack

- **Python**
- **PyTorch**
- **FinRL**
- **Stable-Baselines3**
- **pandas / NumPy**
- **hmmlearn**
- **yfinance**

## Notes

This repository is at the **research stage**, not production deployment.  
The current focus is on building a reliable experimental framework and understanding why validation improvements do not yet transfer to frozen test performance.

## Next Steps

Planned future extensions include:

- stronger walk-forward validation
- more realistic trading constraints
- improved benchmark design
- better robustness across market regimes
- integration with broker APIs for paper/live trading
- continued work on feature engineering, reward shaping, and model selection

---

This README documents the current research baseline and will be extended as the project evolves.

## Repository Structure

```text
RL_for_financial_time_series_forecasting/
│
├── Data_preprocessing.ipynb
│   Prepares the final dataset:
│   technical indicators, WRDS features, GRU forecasts,
│   HMM market regime variables.
│
├── Experiments.ipynb
│   Main research notebook for training, validation,
│   frozen test evaluation, ablation studies, and model comparison.
│
├── comparison_outputs/
│   Final experiment tables and merged datasets for analysis and plotting.
│   ├── mode1_summary_with_baselines.csv
│   ├── mode1_period_summary_with_baselines.csv
│   └── mode1_long_curves_with_baselines.csv
│
├── paper_figures/
│   Figures used for analysis and paper/report writing.
│   ├── 01_validation_vs_test_sharpe.png
│   ├── 02_validation_vs_test_return.png
│   ├── 03_test_risk_return_scatter.png
│   ├── 04_validation_mean_equity_curves.png
│   ├── 05_test_mean_equity_curves.png
│   ├── 06_test_drawdown_curves.png
│   ├── 07_policy_family_test_sharpe.png
│   ├── 08_reward_family_test_sharpe.png
│   └── aggregated_config_metrics.csv
│
└── README.md

