# RL for Financial Time-Series Forecasting on the Dow 30

This repository contains a research pipeline for **reinforcement learning in portfolio trading** on **Dow 30** stocks.  
The project is organized as a sequence of research stages, each designed to answer a different question about **out-of-sample robustness**.

The repository currently documents three connected lines of work:

1. **Configuration Comparison**  
   A controlled comparison of PPO configurations across:
   - reward design (**FinRL / Zhang / Custom quadratic utility**),
   - policy network (**FinRL policy / Custom policy**),
   - training stability settings (**early stopping / dropout / multiple seeds**).

2. **Ablation Ladder v1**  
   The first walk-forward baseline focused on feature-family comparison under a more realistic OOS setting.  
   Its main conclusion was that **macro/exogenous context looked more robust than HMM/GRU-style learned features**, while HMM and GRU did not justify becoming the main route to improvement.

3. **Ablation Ladder v2 (current main result)**  
   The current Horizon A stage, built around robust checkpoint/configuration selection.

---

## Research Goal

The main goal is **scientifically credible out-of-sample evaluation** of RL-based portfolio policies.

---

## Data and State Construction

The base universe is **Dow 30**, enriched with additional market and firm-level information.

The preprocessing pipeline includes:
- price-based and technical indicators,
- WRDS-enriched firm-level features,
- macro / market context,
- causal HMM-based regime features,
- GRU-based short-horizon forecasts,
- calendar/event features,
- and final state assembly for RL training.

For the earlier configuration-comparison stage, the state dimensionality was approximately **629 features**.  
For the current ablation ladder, the project uses a controlled feature registry with smaller, interpretable feature sets rather than a single monolithic state.

---

## Experimental Design

### Configuration Comparison stage
A fixed chronological split was used:

- **Train:** 2010-01-01 → 2021-10-01  
- **Validation:** 2021-10-01 → 2022-01-03  
- **Frozen test:** 2022-01-03 → 2023-03-01  

This stage was designed to compare PPO configurations under a single split and identify a stable reference setup.

### Ablation Ladder / Horizon A stage
The later experiments moved to **walk-forward evaluation**, with repeated train / validation / test windows and multiple seeds.

The current reference setup uses:
- **PPO**
- **custom reward**
- **custom MLP policy**
- **robust checkpoint selection**
- **robust configuration selection**
- **multi-seed evaluation**

The feature ladder currently includes:
- `base`
- `base_macro`
- `base_macro_exogenous_plus`
- `base_macro_hmm` *(negative control)*
- `base_macro_gru` *(negative control)*

---

## Main Findings So Far

### 1) Configuration Comparison

The earlier configuration-comparison experiments showed a consistent pattern:

- many RL configurations achieved **strong validation Sharpe**,
- but **none of the tested RL models beat passive benchmarks on the frozen test**,
- all tested RL configurations had **negative mean test Sharpe**,
- so the main bottleneck was **out-of-sample generalization**, not simply reward complexity or network size.

A useful nuance is that the project had **two different “best model” moments**:

- in the **earlier primary comparison**, `zhang_custom` was used as the strongest validation candidate for the dedicated early-stopping / dropout study;
- in the **later broader six-configuration comparison under fixed stable training settings**, `custom_custom` was the strongest RL configuration overall by mean test Sharpe among the six RL variants, although it still did **not** beat passive baselines.

This distinction matters and should be kept explicit to avoid overstating a single “winner”.

### 2) Early Stopping and Dropout

The dedicated training-stability study showed:

- the original early stopping mode was too aggressive,
- **relaxed** early stopping was more reliable,
- **dropout = 0.1** was preferred over no dropout for the custom policy in that setup.

These settings became the reference training configuration for later experiments.

### 3) Ablation Ladder v1

The first walk-forward baseline established the main feature-level direction of the project:

- **macro/exogenous context looked more robust than HMM/GRU-style learned features**,
- HMM and GRU did not provide convincing evidence of stable OOS improvement,
- therefore HMM and GRU are now treated primarily as **negative controls** rather than the main path to improvement.

This shifted the project away from “more latent complexity” and toward:
- stronger walk-forward methodology,
- stability-driven model selection,
- economically motivated feature engineering,
- and cleaner regime-aware diagnostics.

### 4) Ablation Ladder v2 (current main result)

Ablation Ladder v2 is the current main Horizon A result.

Its most important findings are:

- **`base_macro` remained the strongest robust feature set**
- the new **causal calendar/event exogenous layer** (`base_macro_exogenous_plus`) did **not** improve performance beyond the macro baseline
- **`base_macro_gru`** performed better than **`base_macro_hmm`**, but still did not beat `base_macro`
- **HMM remains a valid negative control**
- **GRU does not yet justify becoming the main branch**

This means the central project conclusion is now clearer:

> the most reliable signal currently comes from compact macro context features, while added learned or exogenous complexity has not yet delivered a stronger OOS result.

### 5) Selection Methodology Matters

One of the most important results of Ablation Ladder v2 is methodological rather than architectural.

The project now uses a more conservative selection protocol that prioritizes **stability** rather than **peak validation Sharpe**.

In the current Horizon A stage, robust configuration selection outperformed Sharpe-only selection in fold-level decision quality:
- better alignment with actual test winners,
- higher selected test Sharpe,
- and substantially lower regret.

This is an important step forward for the project, because it suggests that improving **selection protocol** may matter more than adding yet another model component.

---

## Current Interpretation

At this stage, the project does **not** claim that RL has already achieved reliable superiority over passive benchmarks.

Instead, the repository should be read as a transparent research record showing:

- what looked promising on validation,
- what failed on frozen test,
- which feature families are more or less robust,
- and how the methodology is being redesigned to improve generalization.

The strongest current interpretation is:

1. **generalization remains the central challenge**;
2. **macro context is currently the strongest robust signal**;
3. **HMM/GRU-style learned extensions have not justified their added complexity**;
4. **robust selection rules are more promising than peak-Sharpe model selection**;
5. the next gains are more likely to come from **feature science and evaluation design** than from simply making the network larger.

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
