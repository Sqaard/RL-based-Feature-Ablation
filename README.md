# RL for Financial Time-Series Forecasting

## Project Status

**Closed as a negative-result trading-improvement branch.** The best stable reference remains **`base_macro`**. Feature expansion, stress reward reweighting, action penalties, latent/action audits, and controlled hidden-state primitive interventions did **not** produce a reliable OOS trading improvement.

**Still useful as feature search.** This branch built a feature-search and ablation map: it showed which market, macro, cross-sectional, robustness, and action-side ideas did not reliably improve OOS performance. The likely bottleneck is the **RL model structure/stability**, so the next project is [CHRL-Constrained-Hierarchical-Reinforcement-Learning](https://github.com/Sqaard/CHRL-Constrained-Hierarchical-Reinforcement-Learning): a more stable, constrained, and interpretable successor.

---

Goal: build a Dow 30 trading agent that is both reliable under regime shift and interpretable enough to audit.

The current result is conservative: `base_macro` is still the best reference policy. Feature expansion, stress reweighting, action penalties, and controlled hidden-state primitive interventions did not produce a robust OOS improvement.

![Research ladder](docs/assets/01_research_ladder.png)

This is the project path: baseline PPO, feature ablation, robustness failures, then behavior primitives.

## Experiments As Questions

| Question | Experiment | Answer |
|---|---|---|
| Which PPO setup is worth studying? | Baseline PPO comparison | Custom reward + custom MLP became the working reference, but passive baselines stayed hard to beat. |
| Do more feature families improve OOS reliability? | Feature ablation | No. `base_macro` remained the strongest reference after Horizon A. |
| Does PPO just underweight stress days? | G1 stress reweighting | No reliable OOS improvement. |
| Does PPO just trade too aggressively? | G2 action penalties | No. Lower turnover alone did not improve benchmark-relative quality. |
| Are bad primitives causal handles? | Controlled primitive intervention | No safe control-beating intervention survived the gates. |
| What now? | Model stability | Move from fragile flat PPO toward constrained hierarchical RL. |

## Current Evidence

![Feature ablation scoreboard](docs/assets/02_feature_ablation_scoreboard.png)

The best feature candidates were useful diagnostics, but none replaced `base_macro`.

| Stage | Best/Key result | Decision |
|---|---|---|
| Baseline PPO | RL configuration stabilized | Keep as research baseline |
| Feature ablation | `base_macro` median test Sharpe `1.3378` | Keep as reference |
| G1 domain robustness | stress reward weighting mixed/noisy | Fail |
| G2 conservative actions | one-seed screen found no pass | Screening fail |
| Behavior primitives | 6 primitives, several failure candidates | Descriptive audit useful |
| Causal primitive intervention | `failed_or_artifact`, 0 rollout candidates | Do not promote to PPO intervention |

The failed robustness branches narrowed the problem: the issue is probably not one simple feature family, stress weighting rule, turnover penalty, or SSL intervention type.

## Repository Map

| Path | What to read |
|---|---|
| `experiments/baseline-ppo/` | First PPO setup comparison. |
| `experiments/feature-ablation/` | Horizon A feature-family search and final ranking. |
| `experiments/domain-robustness/` | G1/G2 robustness tests: both failed. |
| `experiments/action-audit/` | Latent-action offline diagnostics. |
| `experiments/behavior-primitives/` | Current interpretability-first audit. |
| `src/` | Reproducible research utilities. |
| `configs/` | Launch configs grouped by experiment stage. |

## Bottom Line

This repo does not claim a finished trading edge. The failure may come from three places: states/features, the RL model, or the self-supervised/intervention methodology. The most likely bottleneck is now the RL model itself: training showed severe instability in some runs (`approx_kl > 10^3`). The next project, [CHRL-Constrained-Hierarchical-Reinforcement-Learning](https://github.com/Sqaard/CHRL-Constrained-Hierarchical-Reinforcement-Learning), is designed to address that with a more stable and interpretable control structure.
