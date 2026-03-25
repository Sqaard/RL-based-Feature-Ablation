# Configuration Comparison

This folder contains the earlier single-split experiment stage of the project.

## Goal
To compare PPO configurations across:
- reward function,
- policy network,
- early stopping,
- dropout,
- random seed.

## Main experiment blocks

### 1. Reward × Policy comparison
The project compares six PPO configurations:
- `finrl_finrl`
- `finrl_custom`
- `zhang_finrl`
- `zhang_custom`
- `custom_finrl`
- `custom_custom`

### 2. Early stopping / dropout stability study
A focused follow-up study was run to understand whether training stability settings materially affect performance.

Main findings:
- overly aggressive early stopping was harmful,
- `relaxed` early stopping was more reliable,
- `dropout = 0.1` was preferred over no dropout for the custom policy.

## Important interpretation note
There are two different “best model” statements in this stage:

- In the **earlier primary comparison**, `zhang_custom` was used as the strongest validation candidate for the dedicated early stopping / dropout study.
- In the **later broader six-configuration comparison under fixed stable settings**, `custom_custom` became the strongest RL configuration by mean test Sharpe among the six RL variants.

These statements are not contradictory; they refer to different sub-stages of the experiment.

## Bottom-line result
The key conclusion of this stage is negative but important:

- several RL configurations looked strong on validation,
- but none of the tested RL models beat passive benchmarks on the frozen test,
- so the main bottleneck is generalization, not lack of modeling complexity.

## Recommended files for `comparison_outputs/`
Core summary files:
- `mode1_combined_final_results.csv`
- `mode1_combined_grouped_summary.csv`
- `mode1_combined_paper_ready.csv`
- `mode1_combined_artifact_comparison.csv`

Benchmark-aware files:
- `mode1_summary_with_baselines.csv`
- `mode1_period_summary_with_baselines.csv`
- `mode1_long_curves_with_baselines.csv`

Training-stability / ablation files:
- `final_results_table.csv`
- `grouped_summary_table.csv`
- `paper_ready_table.csv`
- `seed_stability_table.csv`
- `artifact_comparison_table.csv`

## Figures
Put all exported paper/report figures in `paper_figures/`.
