# Reinforcement Learning for Financial Time-Series Forecasting

This repository is a research record for testing whether a PPO portfolio agent can generalize across market regimes on the Dow 30 universe.

The current conclusion is deliberately conservative: strong validation performance was not enough. After repeated walk-forward ablations, benchmark-relative reporting, and feature-interaction tests, `base_macro` remains the primary reference feature set. The next step is not another Horizon A feature stack; it is Phase-2 latent-action review.

![Horizon A interaction closeout scoreboard](Ablation%20Ladder%20v2/paper_figures/29_horizon_a_interaction_closeout_scoreboard.png)

## Current Status

| Stage | Question | Result |
|---|---|---|
| Configuration Comparison | Which PPO setup is stable enough to study? | PPO with custom reward / custom MLP became the reference training setup, but did not beat passive baselines on the frozen test split. |
| Ablation Ladder v1 | Which feature directions survive a more realistic OOS protocol? | Macro context looked more robust than learned HMM/GRU-style extensions. |
| Ablation Ladder v2 / Horizon A | Can new causal feature families or interaction gates beat `base_macro`? | No promoted replacement. Vol, rates, credit, and xsec/sector families are useful diagnostics, but not robust winners. |
| Next Phase | Where should the project move next? | Latent-action review before generic SSL/state-compression. |

## Final Horizon A Ranking

The final Horizon A bundle merges the historical v2 panel with all next-cycle single-family candidates and both interaction/gating branches:

`Ablation Ladder v2/merged_analysis_history_plus_xsec_breadth_sector_gated_credit_rates_analyst_vol_risk_state_xsec_sector_v2`

| Feature set | Median test Sharpe | Median excess Sharpe vs primary benchmark | Actual winner folds | Decision |
|---|---:|---:|---:|---|
| `base_macro` | 1.3378 | -0.2144 | 2 | Primary reference |
| `base_macro_vol_term_or_implied_vol_proxy` | 1.1405 | -0.2135 | 1 | Retain as diagnostic/top-tier family |
| `base_macro_xsec_sector_complementarity_v2` | 1.1347 | -0.2232 | 0 | Near-miss, do not promote |
| `base_macro_rates_term_structure_lsc` | 1.0974 | -0.2336 | 2 | Retain as diagnostic/top-tier family |
| `base_macro_credit_stress_proxies` | 1.0709 | -0.1903 | 2 | Retain as episodic stress family |
| `base_macro_rates_credit_vol_risk_state_context` | 0.5151 | -0.2919 | 0 | Reject |

No candidate cleared the promotion bar against `base_macro` and the primary benchmark-relative guardrails.

## What The Figures Show

The most important result is not a single ranking table. It is the combination of three diagnostics:

![Benchmark-relative candidate view](Ablation%20Ladder%20v2/paper_figures/26_next_cycle_benchmark_relative_scatter.png)

The benchmark-relative scatter shows why the project avoids overclaiming. Several feature families are useful, but none becomes a clean benchmark-relative OOS winner.

![Selection-rule phase boundary](Ablation%20Ladder%20v2/paper_figures/30_horizon_a_phase_boundary_selection.png)

Selection reliability worsened as the feature panel expanded. This is the main reason Horizon A should close rather than continue adding feature stacks.

![Main candidate cumulative returns](Ablation%20Ladder%20v2/paper_figures/28_next_cycle_main_candidate_cumulative_returns.png)

The cumulative-return view is diagnostic only, but it helps compare the retained single-family candidates and interaction branches on the same aligned test-window basis.

## Research Design

The repository is built around controlled out-of-sample testing rather than leaderboard-style model selection.

- The data universe is Dow 30 equities with technical, macro, WRDS-derived, regime, and candidate feature-family extensions.
- The core model is PPO with a frozen reference setup during Horizon A.
- Evaluation moved from a single train/validation/test split to repeated walk-forward folds and multi-seed runs.
- Reporting includes corrected walk-forward summaries, benchmark-relative metrics, regime diagnostics, selection-rule diagnostics, and pairwise tests.

## Repository Map

| Path | Purpose |
|---|---|
| `Configuration Comparison/` | Early PPO configuration experiments and frozen-test comparison. |
| `Ablation Ladder v1/` | First walk-forward feature-family baseline. |
| `Ablation Ladder v2/` | Current Horizon A package, final merged analysis, closeout docs, and figures. |
| `Ablation Ladder v2/HORIZON_A_CLOSEOUT.md` | Final Horizon A decision record. |
| `Ablation Ladder v2/NEXT_CYCLE_FINAL_RANKING.md` | Detailed candidate ranking before the final interaction closeout. |
| `Ablation Ladder v2/RESEARCH_OUTPUTS_INDEX.md` | Output-folder map and reproducibility notes. |
| `Latent Actions/LATENT_ACTIONS_PHASE2_PLAN.md` | Recommended next research stage. |

## Key Takeaway

This project does not claim that PPO has already achieved reliable superiority over passive benchmarks. Its main contribution is a transparent evaluation path showing what failed, what remained useful, and where the next research intervention should be targeted.

The current decision is:

1. keep `base_macro` as the Horizon A reference;
2. stop the feature-interaction search under the frozen PPO setup;
3. carry vol, rates, credit, and xsec/sector diagnostics forward as context;
4. start Phase-2 latent-action review before generic SSL/state-compression.
