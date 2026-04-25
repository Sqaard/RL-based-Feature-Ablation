# Horizon A Research Cycle

This repo now supports a stricter Horizon A research loop for the Dow 30 RL project without rewriting the core PPO training pipeline.

## What Is Frozen As The Reference Pipeline

The reference pipeline is defined in `dow30_horizon_a.py` and exported into each research output directory as `reference_experiment_config.json`.

The frozen baseline keeps the existing project choices intact:

- Agent: current `DRLAgent` + PPO
- Reward: current environment reward chosen by the reference config
- Policy: current project policy choice chosen by the reference config
- Frictions: existing transaction cost setup
- Action constraints: existing long-only bounded action logic
- Walk-forward schedule: the current project protocol

For the current cycle, all ablations are expected to vary only:

- feature set
- configuration-level selection rule
- regime diagnostics and reporting

The reference base config defaults to `custom_custom`, but the notebook runner can still freeze any of the existing config names (`finrl_finrl`, `finrl_custom`, `zhang_finrl`, `zhang_custom`, `custom_finrl`, `custom_custom`) as long as it is used consistently across the whole cycle.

## Why Selection Is No Longer Sharpe-Only

Validation peak Sharpe on one window is too fragile for this project. The new reporting layer therefore compares multiple configuration-level selection rules:

- `sharpe_only`
- `robust_q25`
- `robust_q25_retention`

The default robust rule is:

```text
robust_selection_score =
    q25(validation_sharpe)
    + lambda_retention * median(retention_ratio)
    + lambda_generalization * median(generalization_ratio)
    - lambda_turnover * median(validation_turnover)
    - lambda_mdd * abs(median(validation_max_drawdown))
```

This is used for configuration-level comparison across fold/seed panels. The existing checkpoint-level selection inside a single run stays minimally invasive and is now logged honestly as `checkpoint_robust_score`.

## Controlled Ablation Ladder

The controlled ladder is defined in `dow30_horizon_a.py`:

1. `base`
2. `base_macro`
3. `base_macro_exogenous_plus`
4. `base_macro_hmm`
5. `base_macro_gru`

Metadata is attached to every feature set:

- `feature_family`
- `is_negative_control`
- `feature_set_description`

`base_macro_hmm` and `base_macro_gru` are explicitly marked as negative controls.

Legacy artifacts from the previous cycle (`base_macro_hmm_gru`, `full`) are still understood by the rebuild/report layer so older CSVs remain analyzable.

## New Exogenous Feature Family

The new exogenous layer is a causal calendar/event layer. It is implemented in `dow30_horizon_a.py` and auto-added by the research runners.

Included features:

- trading-day cyclical encoding
- month start / month end
- quarter start / quarter end
- turn-of-month flag
- year start / year end
- option-expiration proxies derived from the calendar only
- trading-day-of-month / quarter indices
- days-left-in-month / quarter

Leakage constraints:

- all features are derived only from the current trading date
- no future prices or future macro publications are used
- no web fetch or external event calendar is used

## Exogenous Regime Diagnostics

The new regime diagnostics do not depend on HMM states.

Regimes are built causally from:

- trailing market trend
- trailing realized volatility

The labels are:

- `bull_low_vol`
- `bull_high_vol`
- `bear_low_vol`
- `bear_high_vol`
- `unknown` when there is not enough trailing history

The daily test export is the source of truth for regime reporting. When it is present, the pipeline writes:

- `walk_forward_daily_test_returns.csv`
- `regime_run_level_metrics.csv`
- `regime_summary_by_feature_set.csv`
- `regime_summary_by_fold.csv`

When only legacy CSVs are available, the rebuild layer will still fix run-level reporting, but it will warn that exogenous regime diagnostics cannot be rebuilt without daily test rows.

## Rebuilt Reporting

The new reporting layer fixes a previous methodological bug: summary tables are now rebuilt from unique `run_key` rows only.

Key outputs:

- `unique_run_level_results.csv`
- `corrected_walk_forward_summary.csv`
- `selection_rule_comparison.csv`
- `selection_rule_summary.csv`
- `validation_vs_test_winner_by_fold.csv`
- `pairwise_permutation_tests_recomputed.csv`
- `artifact_index.json`

`artifact_index.json` logs:

- raw row count
- unique run key count
- whether expansion was detected
- warnings about missing daily files

## How To Run From The Notebook

The repo now includes a separate minimal runner notebook:

- `Experiments_Ablation_Ladder_v2_Horizon_A_Run.ipynb`

The existing `Experiments_fixed_chatgpt.ipynb` flow stays valid. Run the notebook up to the research runner cell and then execute the walk-forward cell again after reloading the updated modules:

```python
import importlib
import dow30_notebook_research_runner

importlib.reload(dow30_notebook_research_runner)
from dow30_notebook_research_runner import build_notebook_research_runner

research_runner = build_notebook_research_runner(globals())

research_bundle = research_runner(
    df=processed.copy(),
    base_config_name="custom_custom",
    output_dir="./research_outputs_horizon_a",
    seeds=(42, 123, 999),
    total_timesteps=200000,
    max_folds=None,
    es_mode="relaxed",
    dropout_p=0.1,
    eval_freq=8192,
    checkpoint_freq=4096,
    verbose=0,
)
```

This produces the corrected run-level outputs directly from the notebook-bound training entrypoint.

## Exact A1 Next-Cycle Panel

The first implemented next-cycle candidate was:

- `xsec_dispersion_correlation_regime`

Its exact A1 comparison panel is:

- `base_macro`
- `base_macro_hmm`
- `base_macro_gru`
- `base_macro_xsec_dispersion_correlation_regime`

The machine-readable launch template lives at:

- `..\configs\next_cycle_a1_xsec_dispersion_correlation.yaml`

Before the first real run, execute the launch preflight:

```powershell
C:\Users\ivanp\anaconda3\envs\tensorflow\python.exe -m dow30_next_cycle_launch preflight-launch `
  --config ..\configs\next_cycle_a1_xsec_dispersion_correlation.yaml `
  --dataset ..\processed_final_fixed.csv `
  --output-dir .\research_outputs_next_cycle_a1_xsec
```

This writes:

- `launch_preflight_report.json`
- `launch_notebook_cell.py`
- `launch_kwargs.json`
- `launch_config_snapshot.yaml`
- `post_run_rebuild_commands.json`

Use `launch_notebook_cell.py` as the exact config-driven notebook launch cell for the first A1 run.

If the processed dataset is already loaded in the notebook as `processed` and no standalone CSV is available in the workspace, run the equivalent in-memory preflight:

```python
from dow30_next_cycle_launch import run_launch_preflight_from_dataframe

preflight = run_launch_preflight_from_dataframe(
    processed.copy(),
    config_path=r"..\configs\next_cycle_a1_xsec_dispersion_correlation.yaml",
    output_dir=r".\research_outputs_next_cycle_a1_xsec",
)
```

This writes the same launch artifacts without requiring `processed_final_fixed.csv` to exist on disk.
It also writes `processed_dataset_snapshot.csv` into the chosen output directory so the later benchmark rebuild step remains reproducible.

Direct notebook launch example:

```python
from dow30_next_cycle_launch import run_bootstrapped_notebook_launch_from_csv

launch_bundle = run_bootstrapped_notebook_launch_from_csv(
    config_path=r"..\configs\next_cycle_a1_xsec_dispersion_correlation.yaml",
    dataset_path=r"..\processed_final_fixed.csv",
    output_dir=r".\research_outputs_next_cycle_a1_xsec",
)

preflight = launch_bundle["preflight"]
research_bundle = launch_bundle["research_bundle"]
```

This keeps the reference PPO setup frozen while changing only the candidate feature family under test.
It bootstraps the required notebook runtime from the reference experiment notebook, reruns launch preflight, snapshots the processed dataset into the output directory, and then starts the notebook-bound training run from the YAML config.

For the first direct run, `post_run_rebuild_commands.json` is mainly a reproducibility fallback for later merges or rebuilds.
The notebook runner already writes the benchmark suite, enriched summary, and statistical credibility outputs directly into the launch output directory.

## Candidate-Only Cloud Workflow

If the baseline families already exist from prior v1/v2 research outputs and you only want to run a new family on a separate cloud notebook, use:

- `..\configs\next_cycle_candidate_only.yaml`
- `.\Next_Cycle_Candidate_Family_Launch.ipynb`

Implemented candidate families currently available for one-family cloud runs:

- `selected_candidate_family = "rates_term_structure_lsc"`
- `selected_candidate_family = "credit_stress_proxies"`
- `selected_candidate_family = "xsec_dispersion_correlation_regime"`
- `selected_candidate_family = "breadth_internal_structure"`
- `selected_candidate_family = "sector_relative_context"`
- `selected_candidate_family = "xsec_sector_gated_context"`
- `selected_candidate_family = "vol_term_or_implied_vol_proxy"`
- `selected_candidate_family = "analyst_or_fund_revision_features"`

For convenience, dedicated one-cell notebooks are also available:

- `.\Next_Cycle_Rates_Term_Structure_LSC_Launch.ipynb`
- `.\Next_Cycle_Credit_Stress_Proxies_Launch.ipynb`
- `.\Next_Cycle_Breadth_Internal_Structure_Launch.ipynb`
- `.\Next_Cycle_Sector_Relative_Context_Launch.ipynb`
- `.\Next_Cycle_XSec_Sector_Gated_Context_Launch.ipynb`
- `.\Next_Cycle_Vol_Term_Or_Implied_Vol_Proxy_Launch.ipynb`
- `.\Next_Cycle_Analyst_Or_Fund_Revision_Features_Launch.ipynb`

Their matching configs are:

- `..\configs\next_cycle_candidate_only_rates_term_structure_lsc.yaml`
- `..\configs\next_cycle_candidate_only_credit_stress_proxies.yaml`
- `..\configs\next_cycle_candidate_only_breadth_internal_structure.yaml`
- `..\configs\next_cycle_candidate_only_sector_relative_context.yaml`
- `..\configs\next_cycle_candidate_only_xsec_sector_gated_context.yaml`
- `..\configs\next_cycle_candidate_only_vol_term_or_implied_vol_proxy.yaml`
- `..\configs\next_cycle_candidate_only_analyst_or_fund_revision_features.yaml`

External macro and revision-proxy candidate runs should use:

- `..\processed_final_fixed_external_lagclean_full.csv`

This dataset is built reproducibly by:

```powershell
python ..\Preprocessing\build_external_macro_dataset.py `
  --input ..\processed_final_fixed_external.csv `
  --output ..\processed_final_fixed_external_lagclean.csv `
  --audit ..\processed_final_fixed_external_lagclean_audit.json

python ..\Preprocessing\build_revision_proxy_dataset.py `
  --input ..\processed_final_fixed_external_lagclean.csv `
  --output ..\processed_final_fixed_external_lagclean_full.csv `
  --audit ..\processed_final_fixed_external_lagclean_full_revision_proxy_audit.json
```

`analyst_or_fund_revision_features` is a diagnostic proxy built from point-in-time fundamentals. It is not a true analyst estimate revision feed.

Each candidate-only run should keep:

- `panel_scope = "candidate_only"`

This runs only the selected candidate feature set instead of rerunning `base`, `base_macro`, `base_macro_hmm`, or `base_macro_gru`.
The intended workflow is:

1. Reuse the existing baseline outputs as the historical reference panel.
2. Run one new candidate family per notebook or worker.
3. Merge the resulting output directories into a unified rebuilt analysis folder.

After the standalone `xsec`, `breadth`, and `sector` runs, the recommended diagnostic follow-up is:

- `xsec_sector_gated_context`

This candidate includes the original causal `xsec` and `sector_relative` columns plus bounded interaction gates. It is intended to test whether the more stable xsec regime signal can condition the episodic sector-relative signal without changing the reference PPO setup.

## Merge New Seeds And Rebuild Reports

Merge multiple raw result files:

```powershell
python -m dow30_reporting merge-walkforward-results `
  --inputs .\seed_batch_a\walk_forward_results.csv .\seed_batch_b\walk_forward_results.csv `
  --output .\merged\walk_forward_results_merged.csv
```

Rebuild corrected reports from merged raw results:

```powershell
python -m dow30_reporting rebuild-walkforward-report `
  --input .\merged\walk_forward_results_merged.csv `
  --outdir .\merged\analysis
```

If you also merged the daily test export, pass it too:

```powershell
python -m dow30_reporting rebuild-walkforward-report `
  --input .\merged\walk_forward_results_merged.csv `
  --daily-input .\merged\walk_forward_daily_test_returns.csv `
  --outdir .\merged\analysis
```

To rebuild the benchmark suite from the processed dataset and the saved fold schedule:

```powershell
python -m dow30_reporting build-benchmark-suite `
  --dataset ..\processed_final_fixed.csv `
  --folds-input .\merged\walk_forward_folds.csv `
  --output .\merged\benchmark_suite_daily.csv
```

Then include the benchmark suite in the rebuild step:

```powershell
python -m dow30_reporting rebuild-walkforward-report `
  --input .\merged\walk_forward_results_merged.csv `
  --daily-input .\merged\walk_forward_daily_test_returns.csv `
  --benchmark-suite-input .\merged\benchmark_suite_daily.csv `
  --outdir .\merged\analysis
```

To merge whole research output directories from separate notebooks and rebuild a unified report in one step:

```powershell
python -m dow30_reporting merge-research-outputs `
  --inputs .\baseline_reference_outputs .\research_outputs_next_cycle_xsec_dispersion_correlation_regime `
  --output-dir .\merged_candidate_cycle `
  --dataset ..\processed_final_fixed.csv
```

This command merges `walk_forward_results.csv`, available daily test exports, available fold schedules, and benchmark suites when present.
If no benchmark suite is available in the input directories, it rebuilds one from `processed_final_fixed.csv` plus the merged fold schedule before rebuilding the final analysis folder.

## Next-Cycle Reporting Status

The current Horizon A runtime now implements the following next-cycle reporting support:

- multi-benchmark evaluation through `benchmark_suite_daily.csv`,
- benchmark-relative run-level and feature-level summaries,
- an enriched corrected summary with primary-benchmark fields,
- a `statistical_credibility_report.json` scaffold that records implemented guardrails and explicit advanced-statistics TODO items.

The main benchmark suite currently includes:

- `dow30_equal_weight_rebalance_matched`,
- `dow30_market_proxy_buy_hold`,
- `dow30_equal_weight_vol_target`,
- `dow30_trend_filter_overlay`.

Outputs written when benchmark suite data are available include:

- `benchmark_suite_daily.csv`,
- `benchmark_run_level_metrics.csv`,
- `benchmark_summary_by_feature_set.csv`,
- `benchmark_summary_by_fold.csv`,
- `corrected_walk_forward_summary_with_primary_benchmark.csv`,
- `statistical_credibility_report.json`.

## Still Deferred / Not Yet Implemented

The following items remain intentionally deferred even after the reporting and panel-derived candidate upgrades:

- stronger multiple-testing / backtest-overfitting safeguards such as Deflated-Sharpe-style or PBO-style reporting,
- external-data next-cycle candidate families such as rates term structure, credit stress, volatility term structure, and analyst/fund revision features.

Important interpretation rules for these next-cycle items:

- external-data planning candidates are placeholders only and are **not** active members of `build_controlled_feature_registry`,
- they should not be described as available training features until causal data plumbing is implemented,
- future benchmark hardening should use the same walk-forward windows and cost conventions as the agent,
- the core comparable seed set for required next-cycle experiments should stay `(42, 123, 999)`, with any broader seed expansion treated as a later stability extension.

## Tests

Minimal unit tests live in:

- `tests/test_horizon_a_benchmarks.py`
- `tests/test_next_cycle_features.py`
- `tests/test_next_cycle_launch.py`
- `tests/test_reference_notebook_runtime.py`
- `tests/test_reporting_merge.py`

Run them in the working conda env:

```powershell
C:\Users\ivanp\anaconda3\envs\tensorflow\python.exe -m unittest tests.test_horizon_a_benchmarks tests.test_next_cycle_features tests.test_next_cycle_launch tests.test_reference_notebook_runtime tests.test_reporting_merge
```

## Explicit Non-Goals For This Cycle

The following are intentionally not implemented in Horizon A:

- news or LLM signals
- heavy architecture search
- representation learning / SSL branch
- a full standalone CLI training refactor away from the notebook entrypoint

The training entrypoint remains notebook-bound on purpose to avoid risky changes to the existing PPO pipeline.
