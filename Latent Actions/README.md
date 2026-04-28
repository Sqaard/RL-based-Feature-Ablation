# Latent Actions

This folder is the Phase-2 workspace after the Ablation Ladder v2 Horizon A closeout.

Horizon A remains the source for the frozen PPO training/reporting pipeline. This folder holds the latent-action plan, teacher-action audit tooling, and Phase-2 outputs.

## Current Status

- Horizon A feature-interaction search is closed.
- `base_macro` remains the primary teacher/reference policy.
- Generic SSL/state compression remains deferred.
- The first Phase-2 task is teacher action export and audit, not residual-VQ PPO.

## Files

- `LATENT_ACTIONS_PHASE2_PLAN.md`: active Phase-2 plan.
- `latent_action_phase2_tools.py`: standalone teacher-action audit utility.
- `latent_actions_phase2_base_macro_teacher.yaml`: launch config for the first `base_macro` teacher rerun.
- `dow30_research_support.py`: compatibility copy of the Horizon A support module; the canonical pipeline remains in `..\Ablation Ladder v2`.

## First Experiment

Preflight has been generated in:

`research_outputs_phase2_base_macro_teacher`

It confirms:

- feature set: `base_macro`
- seeds: `42`, `123`, `999`
- folds: `14`
- audit status: ready

Next:

1. Run `research_outputs_phase2_base_macro_teacher/launch_notebook_cell.py` from the notebook/runtime that contains the FinRL environment definitions.
2. Confirm the output has `walk_forward_test_actions.csv`.
3. Run the teacher action audit:

```powershell
& "C:\Users\ivanp\anaconda3\envs\tensorflow\python.exe" ".\Latent Actions\latent_action_phase2_tools.py" `
  --actions ".\Latent Actions\research_outputs_phase2_base_macro_teacher\walk_forward_test_actions.csv" `
  --output-dir ".\Latent Actions\research_outputs_phase2_teacher_action_audit" `
  --teacher-feature-set base_macro
```

## Expected Teacher Output

- `walk_forward_results.csv`
- `walk_forward_daily_test_returns.csv`
- `walk_forward_test_actions.csv`
- `benchmark_suite_daily.csv`
- `artifact_index.json`

## Expected Audit Output

- `latent_action_teacher_matrix.csv`
- `latent_action_teacher_simple_codes.csv`
- `latent_action_teacher_action_summary.csv`
- `latent_action_teacher_code_counts.csv`
- `artifact_index.json`
