# Latent Actions Phase-2 Plan

Status: active Phase-2 entry after Horizon A closeout.

This is the next branch after `HORIZON_A_CLOSEOUT.md`. It is not SSL/state compression and it is not a new feature-family search.

## Objective

Test whether action-space modeling improves frozen Dow 30 PPO out-of-sample quality more reliably than adding more observation features.

The first question is deliberately narrow:

Can a simple, auditable action discretization or tokenizer improve policy stability and benchmark-relative OOS performance without hiding behind lower turnover alone?

## Entry Conditions

Satisfied:

- Horizon A feature-interaction search is closed.
- `base_macro` remains the primary reference.
- No interaction/gating branch beat the frozen reference or benchmark-relative bar.
- Selection-rule reliability worsened as the feature panel expanded.

New implementation now available:

- Notebook runner exports `walk_forward_test_actions.csv` when `evaluate_model_on_env` returns `df_action`.
- Merge/rebuild pipeline preserves action traces as `walk_forward_test_actions_merged.csv` and `analysis/walk_forward_test_actions.csv`.
- `latent_action_phase2_tools.py` builds teacher-action matrices and simple action-code diagnostics.

## Immediate Experiment

Do not jump to residual VQ PPO.

Run the Phase-2 action audit first:

1. Rerun the teacher policy with action export enabled.
2. Use `base_macro` as the first teacher.
3. Build the teacher action matrix and simple action-code counts.
4. Inspect whether the policy's action space has stable, repeated structure across folds and seeds.
5. Only if the action codes are not degenerate, move to tokenizer or behavior cloning.

Suggested command after a rerun produces `walk_forward_test_actions.csv`:

```powershell
& "C:\Users\ivanp\anaconda3\envs\tensorflow\python.exe" ".\Latent Actions\latent_action_phase2_tools.py" `
  --actions ".\Latent Actions\<teacher_output>\walk_forward_test_actions.csv" `
  --output-dir ".\Latent Actions\research_outputs_phase2_teacher_action_audit" `
  --teacher-feature-set base_macro
```

## Required Artifacts

Teacher rerun should produce:

- `walk_forward_results.csv`
- `walk_forward_daily_test_returns.csv`
- `walk_forward_test_actions.csv`
- `artifact_index.json`

Teacher action audit should produce:

- `latent_action_teacher_matrix.csv`
- `latent_action_teacher_simple_codes.csv`
- `latent_action_teacher_action_summary.csv`
- `latent_action_teacher_code_counts.csv`
- `artifact_index.json`

## Evaluation Standard

A latent-action branch survives only if it improves OOS quality under the same discipline as Horizon A:

- median test Sharpe,
- benchmark-excess Sharpe,
- benchmark-excess return,
- fold-level winner behavior,
- selection-rule regret,
- paired/permutation diagnostics where comparable,
- transaction-cost-aware daily returns.

Lower turnover is useful only if it comes with equal or better benchmark-relative OOS performance.

## Mathematical Rationale

Under PPO, the policy update maximizes a clipped surrogate objective:

`E[min(r_t(theta) A_t, clip(r_t(theta), 1 - eps, 1 + eps) A_t)]`

where `A_t` is the estimated advantage and `r_t(theta)` is the policy probability ratio. In this project, richer features have mostly made state selection harder without reliably increasing benchmark-relative advantage. Latent actions target a different failure mode: the action distribution may be too high-dimensional or noisy for stable advantage estimation.

The property to maximize is therefore not raw feature count. For Phase-2, maximize action-state sufficiency:

`I(z_t; A_t^*)` high, `H(z_t | s_t)` controlled, and OOS reward/advantage preserved,

where `z_t` is the latent/discrete action code and `A_t^*` is the teacher action. In practical terms:

- action codes should explain teacher actions with low reconstruction error,
- code usage should not collapse to one bucket,
- codes should be stable across folds/seeds,
- PPO fine-tuning should improve OOS reward metrics, not just imitate the teacher.

## Kill Rules

Stop the branch if:

- action traces are missing or not aligned to run/fold/date keys,
- simple action codes collapse to one dominant code,
- action-code diagnostics are unstable across folds,
- behavior cloning cannot reproduce teacher actions better than a trivial baseline,
- PPO fine-tuning improves turnover but worsens benchmark-relative Sharpe/return,
- gains appear only in validation and not in walk-forward test windows.

## Next Implementation Steps

1. Rerun `base_macro` teacher with the updated runner to produce `walk_forward_test_actions.csv`.
2. Run `latent_action_phase2_tools.py` on that output.
3. Review action-code entropy, fold/seed stability, and reconstruction feasibility.
4. If the audit passes, implement the simplest discretized-action PPO baseline before any VQ/residual tokenizer.
5. Defer SSL/state compression until latent-action Phase-2 either fails cleanly or shows that action regularization is not the limiting factor.
