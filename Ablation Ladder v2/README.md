## Ablation Ladder v2

Ablation Ladder v2 is the current Horizon A experiment in this repository.  
Its purpose was not to search for a more complex policy architecture, but to extend the walk-forward framework with a cleaner feature ladder, robust selection infrastructure, daily test exports, and regime-aware diagnostics.

### Experimental setup

The run used a fixed reference PPO agent with:

- `custom_reward`
- `custom_mlp_policy`
- `checkpoint_selection_rule = checkpoint_robust_score`
- `configuration_selection_rule = robust_q25_retention`

Walk-forward evaluation used 3 seeds (`42, 123, 999`) and the following feature ladder:

- `base` — technical baseline
- `base_macro` — baseline + macro context
- `base_macro_exogenous_plus` — baseline + macro + causal calendar/event layer
- `base_macro_hmm` — HMM branch, treated as a negative control
- `base_macro_gru` — GRU branch, treated as a negative control

### Main findings

The strongest result in v2 remained **`base_macro`**, which again emerged as the most robust feature set on test. This confirms the main feature-level conclusion of the project: compact macro context remains more useful than the current learned extensions. :contentReference[oaicite:0]{index=0}

The new **calendar/event exogenous layer** (`base_macro_exogenous_plus`) did **not** improve performance beyond the macro baseline. This is an important negative result: adding an extra exogenous layer is not sufficient by itself if that layer does not add stronger regime information. :contentReference[oaicite:1]{index=1}

Among the learned extensions, **`base_macro_gru`** performed better than **`base_macro_hmm`**, but still did not outperform `base_macro`. In the current pipeline, GRU features look less damaging than HMM features, but neither branch justifies becoming the main research direction. :contentReference[oaicite:2]{index=2}

A crucial clarification is that the shared branches between **v1** and **v2** (`base`, `base_macro`, `base_macro_hmm`) are unchanged in the current stored outputs. Their selected artifact types, validation Sharpe, and test Sharpe are the same in both versions. So v2 should **not** be interpreted as improving those existing branches; rather, it extends the experimental framework around them. :contentReference[oaicite:3]{index=3}

Overall, v2 supports the following interpretation:

- **macro context remains the strongest robust signal**
- **HMM remains a valid negative control**
- **GRU still does not justify a main branch**
- **the main advance in v2 is methodological, not a new winning feature set**

### Selection-rule result

One of the most important outcomes of v2 is methodological rather than architectural.

The robust configuration-selection rule (`robust_q25_retention`) outperformed `sharpe_only` in post-hoc fold-level configuration selection:
- higher selected test Sharpe
- better match rate with actual fold test winners
- lower test-winner regret

This result should be interpreted carefully: it does **not** mean that v2 changed the run-level outcomes of the shared branches. It means that, given the stored fold-level results, robust configuration scoring is more reliable than Sharpe-only scoring when deciding which feature set to choose. 

### Practical conclusion

Ablation Ladder v2 did not produce a new feature winner beyond `base_macro`, and it did not improve the already shared branches relative to v1. Its contribution is different:

1. it formalized the Horizon A reporting and selection framework,
2. it showed that robust configuration-level selection is more reliable than Sharpe-only selection,
3. it showed that the new causal calendar/event layer did not add value over the macro baseline,
4. it added daily test exports and regime-diagnostics infrastructure for future experiments. 
