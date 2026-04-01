## Ablation Ladder v2

Ablation Ladder v2 is the main Horizon A experiment in this repository.  
The goal of this stage was not to search for a more complex policy architecture, but to test whether a cleaner and more robust feature ladder would improve out-of-sample behavior under a stricter walk-forward protocol.

### Experimental setup

The run used a fixed reference PPO agent with:

- `custom_reward`
- `custom_mlp_policy`
- `checkpoint_selection_rule = checkpoint_robust_score`
- `configuration_selection_rule = robust_q25_retention`

Walk-forward evaluation used 3 seeds (`42, 123, 999`) and a fixed feature ladder:

- `base` — technical baseline
- `base_macro` — baseline + macro context
- `base_macro_exogenous_plus` — baseline + macro + causal calendar/event layer
- `base_macro_hmm` — HMM branch, treated as a negative control
- `base_macro_gru` — GRU branch, treated as a negative control

### Main findings

The strongest and most stable result in v2 remained **`base_macro`**.  
This means that compact macro context features still provided the most robust signal for the RL agent.

The new **calendar/event exogenous layer** (`base_macro_exogenous_plus`) did **not** improve performance over the macro baseline.  
This is an important negative result: simply adding more non-technical features is not enough — the added exogenous layer must carry real regime information, not just additional structure.

Among the learned extensions, **`base_macro_gru`** performed better than **`base_macro_hmm`**, but still did not beat the simpler `base_macro` configuration.  
So in the current pipeline, GRU features look less harmful than HMM features, but neither branch justifies becoming the main research direction.

Overall, v2 reinforces the core project conclusion:

- **macro context remains the strongest robust signal**
- **HMM remains a valid negative control**
- **GRU does not yet justify a main branch**
- **feature quality and selection protocol matter more than added model complexity**

### Selection-rule result

One of the most important outcomes of v2 is methodological rather than architectural.

The robust configuration-selection rule (`robust_q25_retention`) clearly outperformed `sharpe_only` in fold-level decision quality:

- higher selected test Sharpe
- better match rate with actual test winners
- much lower test-winner regret

This supports the Horizon A shift away from peak validation Sharpe and toward stability-driven model selection.

### Regime diagnostics

v2 also introduced daily test exports and exogenous regime diagnostics based on interpretable market regimes.  
This infrastructure worked correctly, but most non-`unknown` regimes inside a 3-month test window contained too few days for stable regime-level Sharpe estimates.  
As a result, regime diagnostics are now technically available, but still limited in statistical strength.

### Practical conclusion

Ablation Ladder v2 did not produce a new feature winner beyond `base_macro`, but it still represents a meaningful step forward:

1. it validated the Horizon A reporting and selection framework,
2. it confirmed that robust selection is more reliable than Sharpe-only selection,
3. it showed that the new causal calendar/event layer did not add value yet.
