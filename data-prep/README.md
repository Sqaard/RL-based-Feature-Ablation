# Data Prep

Question: what data feeds the PPO experiments?

| Input | Role |
|---|---|
| Dow 30 prices and indicators | Trading universe and technical state. |
| Macro context | The `base_macro` reference feature family. |
| WRDS-style fundamentals/proxies | Candidate feature-family inputs. |

The public repo keeps the preprocessing notebook, not large raw data dumps. Rebuild data locally before launching new training runs.
