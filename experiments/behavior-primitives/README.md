# Behavior Primitives

Question: which recurring PPO behaviors explain out-of-sample success and failure?

| Question | Answer |
|---|---|
| Why this stage? | G1/G2 failed, so the next step is failure-mode discovery. |
| What is a primitive? | A rolling state-action-return behavior window, not a one-day action code. |
| How many found? | 6 behavior primitives. |
| Main use | Generate targeted hypotheses before another PPO experiment. |

![Primitive leaderboard](figures/primitive_leaderboard.png)

The audit separates reliable-looking behavior from negative excess-return primitives.

![Action risk scatter](figures/action_risk_scatter.png)

Some bad primitives are not just high-turnover; the failure is conditional and behavior-specific.

| Primitive | Share | Excess Sharpe | Initial read |
|---|---:|---:|---|
| `primitive_00` | 0.161 | 1.422 | profitable candidate |
| `primitive_02` | 0.416 | -0.471 | broad failure candidate |
| `primitive_04` | 0.066 | -1.644 | high-action-change failure |
| `primitive_05` | 0.092 | -1.258 | drawdown-stress failure |
