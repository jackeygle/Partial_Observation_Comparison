# History refinement result

| Model | best epoch | validation blind MSE | test blind-walkable RMSE | vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 56 | 0.01849714 | 0.19686588 | +0.000% |
| History loss (w=0.5) | 54 | 0.01871516 | 0.19790989 | +0.530% |
| Framewise | 58 | 0.01849221 | 0.19707327 | +0.105% |

Per-channel blind-walkable MSE:

| Model | density | vx | vy | var |
|---|---:|---:|---:|---:|
| Baseline | 0.01761854 | 0.10597748 | 0.02089461 | 0.01053405 |
| History loss (w=0.5) | 0.01806681 | 0.10686650 | 0.02114428 | 0.01059571 |
| Framewise | 0.01760455 | 0.10635532 | 0.02092085 | 0.01047078 |
