# Figure captions (generated)

**accuracy_rmse** — Reconstruction RMSE on walkable cells that are not observed at that frame, pooled over the four state channels (left group) and per channel, over the 7 held-out test days (frames 1 to T-2 of each day; EnKF from frame 500, after its warm-up, i.e. without the first ~1.3% of each day). Predictions clipped to the physical bounds. Lower is better; exact values in accuracy_table.tex.

**uncertainty_summary** — Predictive uncertainty on unobserved walkable cells, all methods scored in physical units on common frames; 'All' pools the four channels. (a) CRPS in each channel's physical unit (lower is better). (b) Spread / RMSE, spread = root-mean-square predictive SD; dashed line = 1, below it the method is overconfident. Error bars on 'All': 95% bootstrap intervals over the 7 test days (resampling days). DINCAE's velocity-variance channel is log-normal in physical units and scored with the closed-form log-normal CRPS; all other channels are Gaussian.

**inference_latency** — Median GPU compute time per reconstructed frame on one Tesla V100-PCIE-32GB, prebuilt inputs, same frames for all methods; 4DVarNet solves 200-frame windows and EnKF runs 100 members. The 95th percentile over repeats is within 10.3% of the median for every method and is not drawn.
