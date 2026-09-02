# Reply to Thomas (English)

**Subject: Re: map resolution & benchmark comparison**

Dear Thomas,

Thank you for the clarifications.

**On the map resolution:** understood. I will keep the map at the same 1 m resolution as the state, and simply change the occupancy threshold to 50% (a cell is treated as blocked only if at least 50% of it is occupied by an obstacle). I agree this should resolve the narrow-corridor / "see-through-wall" issue.

**On the benchmark:** apologies, the "Vocal Filter" was a typo in my summary — there is no such method. I will compare our model against the existing Ensemble Kalman Filter (EnKF), as you intended.

**One question about uncertainty.** I noticed that our 4DVarNet is a deterministic method: it outputs a single reconstruction with no uncertainty estimate, which is consistent with the original paper (Fablet et al., 2020) — that work does not do uncertainty quantification either. The only method that naturally produces an uncertainty/spread is the ensemble-based EnKF. I would like to ask how you would prefer me to handle this:

- (a) show the uncertainty (ensemble spread) only for the EnKF baseline, and report only the deterministic reconstruction — with a colorbar — for 4DVarNet; or
- (b) additionally add an uncertainty-quantification scheme to 4DVarNet (e.g., MC-dropout or an ensemble), which would go beyond the scope of the original paper.

Best regards,
Xinle
