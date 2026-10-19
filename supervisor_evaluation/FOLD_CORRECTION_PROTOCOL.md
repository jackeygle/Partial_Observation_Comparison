# Corrected 4DVarNet final selection, without further training

Approved by the user on 2026-10-09. No training is resumed. Both final methods use
fold_axes=ct and K=20. Their existing snapshots are frozen before test evaluation:
plain epochs 76/80/90; augmented epochs 80/88. Regular snapshots, the saved
training-monitor best, and the rolling last snapshot are eligible only when their
saved effective iteration count is 20. Duplicate epochs must contain identical
solver weights. Epoch 76 is a candidate, not a model chosen by its training metric.

All candidates are selected solely by pooled MSE on blind walkable cells on all
seven validation days, with clipping, complete 200-frame windows, batch 8, and
per-day observations. Existing verified validation scores are reused after SHA
checks; missing candidate scores are computed under the same protocol. The
lowest validation MSE wins. Candidate paths, hashes, aliases and score provenance
are recorded in candidate_plan.json and each run's select_valid.json.

The original training recipe and Adam eps=1e-8 are unchanged. Runs were stopped
at epochs 90/plain and 88/augmented; 150 is the configured maximum budget, not a
completed training claim. The previous fixed-grid rule is superseded by this
uniform existing-K20-snapshot rule. No test result determines selection.

After selection: archive the legacy package and outputs; package and verify the
corrected winners; rerun their accuracy, unified uncertainty, controlled timing,
images and common-support breakdown. Reuse unchanged methods' valid accuracy
products and the fixed EnKF observations/exports. Update affected thesis prose,
tables, percentages and figures from completed results. No new zero/carry or
jitter experiments, no new limitations items, and no PDF compilation.

Completed on 2026-10-09: validation winners are plain epoch 76 and augmented
epoch 80. All evaluation stages, thesis synchronization and the final cross-check
passed. Provenance is in outputs/full/corrected_adoption_audit.json.
