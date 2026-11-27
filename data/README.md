# Data boundary

The only public result data are the derived lookup tables under
`pooled_metrics/by_seed/`. The release contains no raw SWMM inputs or outputs.

There are five seeds (3407–3411), 31 task configurations, two weighting
strategies (`fixed_loss` and `gradnorm`), five targets, and six forecast
horizons. Each file has one row per included target and horizon and stores
`val_r2`, `val_rmse`, `val_mae`, `test_r2`, `test_rmse`, `test_mae`,
`500A_r2`, `500A_rmse`, and `500A_mae` together with the seed, method, and
configuration identity.

The metrics are pooled over the corresponding evaluation split and reported
after inverse transformation to physical units. Test and 500A values are
evaluation-only; they are not used for checkpoint or task-configuration
selection. Each configuration uses one shared checkpoint for all included
targets, selected by the minimum configuration-level validation loss: the
unweighted sum of task-wise normalized-space MSEs pooled over validation
samples, nodes, and H1–H6.

The public release excludes event-split arrays, topology matrices, scalers,
predictions, node-level or event-level diagnostics, trained checkpoints,
absolute local paths, and checkpoint hashes. Training and evaluation entry
points accept the restricted study inputs from local paths.
