# Design record

## Scope

This repository is the public code boundary for the node-level SWMM surrogate
workflow. It contains the model, training runners, checkpoint metadata, the
test evaluation path, and the compact per-seed metrics release under
`data/pooled_metrics/by_seed/`. It deliberately excludes the study inputs,
trained checkpoints, node- and event-level aggregation, and figure-building
code.

## Scientific contract

- The canonical node target order is `Depth`, `Head`, `LatIn`, `TotIn`, and
  `Flood`.
- The complete non-empty task-composition space contains 31 subsets: five STL
  subsets, 25 selective subsets, and one Full-MTL subset.
- Inputs are event-window tensors with six historical steps; outputs contain
  six forecast horizons and the selected target channels.
- Fixed training uses the task-wise loss implemented in `main1.py`; GradNorm
  uses the task-balancing implementation in `gradnorm_training.py`.
- Each task configuration keeps one shared checkpoint for all included targets,
  selected by the minimum configuration-level normalized-space validation loss,
  defined as the unweighted sum of task-wise MSEs pooled over validation
  samples, nodes, and H1–H6. The configuration used for test and 500A is chosen
  by mean validation R² across seeds, never by test performance.
- Test evaluation is performed after training and inverse-transforms predictions
  and targets before calculating R², RMSE, and MAE.

## Architecture

```text
X (B, C_in, N, T)
  -> input projection
  -> six gated graph-temporal blocks
  -> fixed topology support and optional adaptive adjacency
  -> skip aggregation and temporal pooling
  -> target-specific output projection
  -> Y_hat (B, C_target, N, H)
```

The model code retains the study configuration: residual width 64, skip width
128, dilations `[1, 2, 4, 8, 1, 2]`, second-order diffusion, and optional
adaptive adjacency. The exact input topology, scaler, and checkpoint files are
provided only through local paths.

## Release boundary

The package does not access network services or download files. Training and
evaluation read caller-supplied local inputs and write only to the requested
output roots. The repository does not contain raw SWMM series, processed
arrays, topology/scaler files, trained weights, node/event-level records, or
paper figures. The only result tables are the per-seed lookup tables under
`data/pooled_metrics/by_seed/`. Each row carries the pooled R², RMSE, and MAE
for validation, test, and 500A. The smoke tests create small in-memory tensors
only; those fixtures are not study results.

## Decision record

### 2026-11-27 — Shared configuration-level checkpoint

Checkpoint selection is defined by one unweighted, sample-pooled validation
MSE sum across all tasks included in a configuration. The resulting
configuration-level checkpoint is shared by every included target; target-level
checkpoint selection is not part of the public evaluation workflow.
