# Node Hydraulic Task Sharing

Repository: `node-hydraulic-task-sharing`

## Purpose

This repository contains the source code for a node-level SWMM surrogate model.
It is intended for code inspection and reuse of the NODE modelling, training,
checkpoint, and evaluation workflow.

The model predicts five node-level hydraulic targets:

```text
Depth   - node depth
Head    - hydraulic head
LatIn   - lateral inflow
TotIn   - total inflow
Flood   - node flooding loss
```

The training code supports all 31 non-empty task subsets, fixed equal loss
weighting, and GradNorm task weighting. Each configuration keeps one shared
checkpoint for all included targets. It is selected by the minimum
configuration-level validation loss: the unweighted sum of the task-wise
normalized-space MSEs pooled over the validation samples, nodes, and H1–H6.
The same checkpoint is used for every target in that configuration, and test
metrics are computed only after training. The evaluation code reports
target-level R², RMSE, and MAE after inverse transformation to the physical
scale.

## Code layout

```text
src/node_hydraulic_task_sharing/
├── config.py                    # Node experiment configuration
├── main1.py                     # Dataset, model, and fixed-loss training
├── utils.py                     # Graph WaveNet modules
├── node_metrics.py              # Scaling and validation metrics
├── node_run_support.py          # Run paths and manifests
├── gradnorm_training.py         # GradNorm training
├── run_node_mtl_combinations.py # Fixed-loss task-subset runner
├── run_node_gradnorm.py         # Full-MTL GradNorm runner
├── run_node_gradnorm_subsets.py # GradNorm task-subset runner
└── evaluate_node_test.py        # Test-set evaluation
```

The repository includes one compact derived metrics release under
`data/pooled_metrics/by_seed/`. It contains 310 lookup tables: five seeds × 31
task configurations × two weighting strategies. Each file is named
`<seed>/<config_id>_<method>.csv` and contains one row per included target and
forecast horizon (H1–H6), with `val`, `test`, and `500A` R², RMSE, and MAE
columns. Thus, every released row identifies one seed, model configuration,
target, and horizon while keeping the three evaluation splits together.

These are pooled collection-level metrics after inverse transformation to the
physical scale. The release contains no raw SWMM data, predictions,
node/event-level diagnostics, topology/scaler files, trained checkpoints, or
paper figures. The `by_seed/` tables are the only public result tables; no
additional source-table or manifest files are required to interpret them.

Run the smoke and data-integrity tests after installation with:

```powershell
python -m unittest discover -s tests -v
```

The data-integrity test checks the five-seed, 31-configuration, two-method
coverage and the schema of every `by_seed/` table.
