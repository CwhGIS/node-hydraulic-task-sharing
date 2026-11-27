"""Evaluate validation-selected Node checkpoints on the untouched test events."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__:  # Package import when installed.
    from .main1 import GraphWaveNetMT, load_coo_as_dense_rw
    from .node_metrics import JsonInverseScaler
    from .node_run_support import prepare_run_directory
else:  # Flat import when run as a script.
    from main1 import GraphWaveNetMT, load_coo_as_dense_rw
    from node_metrics import JsonInverseScaler
    from node_run_support import prepare_run_directory


@dataclass
class TestEvalContext:
    event_files: list[tuple[str, Path, Path]]
    supports: list[torch.Tensor]
    scaler: JsonInverseScaler
    device: torch.device
    seed: int
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="NODES directory containing test/.")
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--scaler-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=("fixed_loss", "gradnorm"), required=True)
    return parser.parse_args()


def _paired_event_files(test_dir: Path) -> list[tuple[str, Path, Path]]:
    x_files = {path.name[:-6]: path for path in test_dir.glob("*_X.npy")}
    y_files = {path.name[:-6]: path for path in test_dir.glob("*_Y.npy")}
    if not x_files or set(x_files) != set(y_files):
        raise ValueError(f"Missing or unmatched X/Y test events in {test_dir}")
    return [(key, x_files[key], y_files[key]) for key in sorted(x_files)]


def _sufficient_statistics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    count = int(y.size)
    if count == 0:
        return {
            "n_valid": 0,
            "n_nonzero_true": 0,
            "nonzero_fraction": float("nan"),
            "sum_y": 0.0,
            "sum_y2": 0.0,
            "sse": 0.0,
            "sae": 0.0,
            "r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }
    error = p - y
    sum_y = float(y.sum())
    sum_y2 = float(np.square(y).sum())
    sse = float(np.square(error).sum())
    sae = float(np.abs(error).sum())
    total = max(0.0, sum_y2 - sum_y * sum_y / count)
    # Inverting log1p turns an exact zero into ~1e-17, so an exact `> 0` test
    # reports almost every value as nonzero; use a scale-aware tolerance.
    tolerance = max(1e-9, 1e-12 * float(np.max(np.abs(y))))
    nonzero = int(np.count_nonzero(y > tolerance))
    return {
        "n_valid": count,
        "n_nonzero_true": nonzero,
        "nonzero_fraction": nonzero / count,
        "sum_y": sum_y,
        "sum_y2": sum_y2,
        "sse": sse,
        "sae": sae,
        "r2": float(1.0 - sse / total) if total > 1e-12 else float("nan"),
        "rmse": math.sqrt(sse / count),
        "mae": sae / count,
    }


def _build_model(
    checkpoint: dict[str, Any],
    sample_x: np.ndarray,
    horizon: int,
    context: TestEvalContext,
) -> torch.nn.Module:
    config = checkpoint["config"]
    target_idx = checkpoint["task_idx"]
    model = GraphWaveNetMT(
        N=sample_x.shape[1],
        C_in=sample_x.shape[-1],
        C_out=len(target_idx),
        C_res=config["C_res"],
        C_skip=config["C_skip"],
        K=config["diffusion_steps_K"],
        dilations=config["dilations"],
        use_adaptive=config["use_adaptive"],
        num_static_supports=len(context.supports),
        horizon=horizon,
    ).to(context.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _evaluate_checkpoint(
    checkpoint_path: Path,
    experiment_name: str,
    target_name: str,
    context: TestEvalContext,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    task_idx = list(checkpoint["task_idx"])
    if target_name not in checkpoint["task_names"]:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain {target_name}")
    local_task_idx = checkpoint["task_names"].index(target_name)
    rows = []
    batch_size = int(checkpoint["config"].get("batch_size", 256))
    first_x = np.load(context.event_files[0][1], mmap_mode="r")
    first_y = np.load(context.event_files[0][2], mmap_mode="r")
    model = _build_model(checkpoint, first_x[0], first_y.shape[1], context)

    for event_index, (event_id, x_path, y_path) in enumerate(context.event_files):
        if event_index == 0:
            x = np.asarray(first_x, dtype=np.float32)
            y = np.asarray(first_y, dtype=np.float32)
        else:
            x = np.load(x_path).astype(np.float32)
            y = np.load(y_path).astype(np.float32)
        if x.shape[0] == 0 or y.shape[0] != x.shape[0]:
            raise ValueError(f"Invalid sample dimensions for test event {event_id}")
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        prediction_batches = []
        target_batches = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(context.device, non_blocking=True).permute(0, 3, 2, 1)
            prediction = model(x_batch, context.supports)
            prediction_batches.append(prediction[:, local_task_idx].float().cpu().numpy())
            target_batches.append(y_batch[..., task_idx[local_task_idx]].permute(0, 2, 1).numpy())
        prediction_norm = np.concatenate(prediction_batches, axis=0)
        target_norm = np.concatenate(target_batches, axis=0)
        prediction = context.scaler.inverse_transform(prediction_norm, target_name)
        target = context.scaler.inverse_transform(target_norm, target_name)
        for horizon in range(target.shape[-1]):
            row = {
                "method": context.method,
                "seed": context.seed,
                "configuration": experiment_name,
                "target": target_name,
                "event_id": event_id,
                "horizon": horizon + 1,
                "selected_epoch": int(checkpoint["epoch"]),
                "n_nodes": int(target.shape[1]),
            }
            row.update(_sufficient_statistics(target[..., horizon], prediction[..., horizon]))
            rows.append(row)
    del model
    return rows


def _pooled_metrics(event_metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "seed", "configuration", "target", "horizon"]
    sums = event_metrics.groupby(group_cols, as_index=False).agg(
        n_valid=("n_valid", "sum"),
        n_nonzero_true=("n_nonzero_true", "sum"),
        sum_y=("sum_y", "sum"),
        sum_y2=("sum_y2", "sum"),
        sse=("sse", "sum"),
        sae=("sae", "sum"),
        n_events=("event_id", "nunique"),
    )
    total = np.maximum(0.0, sums["sum_y2"] - np.square(sums["sum_y"]) / sums["n_valid"].clip(lower=1))
    sums["r2"] = np.where(total > 1e-12, 1.0 - sums["sse"] / total, np.nan)
    sums["rmse"] = np.sqrt(sums["sse"] / sums["n_valid"].clip(lower=1))
    sums["mae"] = sums["sae"] / sums["n_valid"].clip(lower=1)
    sums["nonzero_fraction"] = sums["n_nonzero_true"] / sums["n_valid"].clip(lower=1)
    return sums


def main() -> None:
    args = parse_args()
    if not args.checkpoint_root.is_dir():
        raise NotADirectoryError(args.checkpoint_root)
    output_dir = prepare_run_directory(args.output_dir)
    test_dir = args.data_root.expanduser().resolve() / "test"
    event_files = _paired_event_files(test_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Test evaluation requires CUDA on the HPC node")
    scaler = JsonInverseScaler(args.scaler_config)
    adjacency, _ = load_coo_as_dense_rw(str(args.matrix_root / "nodes_topo.npz"), device)
    if adjacency is None:
        raise RuntimeError("Could not load the Node topology matrix")
    supports = [adjacency]

    checkpoint_paths = sorted(args.checkpoint_root.glob("*/best_model.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No configuration checkpoints found in {args.checkpoint_root}")
    context = TestEvalContext(
        event_files=event_files,
        supports=supports,
        scaler=scaler,
        device=device,
        seed=args.seed,
        method=args.method,
    )
    all_rows = []
    for checkpoint_path in checkpoint_paths:
        experiment_name = checkpoint_path.parent.name
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for target_name in checkpoint["task_names"]:
            print(f"Test evaluation: {experiment_name} / {target_name} (shared checkpoint)")
            all_rows.extend(_evaluate_checkpoint(
                checkpoint_path,
                experiment_name,
                target_name,
                context,
            ))

    event_metrics = pd.DataFrame(all_rows)
    if event_metrics.empty:
        raise RuntimeError("Test evaluation produced no metrics")
    event_metrics.to_csv(output_dir / "test_metrics_by_event.csv", index=False)
    _pooled_metrics(event_metrics).to_csv(output_dir / "test_metrics_pooled.csv", index=False)
    print(f"Saved {len(event_metrics)} event-horizon rows to {output_dir}")


if __name__ == "__main__":
    main()
