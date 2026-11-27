"""Shared scaling and validation metrics for node-level experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TASK_TO_SCALER_KEY = {
    "Depth": "invert_depth",
    "Head": "hydraulic_head",
    "LatIn": "lateral_inflow",
    "TotIn": "total_inflow",
    "Flood": "flooding_losses",
}


class JsonInverseScaler:
    def __init__(self, config_path: str | Path) -> None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Scaler configuration not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            self.config = json.load(stream)

    def inverse_transform(self, values: np.ndarray, target_name: str) -> np.ndarray:
        key = TASK_TO_SCALER_KEY.get(target_name, target_name)
        if key not in self.config:
            raise KeyError(f"No scaler entry for target {target_name!r} ({key!r})")
        params = self.config[key]
        scaler_type = params.get("scaler_type", "identity")
        values = np.asarray(values, dtype=np.float64)
        if scaler_type == "standard":
            values = values * float(params.get("std", 1.0)) + float(params.get("mean", 0.0))
        elif scaler_type == "minmax":
            low = float(params.get("min", 0.0))
            high = float(params.get("max", 1.0))
            values = values * (high - low) + low
        elif scaler_type != "identity":
            raise ValueError(f"Unsupported scaler type {scaler_type!r} for {key}")
        if params.get("use_log1p", False):
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.expm1(values)
            values = np.maximum(values, 0.0)
        return values


def safe_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return float("nan")
    y = y[valid]
    p = p[valid]
    total = np.square(y - y.mean()).sum()
    if total <= 1e-12:
        return float("nan")
    return float(1.0 - np.square(y - p).sum() / total)


def validation_loss_by_task(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> list[float]:
    """Return mean squared error per output task in normalized space.

    Inputs have shape [samples, tasks, nodes, horizons]. This is the same
    quantity HybridLoss minimises for each task, so the per-task values are
    directly comparable with the validation loss recorded during training.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    axes = (0, 2, 3)
    squared = np.square(pred - true)
    finite = np.isfinite(squared).sum(axis=axes)
    sums = np.where(np.isfinite(squared), squared, 0.0).sum(axis=axes)
    with np.errstate(divide="ignore", invalid="ignore"):
        losses = np.where(finite > 0, sums / finite, np.nan)
    return [float(value) for value in losses]


def configuration_validation_loss(loss_by_task: list[float]) -> float:
    """Return the configuration-level validation loss used for checkpointing.

    The configuration objective is the unweighted sum of the normalized-space
    task MSEs.  Each task MSE is already pooled over all validation samples,
    nodes, and forecast horizons, so this sum gives every included task equal
    weight while retaining one common checkpoint for the whole configuration.
    """
    losses = np.asarray(loss_by_task, dtype=np.float64)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("At least one task loss is required")
    if not np.all(np.isfinite(losses)):
        raise ValueError("Configuration validation loss contains non-finite task losses")
    return float(losses.sum())


def validation_r2_by_task(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_idx: list[int],
    target_names: list[str],
    inverse_scaler: JsonInverseScaler,
) -> tuple[list[list[float]], list[float]]:
    """Return physical-space R2 by horizon and each model output task.

    Inputs have shape [samples, tasks, nodes, horizons]. Each horizon pools
    samples and nodes, matching the paper's test metric convention.
    """
    horizon_scores: list[list[float]] = []
    mean_scores: list[float] = []
    for local_idx, task_idx_value in enumerate(target_idx):
        name = target_names[task_idx_value]
        pred = inverse_scaler.inverse_transform(predictions[:, local_idx], name)
        true = inverse_scaler.inverse_transform(targets[:, local_idx], name)
        scores = [safe_r2(true[..., h], pred[..., h]) for h in range(true.shape[-1])]
        finite = [score for score in scores if np.isfinite(score)]
        horizon_scores.append(scores)
        mean_scores.append(float(np.mean(finite)) if finite else float("nan"))
    return horizon_scores, mean_scores
