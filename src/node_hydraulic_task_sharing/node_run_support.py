"""Run manifests, event inventories, and safe output-root handling."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_inventory(split_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(split_dir)
    x_files = {path.name[:-6]: path for path in directory.glob("*_X.npy")}
    y_files = {path.name[:-6]: path for path in directory.glob("*_Y.npy")}
    if not x_files or set(x_files) != set(y_files):
        raise ValueError(f"Missing or unmatched event X/Y arrays in {directory}")
    return [
        {
            "event_id": event_id,
            "x_bytes": x_files[event_id].stat().st_size,
            "y_bytes": y_files[event_id].stat().st_size,
        }
        for event_id in sorted(x_files)
    ]


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=True)
    os.replace(temporary, path)


def prepare_run_directory(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Refusing to reuse a non-empty run directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def apply_data_paths(config: Any, data_root: str | Path, matrix_root: str | Path, scaler_path: str | Path) -> None:
    data_root = Path(data_root).expanduser().resolve()
    matrix_root = Path(matrix_root).expanduser().resolve()
    scaler_path = Path(scaler_path).expanduser().resolve()
    config.train_data_dir = str(data_root / "train")
    config.val_data_dir = str(data_root / "val")
    config.test_data_dir = str(data_root / "test")
    config.topo_matrix_path = str(matrix_root / "nodes_topo.npz")
    config.rbf_matrix_path = str(matrix_root / "nodes_rbf.npz")
    config.scaler_config_path = str(scaler_path)
    required = [
        Path(config.train_data_dir),
        Path(config.val_data_dir),
        Path(config.test_data_dir),
        Path(config.topo_matrix_path),
        scaler_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required Node inputs are missing: " + ", ".join(missing))


def build_run_manifest(
    *,
    method: str,
    seed: int,
    output_root: Path,
    config: Any,
    gradnorm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_events = event_inventory(config.train_data_dir)
    val_events = event_inventory(config.val_data_dir)
    test_events = event_inventory(config.test_data_dir)
    matrix_path = Path(config.topo_matrix_path)
    code_dir = Path(__file__).resolve().parent
    source_paths = [
        code_dir / name for name in (
            "config.py", "main1.py", "utils.py", "node_metrics.py",
            "node_run_support.py", "gradnorm_training.py", "evaluate_node_test.py",
            "run_node_mtl_combinations.py", "run_node_gradnorm.py",
            "run_node_gradnorm_subsets.py",
        )
    ]
    manifest: dict[str, Any] = {
        "status": "running",
        "method": method,
        "seed": int(seed),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(output_root),
        "selection_policy": (
            "One configuration-level checkpoint shared by all included targets, selected by "
            "the minimum sample-weighted normalized-space validation loss, defined as the "
            "sum of task-wise MSEs over H1-H6; test metrics are computed only after training."
        ),
        "test_metric_definition": (
            "All finite sample-node cases are retained for every target, including zero-valued "
            "inflow and flooding cases; no node or time-step filtering is applied."
        ),
        "task_order": list(config.target_names),
        "training": {
            "epochs": config.epochs,
            "patience": config.patience,
            "batch_size": config.batch_size,
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "train_drop_last": True,
            "validation_drop_last": False,
            "configuration_level_shared_checkpoint": True,
            "target_specific_validation_checkpoints": False,
        },
        "data": {
            "train_directory": str(config.train_data_dir),
            "train_events": train_events,
            "validation_directory": str(config.val_data_dir),
            "validation_events": val_events,
            "test_directory": str(config.test_data_dir),
            "test_events": test_events,
            "test_used_for_selection": False,
            "topology_matrix": str(matrix_path),
            "topology_matrix_sha256": sha256_file(matrix_path),
            "scaler_config": str(config.scaler_config_path),
            "scaler_config_sha256": sha256_file(config.scaler_config_path),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "code_sha256": {path.name: sha256_file(path) for path in source_paths},
        "completed_configurations": [],
        "failed_configurations": [],
    }
    if gradnorm:
        manifest["gradnorm"] = gradnorm
    return manifest
