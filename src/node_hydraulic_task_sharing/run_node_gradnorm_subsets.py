"""Train and evaluate GradNorm across all 31 Node task subsets for one seed."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

if __package__:  # Package import when installed.
    from .config import ConfigNode
    from .gradnorm_training import run_training_gradnorm
    from .node_run_support import (
        apply_data_paths,
        build_run_manifest,
        prepare_run_directory,
        sha256_file,
        write_manifest,
    )
    from .run_node_mtl_combinations import define_experiments
else:  # Flat import when run as a script.
    from config import ConfigNode
    from gradnorm_training import run_training_gradnorm
    from node_run_support import (
        apply_data_paths,
        build_run_manifest,
        prepare_run_directory,
        sha256_file,
        write_manifest,
    )
    from run_node_mtl_combinations import define_experiments


FULL_MTL_SOURCE_NAME = "NODE_Topo_FullMTL_GradNorm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--scaler-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-full-mtl-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--gradnorm-lr", type=float, default=0.025)
    return parser.parse_args()


def _find_reusable_full_run(root: Path, seed: int, alpha: float, gradnorm_lr: float) -> Path:
    candidates = []
    for run_dir in sorted(root.glob("slurm_*")):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if (
            manifest.get("status") != "complete"
            or int(manifest.get("seed", -1)) != seed
            or manifest.get("method") != "Full-MTL with GradNorm task weighting"
            or manifest.get("completed_configurations") != [FULL_MTL_SOURCE_NAME]
        ):
            continue
        settings = manifest.get("gradnorm", {})
        if not math.isclose(float(settings.get("alpha", -1)), alpha):
            continue
        if not math.isclose(float(settings.get("learning_rate", -1)), gradnorm_lr):
            continue
        code_hashes = manifest.get("code_sha256", {})
        if not code_hashes.get("gradnorm_training.py") or not code_hashes.get("run_node_gradnorm.py"):
            continue
        candidates.append((run_dir, manifest))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one completed Full-MTL GradNorm source for seed {seed}; "
            f"found {len(candidates)} under {root}"
        )

    run_dir, _ = candidates[0]
    checkpoint_dir = run_dir / "checkpoints" / FULL_MTL_SOURCE_NAME
    expected_targets = ConfigNode().target_names
    source_checkpoint = checkpoint_dir / "best_model.pth"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Full-MTL source is missing configuration checkpoint: {source_checkpoint}")

    event_path = run_dir / "test_metrics" / "test_metrics_by_event.csv"
    pooled_path = run_dir / "test_metrics" / "test_metrics_pooled.csv"
    validation_path = run_dir / "validation_checkpoints.csv"
    if not event_path.is_file() or not pooled_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(f"Full-MTL source metrics are incomplete: {run_dir}")
    events = pd.read_csv(event_path)
    pooled = pd.read_csv(pooled_path)
    if len(events) != 300 or len(pooled) != 30:
        raise ValueError(f"Full-MTL source metrics have unexpected row counts: {run_dir}")
    validation = pd.read_csv(validation_path)
    if (
        len(validation) != 5
        or set(validation["target"]) != set(expected_targets)
        or set(validation["seed"].astype(int)) != {seed}
        or set(validation["method"]) != {"gradnorm"}
        or set(validation["configuration"]) != {FULL_MTL_SOURCE_NAME}
        or set(events["method"]) != {"gradnorm"}
        or set(events["seed"].astype(int)) != {seed}
        or set(events["configuration"]) != {FULL_MTL_SOURCE_NAME}
        or set(pooled["method"]) != {"gradnorm"}
        or set(pooled["seed"].astype(int)) != {seed}
        or set(pooled["configuration"]) != {FULL_MTL_SOURCE_NAME}
    ):
        raise ValueError(f"Full-MTL source validation table is incomplete: {run_dir}")
    return run_dir


def _validate_checkpoint(
    checkpoint_path: Path,
    config: Any,
    seed: int,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing configuration checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_names = [config.target_names[index] for index in config.target_idx]
    if int(checkpoint.get("seed", -1)) != seed:
        raise ValueError(f"Checkpoint seed mismatch: {checkpoint_path}")
    if list(checkpoint.get("task_idx", [])) != list(config.target_idx):
        raise ValueError(f"Checkpoint task indices mismatch: {checkpoint_path}")
    if list(checkpoint.get("task_names", [])) != expected_names:
        raise ValueError(f"Checkpoint task names mismatch: {checkpoint_path}")
    if checkpoint.get("selection_metric") != (
        "minimum configuration-level normalized-space validation loss "
        "(sum of task-wise MSEs over H1-H6)"
    ):
        raise ValueError(f"Checkpoint does not use configuration-level selection: {checkpoint_path}")
    return checkpoint


def _validation_rows(config: Any, seed: int, checkpoint_dir: Path) -> list[dict[str, Any]]:
    checkpoint_path = checkpoint_dir / "best_model.pth"
    checkpoint = _validate_checkpoint(checkpoint_path, config, seed)
    rows = []
    for local_idx, target_name in enumerate(checkpoint["task_names"]):
        row = {
            "method": "gradnorm",
            "seed": seed,
            "configuration": config.task_name,
            "target": target_name,
            "selected_epoch": checkpoint["epoch"],
            "validation_loss": checkpoint["best_val_loss_by_task"][local_idx],
            "configuration_validation_loss": checkpoint["best_val_loss"],
            "validation_mean_r2": checkpoint["best_val_r2_by_task"][local_idx],
            "checkpoint": str(checkpoint_path),
        }
        row.update({
            f"validation_r2_H{horizon + 1}": value
            for horizon, value in enumerate(checkpoint["best_val_r2_by_horizon"][local_idx])
        })
        rows.append(row)
    return rows


def _write_validation_table(rows: list[dict[str, Any]], output_root: Path) -> None:
    path = output_root / "validation_checkpoints.csv"
    temporary = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _reuse_full_mtl(
    config: Any,
    seed: int,
    source_run: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    source_dir = source_run / "checkpoints" / FULL_MTL_SOURCE_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    source = source_dir / "best_model.pth"
    destination = checkpoint_dir / source.name
    shutil.copy2(source, destination)
    _validate_checkpoint(destination, config, seed)
    hashes = {"configuration": sha256_file(destination)}
    source_manifest = source_run / "run_manifest.json"
    return {
        "configuration": config.task_name,
        "source_run": str(source_run),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_gradnorm_training_sha256": json.loads(
            source_manifest.read_text(encoding="utf-8")
        )["code_sha256"]["gradnorm_training.py"],
        "copied_checkpoint_sha256": hashes,
    }


def _run_test_evaluation(args: argparse.Namespace, output_root: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_node_test.py")),
        "--checkpoint-root", str(output_root / "checkpoints"),
        "--data-root", str(args.data_root.expanduser().resolve()),
        "--matrix-root", str(args.matrix_root.expanduser().resolve()),
        "--scaler-config", str(args.scaler_config.expanduser().resolve()),
        "--output-dir", str(output_root / "test_metrics"),
        "--seed", str(args.seed),
        "--method", "gradnorm",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.alpha <= 0 or args.gradnorm_lr <= 0:
        raise ValueError("Seed must be non-negative; GradNorm parameters must be positive")

    experiments = define_experiments()
    expected = [experiment["name"] for experiment in experiments]
    if len(expected) != 31 or len(set(expected)) != 31:
        raise RuntimeError(f"Expected 31 unique Node subsets, found {len(expected)}")

    base_config = ConfigNode()
    base_config.seed = args.seed
    apply_data_paths(base_config, args.data_root, args.matrix_root, args.scaler_config)
    output_root = prepare_run_directory(args.output_root)
    (output_root / "checkpoints").mkdir()
    manifest = build_run_manifest(
        method="GradNorm across all 31 Node task subsets",
        seed=args.seed,
        output_root=output_root,
        config=base_config,
        gradnorm={
            "alpha": args.alpha,
            "learning_rate": args.gradnorm_lr,
            "shared_parameter": "model.out[3].weight",
            "initial_task_weights": "all ones, normalized to active task count",
            "single_task_behavior": "unit weight; no task-balancing degree of freedom",
            "base_task_loss": "per-target normalized-space MSE",
        },
    )
    manifest["expected_configurations"] = expected
    manifest["reused_configurations"] = []
    manifest["configuration_summaries"] = []
    manifest_path = output_root / "run_manifest.json"
    write_manifest(manifest_path, manifest)

    try:
        source_run = _find_reusable_full_run(
            args.reuse_full_mtl_root.expanduser().resolve(),
            args.seed,
            args.alpha,
            args.gradnorm_lr,
        )
        validation_rows: list[dict[str, Any]] = []
        full_indices = list(range(len(base_config.target_names)))

        for position, experiment in enumerate(experiments, start=1):
            config = experiment["config"]
            config.seed = args.seed
            apply_data_paths(config, args.data_root, args.matrix_root, args.scaler_config)
            checkpoint_dir = output_root / "checkpoints" / config.task_name
            config.save_dir = str(checkpoint_dir)
            manifest["current_configuration"] = config.task_name
            write_manifest(manifest_path, manifest)
            print(f"\n>>> [{position}/31] {config.task_name} | seed={args.seed}")

            if list(config.target_idx) == full_indices:
                reuse_record = _reuse_full_mtl(
                    config, args.seed, source_run, checkpoint_dir
                )
                manifest["reused_configurations"].append(reuse_record)
                summary = {"status": "reused_full_mtl", "source_run": str(source_run)}
            else:
                checkpoint_dir.mkdir(parents=True, exist_ok=False)
                summary = run_training_gradnorm(config, args.alpha, args.gradnorm_lr)

            rows = _validation_rows(config, args.seed, checkpoint_dir)
            validation_rows.extend(rows)
            _write_validation_table(validation_rows, output_root)
            manifest["completed_configurations"].append(config.task_name)
            manifest["configuration_summaries"].append({
                "configuration": config.task_name,
                "task_indices": list(config.target_idx),
                "task_names": [config.target_names[index] for index in config.target_idx],
                **summary,
            })
            manifest.pop("current_configuration", None)
            write_manifest(manifest_path, manifest)

        if set(manifest["completed_configurations"]) != set(expected):
            raise RuntimeError("GradNorm configuration sweep did not complete all 31 subsets")
        manifest["current_stage"] = "test_evaluation"
        write_manifest(manifest_path, manifest)
        _run_test_evaluation(args, output_root)
        manifest.pop("current_stage", None)
        manifest["status"] = "complete"
    except BaseException as error:
        failed_item = manifest.get("current_configuration") or manifest.get(
            "current_stage", "initialization"
        )
        manifest["failed_configurations"].append({
            "configuration_or_stage": failed_item,
            "error": f"{type(error).__name__}: {error}",
        })
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
