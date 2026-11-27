"""Train all 31 fixed-loss Node task subsets for one seed."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

if __package__:  # Package import when installed.
    from .config import ConfigNode
    from .main1 import run_training_standard
    from .node_run_support import (
        apply_data_paths,
        build_run_manifest,
        prepare_run_directory,
        write_manifest,
    )
else:  # Flat import when run as a script.
    from config import ConfigNode
    from main1 import run_training_standard
    from node_run_support import (
        apply_data_paths,
        build_run_manifest,
        prepare_run_directory,
        write_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 31 fixed-loss Node task subsets for one seed."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="NODES directory containing train/, val/, and test/.")
    parser.add_argument("--matrix-root", type=Path, required=True,
                        help="Directory containing nodes_topo.npz.")
    parser.add_argument("--scaler-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Fresh per-job directory; existing non-empty roots are rejected.")
    parser.add_argument("--skip-test-eval", action="store_true",
                        help="Skip test inference (only for debugging, not paper runs).")
    return parser.parse_args()


def define_experiments() -> list[dict[str, Any]]:
    names = ConfigNode().target_names
    experiments = []
    for size in range(1, len(names) + 1):
        for task_indices in itertools.combinations(range(len(names)), size):
            cfg = ConfigNode()
            cfg.target_idx = list(task_indices)
            cfg.supports_to_use = ["topo"]
            cfg.use_adaptive = True
            cfg.task_name = f"NODE_Topo_Tasks_{'+'.join(map(str, task_indices))}"
            experiments.append({"name": cfg.task_name, "config": cfg})
    if len(experiments) != 31:
        raise RuntimeError(f"Expected 31 non-empty subsets, found {len(experiments)}")
    return experiments


def _checkpoint_rows(
    seed: int,
    experiment_name: str,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    rows = []
    for local_idx, target_name in enumerate(checkpoint["task_names"]):
        row = {
            "method": "fixed_loss",
            "seed": seed,
            "configuration": experiment_name,
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


def _train_configurations(args, output_root, manifest, manifest_path):
    results: list[dict[str, Any]] = []
    experiments = define_experiments()
    print(f"Starting {len(experiments)} Node configurations; seed={args.seed}")
    for position, experiment in enumerate(experiments, start=1):
        cfg = experiment["config"]
        cfg.seed = args.seed
        apply_data_paths(cfg, args.data_root, args.matrix_root, args.scaler_config)
        cfg.save_dir = str(output_root / "checkpoints" / experiment["name"])
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=False)
        print(f"\n>>> [{position}/{len(experiments)}] {experiment['name']} | seed={args.seed}")
        run_training_standard(cfg)
        checkpoint_path = Path(cfg.save_dir) / "best_model.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing configuration checkpoint: {checkpoint_path}")
        results.extend(_checkpoint_rows(args.seed, experiment["name"], checkpoint_path))
        pd.DataFrame(results).to_csv(output_root / "validation_checkpoints.csv", index=False)
        manifest["completed_configurations"].append(experiment["name"])
        write_manifest(manifest_path, manifest)


def _run_test_evaluation(args, output_root):
    if args.skip_test_eval:
        return
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_node_test.py")),
        "--checkpoint-root", str(output_root / "checkpoints"),
        "--data-root", str(args.data_root.expanduser().resolve()),
        "--matrix-root", str(args.matrix_root.expanduser().resolve()),
        "--scaler-config", str(args.scaler_config.expanduser().resolve()),
        "--output-dir", str(output_root / "test_metrics"),
        "--seed", str(args.seed),
        "--method", "fixed_loss",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    base_config = ConfigNode()
    apply_data_paths(base_config, args.data_root, args.matrix_root, args.scaler_config)
    output_root = prepare_run_directory(args.output_root)
    manifest = build_run_manifest(
        method="fixed-loss exhaustive task subsets",
        seed=args.seed,
        output_root=output_root,
        config=base_config,
    )
    manifest_path = output_root / "run_manifest.json"
    write_manifest(manifest_path, manifest)
    (output_root / "checkpoints").mkdir()
    try:
        _train_configurations(args, output_root, manifest, manifest_path)
        _run_test_evaluation(args, output_root)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
