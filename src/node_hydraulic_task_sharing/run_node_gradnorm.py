"""Train and evaluate the Full-MTL GradNorm Node control for one seed."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

if __package__:  # Package import when installed.
    from .config import ConfigNode
    from .gradnorm_training import run_training_gradnorm
    from .node_run_support import (
        apply_data_paths,
        build_run_manifest,
        prepare_run_directory,
        write_manifest,
    )
else:  # Flat import when run as a script.
    from config import ConfigNode
    from gradnorm_training import run_training_gradnorm
    from node_run_support import (
        apply_data_paths,
        build_run_manifest,
        prepare_run_directory,
        write_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="NODES directory containing train/, val/, and test/.")
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--scaler-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--gradnorm-lr", type=float, default=0.025)
    parser.add_argument("--skip-test-eval", action="store_true",
                        help="Skip test inference (only for debugging, not paper runs).")
    return parser.parse_args()


def _create_config(args):
    config = ConfigNode()
    config.target_idx = list(range(len(config.target_names)))
    config.supports_to_use = ["topo"]
    config.use_adaptive = True
    config.task_name = "NODE_Topo_FullMTL_GradNorm"
    config.seed = args.seed
    apply_data_paths(config, args.data_root, args.matrix_root, args.scaler_config)
    return config


def _write_validation_results(config, args, checkpoint_dir, output_root):
    checkpoint_path = checkpoint_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing configuration checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    rows = []
    for local_idx, target_name in enumerate(checkpoint["task_names"]):
        row = {
            "method": "gradnorm",
            "seed": args.seed,
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
    pd.DataFrame(rows).to_csv(output_root / "validation_checkpoints.csv", index=False)


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
        "--method", "gradnorm",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.alpha <= 0 or args.gradnorm_lr <= 0:
        raise ValueError("Seed must be non-negative; GradNorm alpha and learning rate positive")
    config = _create_config(args)
    output_root = prepare_run_directory(args.output_root)
    checkpoint_dir = output_root / "checkpoints" / config.task_name
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    config.save_dir = str(checkpoint_dir)
    manifest = build_run_manifest(
        method="Full-MTL with GradNorm task weighting",
        seed=args.seed,
        output_root=output_root,
        config=config,
        gradnorm={
            "alpha": args.alpha,
            "learning_rate": args.gradnorm_lr,
            "shared_parameter": "model.out[3].weight",
            "initial_task_weights": "all ones, normalized to sum to five",
            "base_task_loss": "per-target normalized-space MSE",
        },
    )
    manifest_path = output_root / "run_manifest.json"
    write_manifest(manifest_path, manifest)
    try:
        manifest["training_summary"] = run_training_gradnorm(config, args.alpha, args.gradnorm_lr)
        _write_validation_results(config, args, checkpoint_dir, output_root)
        manifest["completed_configurations"].append(config.task_name)
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
