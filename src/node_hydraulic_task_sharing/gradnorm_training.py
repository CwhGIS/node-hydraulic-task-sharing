"""GradNorm training path for Node models with one or more active tasks."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

if __package__:  # Package import when installed.
    from .main1 import (
        DrainageNPDataset,
        GraphWaveNetMT,
        HybridLoss,
        _checkpoint_config,
        evaluate,
        load_coo_as_dense_rw,
        seed_everything,
    )
    from .node_metrics import JsonInverseScaler
else:  # Flat import when run as a script.
    from main1 import (
        DrainageNPDataset,
        GraphWaveNetMT,
        HybridLoss,
        _checkpoint_config,
        evaluate,
        load_coo_as_dense_rw,
        seed_everything,
    )
    from node_metrics import JsonInverseScaler


class GradNormBalancer:
    """Learn task weights while keeping their sum equal to the task count."""

    def __init__(
        self,
        num_tasks: int,
        device: torch.device,
        alpha: float = 1.5,
        learning_rate: float = 0.025,
    ) -> None:
        if num_tasks < 1:
            raise ValueError("GradNorm requires at least one active task")
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.eps = 1e-8
        self.weights = torch.nn.Parameter(torch.ones(num_tasks, device=device))
        self.initial_losses: torch.Tensor | None = None
        self.optimizer = optim.Adam([self.weights], lr=learning_rate)

    def combine(
        self,
        task_losses: torch.Tensor,
        shared_parameters: list[torch.nn.Parameter],
    ) -> tuple[torch.Tensor, float]:
        losses = task_losses.float()
        if losses.numel() != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} task losses, received {losses.numel()}"
            )
        if self.num_tasks == 1:
            return losses.sum(), 0.0
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clamp_min(self.eps)

        weighted_losses = self.weights * losses
        model_weights = self.weights.detach().clone()
        combined_loss = torch.sum(model_weights * losses)
        gradient_norms = []
        for task_idx in range(self.num_tasks):
            gradients = torch.autograd.grad(
                weighted_losses[task_idx],
                shared_parameters,
                retain_graph=True,
                create_graph=True,
                allow_unused=False,
            )
            squared_norm = sum(gradient.float().pow(2).sum() for gradient in gradients)
            gradient_norms.append(torch.sqrt(squared_norm + self.eps))
        norms = torch.stack(gradient_norms)
        rates = losses.detach() / self.initial_losses
        rates = rates / rates.mean().clamp_min(self.eps)
        target_norms = norms.detach().mean() * rates.pow(self.alpha)
        gradnorm_loss = torch.abs(norms - target_norms.detach()).sum()

        self.optimizer.zero_grad(set_to_none=True)
        (weight_gradient,) = torch.autograd.grad(gradnorm_loss, self.weights, retain_graph=True)
        self.weights.grad = weight_gradient
        self.optimizer.step()
        with torch.no_grad():
            self.weights.clamp_(min=self.eps)
            self.weights.mul_(self.num_tasks / self.weights.sum().clamp_min(self.eps))
        return combined_loss, float(gradnorm_loss.detach().cpu())

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.detach().cpu(),
            "initial_losses": None if self.initial_losses is None else self.initial_losses.detach().cpu(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "alpha": self.alpha,
        }


def _task_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        F.mse_loss(prediction[:, idx].float(), target[:, idx].float())
        for idx in range(prediction.shape[1])
    ])


def _atomic_save(checkpoint: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def run_training_gradnorm(
    config: Any,
    alpha: float = 1.5,
    gradnorm_lr: float = 0.025,
) -> dict[str, Any]:
    target_idx = list(config.target_idx)
    if (
        not target_idx
        or len(set(target_idx)) != len(target_idx)
        or any(index < 0 or index >= len(config.target_names) for index in target_idx)
    ):
        raise ValueError("target_idx must contain unique, in-range Node task indices")
    if getattr(config, "model_type", "standard") != "standard":
        raise ValueError("GradNorm supports the standard GraphWaveNetMT model only")

    seed_everything(int(config.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The GradNorm HPC control requires a CUDA device")
    config.use_amp = False
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    supports = []
    if "topo" in config.supports_to_use:
        adjacency, _ = load_coo_as_dense_rw(config.topo_matrix_path, device)
        if adjacency is None:
            raise RuntimeError(f"Could not load topology matrix: {config.topo_matrix_path}")
        supports.append(adjacency)
    if "rbf" in config.supports_to_use:
        adjacency, _ = load_coo_as_dense_rw(config.rbf_matrix_path, device)
        if adjacency is None:
            raise RuntimeError(f"Could not load RBF matrix: {config.rbf_matrix_path}")
        supports.append(adjacency)
    config.supports_list = supports

    train_dataset = DrainageNPDataset(config.train_data_dir, config.task_name)
    val_dataset = DrainageNPDataset(config.val_data_dir, config.task_name)
    if len(train_dataset) < config.batch_size or len(val_dataset) == 0:
        raise RuntimeError("Train/validation split is empty or smaller than one training batch")
    loader_kwargs = {
        "batch_size": config.batch_size,
        "pin_memory": True,
        "num_workers": 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    if not len(train_loader):
        raise RuntimeError("Training loader has zero steps per epoch")

    sample_x, sample_y = train_dataset[0]
    config.horizon = sample_y.shape[0]
    model = GraphWaveNetMT(
        N=sample_x.shape[1],
        C_in=sample_x.shape[-1],
        C_out=len(config.target_idx),
        C_res=config.C_res,
        C_skip=config.C_skip,
        K=config.diffusion_steps_K,
        dilations=config.dilations,
        use_adaptive=config.use_adaptive,
        num_static_supports=len(supports),
        horizon=config.horizon,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_epochs
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs - config.warmup_epochs), eta_min=config.min_lr
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs]
    )
    balancer = GradNormBalancer(len(config.target_idx), device, alpha, gradnorm_lr)
    shared_parameters = [model.out[3].weight]
    criterion = HybridLoss(
        config.target_idx,
        use_dtw=config.use_dtw,
        dtw_weight=config.dtw_weight,
        dtw_gamma=config.dtw_gamma,
    ).to(device)
    inverse_scaler = JsonInverseScaler(config.scaler_config_path)
    config._inverse_scaler = inverse_scaler
    task_names = [config.target_names[idx] for idx in config.target_idx]
    history_path = save_dir / "history.csv"
    columns = [
        "epoch", "train_weighted_loss", "train_mean_task_mse", "val_loss",
        "val_mean_r2", "gradnorm_loss", "learning_rate",
    ]
    columns += [f"weight_{name}" for name in task_names]
    columns += [f"val_loss_{name}" for name in task_names]
    columns += [f"val_r2_{name}_h{h + 1}" for name in task_names for h in range(config.horizon)]

    best_mean_r2 = float("-inf")
    best_val_loss = float("inf")
    best_val_loss_by_task = [float("nan")] * len(task_names)
    best_val_r2_by_task = [float("nan")] * len(task_names)
    best_epoch = -1
    patience_count = 0
    started_at = time.time()
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=columns)
        writer.writeheader()
        for epoch in range(config.epochs):
            model.train()
            weighted_sum = 0.0
            gradnorm_sum = 0.0
            task_loss_sum = np.zeros(len(task_names), dtype=np.float64)
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device, non_blocking=True).permute(0, 3, 2, 1)
                y_batch = y_batch.to(device, non_blocking=True)
                target = y_batch[..., config.target_idx].permute(0, 3, 2, 1)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x_batch, config.supports_list)
                task_losses = _task_mse(prediction, target)
                weighted_loss, gradnorm_loss = balancer.combine(task_losses, shared_parameters)
                optimizer.zero_grad(set_to_none=True)
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                optimizer.step()
                task_loss_sum += task_losses.detach().cpu().numpy()
                weighted_sum += float(weighted_loss.detach().cpu())
                gradnorm_sum += gradnorm_loss

            train_task_mse = task_loss_sum / len(train_loader)
            val_loss, horizon_r2, mean_r2, loss_by_task = evaluate(
                model, val_loader, criterion, device, config
            )
            finite_scores = [score for score in mean_r2 if np.isfinite(score)]
            current_mean_r2 = float(np.mean(finite_scores)) if finite_scores else float("nan")
            weights = balancer.weights.detach().cpu().numpy().tolist()
            row: dict[str, Any] = {
                "epoch": epoch + 1,
                "train_weighted_loss": weighted_sum / len(train_loader),
                "train_mean_task_mse": float(train_task_mse.mean()),
                "val_loss": val_loss,
                "val_mean_r2": current_mean_r2,
                "gradnorm_loss": gradnorm_sum / len(train_loader),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            row.update({f"weight_{name}": weights[i] for i, name in enumerate(task_names)})
            row.update({
                f"val_loss_{name}": loss_by_task[i] for i, name in enumerate(task_names)
            })
            for task_idx, name in enumerate(task_names):
                row.update({
                    f"val_r2_{name}_h{h + 1}": value
                    for h, value in enumerate(horizon_r2[task_idx])
                })
            writer.writerow(row)
            history_file.flush()
            scheduler.step()
            print(
                f"Epoch {epoch + 1:04d} | train={row['train_weighted_loss']:.6f} "
                f"val_R2={current_mean_r2:.6f} weights={weights}"
            )

            if val_loss < best_val_loss:
                best_mean_r2 = current_mean_r2
                best_val_loss = val_loss
                best_val_loss_by_task = list(loss_by_task)
                best_val_r2_by_task = list(mean_r2)
                best_epoch = epoch + 1
                patience_count = 0
                _atomic_save({
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "gradnorm_state": balancer.state_dict(),
                    "config": _checkpoint_config(config),
                    "seed": int(config.seed),
                    "task_idx": list(config.target_idx),
                    "task_names": task_names,
                    "selection_metric": (
                        "minimum configuration-level normalized-space validation loss "
                        "(sum of task-wise MSEs over H1-H6)"
                    ),
                    "best_val_loss": best_val_loss,
                    "best_val_loss_by_task": best_val_loss_by_task,
                    "best_val_r2_mean": best_mean_r2,
                    "best_val_r2_by_task": best_val_r2_by_task,
                    "best_val_r2_by_horizon": horizon_r2,
                }, save_dir / "best_model.pth")
            else:
                patience_count += 1
                if patience_count >= config.patience:
                    print(f"Early stopping at epoch {epoch + 1} (patience={config.patience})")
                    break

    if best_epoch < 0:
        raise RuntimeError("GradNorm training ended without a valid checkpoint")
    summary = {
        "status": "success",
        "method": "GradNorm task weighting",
        "configuration": config.task_name,
        "seed": int(config.seed),
        "task_indices": list(config.target_idx),
        "task_names": task_names,
        "alpha": float(alpha),
        "gradnorm_lr": float(gradnorm_lr),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_by_task": dict(zip(task_names, best_val_loss_by_task)),
        "best_val_r2_mean_by_task": dict(zip(task_names, best_val_r2_by_task)),
        "final_gradnorm_weights": dict(zip(task_names, weights)),
        "elapsed_seconds": time.time() - started_at,
        "checkpoint_dir": str(save_dir),
    }
    temporary = save_dir / "summary.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=True)
    os.replace(temporary, save_dir / "summary.json")
    return summary
