# main1.py

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

# 导入自定义模块
if __package__:  # Package import when installed.
    from .utils import GraphWaveNetMT, GraphWaveNet_ViewFusion
    from .node_metrics import (
        JsonInverseScaler,
        configuration_validation_loss,
        validation_loss_by_task,
        validation_r2_by_task,
    )
else:  # Flat import when run as a script.
    from utils import GraphWaveNetMT, GraphWaveNet_ViewFusion
    from node_metrics import (
        JsonInverseScaler,
        configuration_validation_loss,
        validation_loss_by_task,
        validation_r2_by_task,
    )



# =============================================================================
#  🛠️ 1. 工具函数与数据集定义
# =============================================================================

def seed_everything(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ⭐ 新增：开启 TF32 加速 (3090/4090 必备)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_coo_as_dense_rw(npz_path, device=None):
    """加载稀疏矩阵并转换为随机游走归一化的稠密张量"""
    try:
        coo = sp.load_npz(npz_path).tocoo()
        N = coo.shape[0]
        A = torch.zeros((N, N), dtype=torch.float32)
        if device:
            A = A.to(device)

        # 填充值
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)).long()
        values = torch.from_numpy(coo.data.astype(np.float32))

        if device:
            indices = indices.to(device)
            values = values.to(device)

        A.index_put_((indices[0], indices[1]), values)

        # Row-Normalize (Random Walk)
        d = torch.clamp(A.sum(dim=1, keepdim=True), min=1e-6)
        A_rw = A / d
        return A_rw, A
    except Exception as e:
        print(f"Error loading matrix {npz_path}: {e}")
        return None, None


# =============================================================================
#  🌊 Dataset (适配分离后的新文件名)
# =============================================================================
class DrainageNPDataset(Dataset):
    def __init__(self, data_dir, task_name="Task"):
        super().__init__()
        # 匹配 {id}_X.npy 和 {id}_Y.npy
        self.files_X = sorted(glob.glob(os.path.join(data_dir, "*_X.npy")))
        self.files_Y = sorted(glob.glob(os.path.join(data_dir, "*_Y.npy")))

        assert len(self.files_X) == len(self.files_Y), "X and Y files mismatch!"
        assert len(self.files_X) > 0, f"No .npy files found in {data_dir}"

        self.cache_X = []
        self.cache_Y = []

        print(f"Loading {task_name} from {data_dir} ({len(self.files_X)} files)...")
        # 预加载到内存
        for fx, fy in tqdm(zip(self.files_X, self.files_Y), total=len(self.files_X), leave=False):
            bx = np.load(fx).astype(np.float32)
            by = np.load(fy).astype(np.float32)
            self.cache_X.append(bx)
            self.cache_Y.append(by)

        self.X_all = np.concatenate(self.cache_X, axis=0)
        self.Y_all = np.concatenate(self.cache_Y, axis=0)

    def __len__(self):
        return self.X_all.shape[0]

    def __getitem__(self, idx):
        return self.X_all[idx], self.Y_all[idx]


# =============================================================================
#  ⚖️ 2. 核心：智能多任务 Loss 函数 (含 SoftDTW)
# =============================================================================
class SoftDTW(torch.nn.Module):
    def __init__(self, gamma=1.0, normalize=False, backend='diagonal'):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize
        self.backend = backend

    def forward(self, x, y):
        # 强制 float32 避免 AMP 问题
        with torch.amp.autocast('cuda', enabled=False):
            x, y = x.float(), y.float()
            B, N, T = x.shape
            x = x.reshape(B * N, T, 1)
            y = y.reshape(B * N, T, 1)

            # 根据序列长度自动选择后端
            if T > 100 and self.backend == 'diagonal':
                loss = self._soft_dtw_vectorized(x, y, self.gamma)
            else:
                loss = self._soft_dtw(x, y, self.gamma)

            if self.normalize:
                loss_x = self._soft_dtw(x, x, self.gamma)
                loss_y = self._soft_dtw(y, y, self.gamma)
                loss = loss - 0.5 * (loss_x + loss_y)

            return loss.mean()

    def _soft_dtw(self, x, y, gamma):
        B, T, D = x.shape
        dist = torch.cdist(x, y, p=2).clamp(max=100) ** 2
        INF = 1e10
        prev_row = torch.full((B, T + 2), INF, device=x.device, dtype=torch.float32)
        prev_row[:, 0] = 0

        for i in range(1, T + 1):
            curr_row = torch.full((B, T + 2), INF, device=x.device, dtype=torch.float32)
            for j in range(1, T + 1):
                prev_vals = torch.stack([curr_row[:, j - 1], prev_row[:, j], prev_row[:, j - 1]], dim=1)
                soft_min = -gamma * torch.logsumexp(-prev_vals / gamma, dim=1)
                curr_row[:, j] = dist[:, i - 1, j - 1] + soft_min
            prev_row = curr_row
        return prev_row[:, T]

    def _soft_dtw_vectorized(self, x, y, gamma):
        B, T, D = x.shape
        dist = torch.cdist(x, y, p=2).clamp(max=100) ** 2
        INF = 1e10
        R = torch.full((B, T + 2, T + 2), INF, device=x.device, dtype=torch.float32)
        R[:, 0, 0] = 0
        for diag in range(2, 2 * T + 1):
            i_start = max(1, diag - T)
            i_end = min(T, diag - 1)
            i_indices = torch.arange(i_start, i_end + 1, device=x.device)
            j_indices = diag - i_indices
            valid_mask = (j_indices > 0) & (j_indices <= T)
            i_valid = i_indices[valid_mask]
            j_valid = j_indices[valid_mask]
            if len(i_valid) > 0:
                prev_vals = torch.stack(
                    [R[:, i_valid, j_valid - 1], R[:, i_valid - 1, j_valid], R[:, i_valid - 1, j_valid - 1]], dim=2)
                soft_min = -gamma * torch.logsumexp(-prev_vals / gamma, dim=2)
                R[:, i_valid, j_valid] = dist[:, i_valid - 1, j_valid - 1] + soft_min
        return R[:, T, T]


class HybridLoss(nn.Module):
    def __init__(self, target_idx, weight_overflow=10.0, use_dtw=False, dtw_weight=0.1, dtw_gamma=0.1):
        super().__init__()
        self.target_idx = target_idx
        self.mse = nn.MSELoss()
        self.use_dtw = use_dtw
        self.dtw_weight = dtw_weight
        if use_dtw:
            self.dtw_loss_fn = SoftDTW(gamma=dtw_gamma, normalize=False)

    def forward(self, pred, target, prob_gate=None):
        loss_total = 0.0
        # 遍历所有任务通道
        for i, task_id in enumerate(self.target_idx):
            p = pred[:, i, ...]
            t = target[:, i, ...]
            loss_task = self.mse(p, t)

            if self.use_dtw:
                dtw_val = self.dtw_loss_fn(p, t)
                loss_task += dtw_val * self.dtw_weight

            loss_total += loss_task
        return loss_total


# =============================================================================
#  🔄 3. 训练与评估循环
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device, config, scaler=None):
    model.train()
    total_loss = 0

    for X, Y in tqdm(dataloader, desc="Training", leave=False):
        X, Y = X.to(device), Y.to(device)
        X = X.permute(0, 3, 2, 1)
        target = Y[..., config.target_idx].permute(0, 3, 2, 1)

        optimizer.zero_grad()

        # 混合精度上下文
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            out_tuple = model(X, config.supports_list)
            if isinstance(out_tuple, tuple):
                output, prob_gate = out_tuple[0], out_tuple[1]
            else:
                output, prob_gate = out_tuple, None

            loss = criterion(output, target, prob_gate)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    sample_count = 0
    preds_list = []
    targets_list = []

    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        X = X.permute(0, 3, 2, 1)
        target = Y[..., config.target_idx].permute(0, 3, 2, 1)

        out_tuple = model(X, config.supports_list)
        if isinstance(out_tuple, tuple):
            output, prob_gate = out_tuple[0], out_tuple[1]
        else:
            output, prob_gate = out_tuple, None

        loss = criterion(output, target, prob_gate)
        batch_size = int(X.shape[0])
        total_loss += loss.item() * batch_size
        sample_count += batch_size

        preds_list.append(output.cpu().numpy())
        targets_list.append(target.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    if sample_count == 0:
        raise RuntimeError("Validation loader is empty")
    scaler = getattr(config, "_inverse_scaler", None)
    if scaler is None:
        scaler = JsonInverseScaler(config.scaler_config_path)
    horizon_r2, mean_r2 = validation_r2_by_task(
        preds, targets, config.target_idx, config.target_names, scaler
    )
    # Checkpoint selection is configuration-level: every included task
    # contributes its pooled normalized-space MSE to one common loss.
    loss_by_task = validation_loss_by_task(preds, targets)
    config_loss = configuration_validation_loss(loss_by_task)
    return config_loss, horizon_r2, mean_r2, loss_by_task


def _checkpoint_config(config):
    return {
        key: value for key, value in vars(config).items()
        if key != "supports_list" and not key.startswith("_")
    }


def _load_training_data(config, device):
    supports = []
    if "topo" in config.supports_to_use:
        adjacency, _ = load_coo_as_dense_rw(config.topo_matrix_path, device)
        supports.append(adjacency)
    if "rbf" in config.supports_to_use:
        adjacency, _ = load_coo_as_dense_rw(config.rbf_matrix_path, device)
        supports.append(adjacency)
    config.supports_list = supports

    train_dataset = DrainageNPDataset(config.train_data_dir, config.task_name)
    val_dataset = DrainageNPDataset(config.val_data_dir, config.task_name)
    loader_args = {"batch_size": config.batch_size, "pin_memory": True, "num_workers": 0}
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("Training or validation loader has zero batches")

    sample_x, sample_y = train_dataset[0]
    config.horizon = sample_y.shape[0]
    dimensions = {
        "nodes": sample_x.shape[1],
        "input_channels": sample_x.shape[-1],
        "output_channels": len(config.target_idx),
    }
    return train_loader, val_loader, dimensions


def _build_scheduler(config, optimizer):
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_epochs
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.min_lr,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs]
    )


def _build_training_state(config, device, dimensions):
    model_type = getattr(config, "model_type", "standard")
    model_class = GraphWaveNetMT
    if model_type == "fusion":
        model_class = GraphWaveNet_ViewFusion
    model = model_class(
        N=dimensions["nodes"],
        C_in=dimensions["input_channels"],
        C_out=dimensions["output_channels"],
        C_res=config.C_res,
        C_skip=config.C_skip,
        K=config.diffusion_steps_K,
        dilations=config.dilations,
        use_adaptive=config.use_adaptive,
        num_static_supports=len(config.supports_list),
        horizon=config.horizon,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = HybridLoss(
        target_idx=config.target_idx,
        use_dtw=config.use_dtw,
        dtw_weight=config.dtw_weight,
        dtw_gamma=config.dtw_gamma,
    ).to(device)
    config._inverse_scaler = JsonInverseScaler(config.scaler_config_path)
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": _build_scheduler(config, optimizer),
        "criterion": criterion,
        "amp_scaler": torch.amp.GradScaler("cuda") if config.use_amp else None,
    }


def _atomic_checkpoint(checkpoint, path):
    temporary_path = f"{path}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def _save_primary_checkpoint(config, state, result, best_state):
    checkpoint = {
        "epoch": result["epoch"],
        "model_state_dict": state["model"].state_dict(),
        "optimizer_state_dict": state["optimizer"].state_dict(),
        "config": _checkpoint_config(config),
        "seed": config.seed,
        "task_idx": list(config.target_idx),
        "task_names": [config.target_names[idx] for idx in config.target_idx],
        "selection_metric": (
            "minimum configuration-level normalized-space validation loss "
            "(sum of task-wise MSEs over H1-H6)"
        ),
        "best_val_loss": result["val_loss"],
        "best_val_loss_by_task": result["loss_by_task"],
        "best_val_r2_mean": result["mean_r2"],
        "best_val_r2_by_task": result["r2_scores"],
        "best_val_r2_by_horizon": result["horizon_r2"],
    }
    _atomic_checkpoint(checkpoint, os.path.join(config.save_dir, "best_model.pth"))
    best_state.update({
        "mean_r2": result["mean_r2"],
        "val_loss": result["val_loss"],
        "r2_scores": result["r2_scores"],
        "loss_by_task": result["loss_by_task"],
        "patience": 0,
    })


def _run_epoch(config, data, state, writer, epoch):
    train_loss = train_epoch(
        state["model"], data["train_loader"], state["optimizer"],
        state["criterion"], data["device"], config, state["amp_scaler"],
    )
    val_loss, horizon_r2, r2_scores, loss_by_task = evaluate(
        state["model"], data["val_loader"], state["criterion"],
        data["device"], config,
    )
    state["scheduler"].step()
    current_lr = state["optimizer"].param_groups[0]["lr"]
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", val_loss, epoch)
    log = f"Ep {epoch + 1:03d} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}"
    for output_idx, score in enumerate(r2_scores):
        target_name = config.target_names[config.target_idx[output_idx]]
        writer.add_scalar(f"Metrics/R2_{target_name}", score, epoch)
        log += f" | {target_name}: {score:.3f}"
    print(log)
    return {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "horizon_r2": horizon_r2,
        "r2_scores": r2_scores,
        "loss_by_task": loss_by_task,
        "learning_rate": current_lr,
    }


def _run_standard_epochs(config, data, state, writer):
    count = len(config.target_idx)
    best = {
        "mean_r2": float("-inf"),
        "val_loss": float("inf"),
        "r2_scores": [float("nan")] * count,
        "loss_by_task": [float("nan")] * count,
        "patience": 0,
    }
    for epoch in range(config.epochs):
        result = _run_epoch(config, data, state, writer, epoch)
        finite_scores = [score for score in result["r2_scores"] if np.isfinite(score)]
        result["mean_r2"] = float(np.mean(finite_scores)) if finite_scores else float("nan")
        if result["val_loss"] < best["val_loss"]:
            _save_primary_checkpoint(config, state, result, best)
        else:
            best["patience"] += 1
            if best["patience"] >= config.patience:
                print(f"⏹️ 早停触发 (Patience={config.patience} epochs)")
                break
    return best


def run_training_standard(config):
    from torch.utils.tensorboard import SummaryWriter

    print(f"\n--- 启动多步预测训练 (Horizon={getattr(config, 'horizon', 'Auto')}): {config.task_name} ---")
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config.save_dir, "logs"))
    train_loader, val_loader, dimensions = _load_training_data(config, device)
    state = _build_training_state(config, device, dimensions)
    data = {"device": device, "train_loader": train_loader, "val_loader": val_loader}
    try:
        best = _run_standard_epochs(config, data, state, writer)
    finally:
        writer.close()
    print(f"训练结束。最佳 Val Loss={best['val_loss']:.6f} (Val R2={best['mean_r2']:.6f})")
    return best["val_loss"], list(best["loss_by_task"])
