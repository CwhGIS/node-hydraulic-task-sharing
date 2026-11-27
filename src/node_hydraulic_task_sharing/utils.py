# utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _as_row_rw(A: torch.Tensor) -> torch.Tensor:
    d = torch.clamp(A.sum(dim=1, keepdim=True), min=1e-6)
    return A / d


class AdaptiveAdjacency(nn.Module):
    def __init__(self, N: int, rank: int = 16):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(N, rank) * 0.1)
        self.E2 = nn.Parameter(torch.randn(N, rank) * 0.1)

    def forward(self) -> torch.Tensor:
        A = torch.relu(self.E1 @ self.E2.t())
        return torch.softmax(A, dim=1)


class DiffusionConv(nn.Module):
    def __init__(self, c_in, c_out, K, num_supports):
        super().__init__()
        self.K = K
        self.mlp = nn.Conv2d((2 * K) * num_supports * c_in, c_out, kernel_size=(1, 1))

    def _nconv(self, x, A_rw):
        # x: [B, C, N, T], A_rw: [N, N] -> [B, C, N, T]
        return torch.einsum('bcnt,nm->bcmt', x, A_rw)

    def forward(self, x, supports):
        outs = []
        for A in supports:
            A_rw = _as_row_rw(A)
            AT_rw = _as_row_rw(A.t())
            x_f, x_b = x, x
            for _ in range(self.K):
                x_f = self._nconv(x_f, A_rw)
                x_b = self._nconv(x_b, AT_rw)
                outs.append(x_f)
                outs.append(x_b)
        x_cat = torch.cat(outs, dim=1)
        return self.mlp(x_cat)


class GWBlock(nn.Module):
    def __init__(self, c_in, c_res, c_skip, K, num_supports, dilation):
        super().__init__()
        self.dilation = dilation
        # TCN: (1, 3) 卷积核
        self.tconv = nn.Conv2d(c_in, 2 * c_res, kernel_size=(1, 3), dilation=(1, dilation))
        self.dconv = DiffusionConv(c_res, c_res, K, num_supports)
        self.res_proj = nn.Conv2d(c_res, c_in, kernel_size=(1, 1))
        self.skip_proj = nn.Conv2d(c_res, c_skip, kernel_size=(1, 1))

        # ⭐ 新增：归一化层，加速收敛
        self.bn = nn.BatchNorm2d(c_res)

    def forward(self, x, supports):
        # x: [B, C_in, N, T]
        x_in = x

        # 因果卷积 Padding
        pad = (3 - 1) * self.dilation
        x_padded = F.pad(x, (pad, 0, 0, 0))

        # 1. 门控 TCN
        t = self.tconv(x_padded)
        P, Q = torch.chunk(t, 2, dim=1)
        # Gated Tanh Unit (GTU)
        h = torch.tanh(P) * torch.sigmoid(Q)

        # 2. 图卷积 + BN
        h = self.dconv(h, supports)
        h = self.bn(h)

        # 3. 残差与跳跃连接
        res = self.res_proj(h)
        skip = self.skip_proj(h)

        # 残差连接 (x_in + res)
        return x_in + res, skip


class GraphWaveNetMT(nn.Module):
    def __init__(self, N, C_in, C_out,
                 C_res=32, C_skip=64,
                 K=2,
                 # ⭐ 默认两组 1,2,4,8
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 use_adaptive=True, num_static_supports=1,
                 horizon=1):
        super().__init__()
        self.horizon = horizon
        self.use_adp = use_adaptive
        self.adp = AdaptiveAdjacency(N) if use_adaptive else None

        # 输入投影：将原始特征维度映射到残差维度 C_res
        self.input_proj = nn.Conv2d(C_in, C_res, kernel_size=(1, 1))

        self.blocks = nn.ModuleList([
            GWBlock(c_in=C_res, c_res=C_res, c_skip=C_skip, K=K,
                    num_supports=(num_static_supports + (1 if use_adaptive else 0)),
                    dilation=d) for d in dilations
        ])

        # ⭐ 核心修复 1: 全局时间平均池化
        # [B, C, N, T] -> [B, C, N, 1]
        # 这样能利用整个序列的信息，而不仅仅是最后一步
        self.temporal_pool = nn.AdaptiveAvgPool2d((None, 1))

        # ⭐ 核心修复 2: 更深的输出层 (MLP)
        self.out = nn.Sequential(
            nn.Conv2d(C_skip, C_skip, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Conv2d(C_skip, C_skip // 2, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(C_skip // 2, C_out * horizon, kernel_size=(1, 1))
        )

    def forward(self, x, static_supports, return_adp_matrix=False):
        # x: [B, C_in, N, T]
        supports = list(static_supports)
        adp_matrix = None
        if self.use_adp and self.adp is not None:
            adp_matrix = self.adp()
            supports.append(adp_matrix)

        # 1. 投影到隐空间
        x = self.input_proj(x)

        skip_total = None
        for blk in self.blocks:
            x, skip = blk(x, supports)
            # 累加跳跃连接
            skip_total = skip if skip_total is None else skip_total + skip

        # 2. 聚合时间维度信息
        # skip_total: [B, C_skip, N, T]
        # pooled: [B, C_skip, N, 1]
        feature_pooled = self.temporal_pool(skip_total)

        # 3. 输出映射
        y = self.out(feature_pooled)  # [B, C_out*H, N, 1]

        # 4. 重塑维度 -> [B, C_out, N, Horizon]
        B, _, N, _ = y.shape
        y = y.view(B, -1, self.horizon, N).permute(0, 1, 3, 2)

        if return_adp_matrix:
            return y, adp_matrix
        else:
            return y



# =============================================================================
# ⭐ View-Gated Fusion Implementation
# 核心思想: 使用门控机制替代 Concat，实现多视图的自适应加权融合
# =============================================================================
class ViewGatedDiffusionConv(nn.Module):
    """
    稳健型视图融合卷积 (Stabilized View-Gated Fusion)
    策略:
    1. Bottleneck: C -> C/4 -> 2KC (保持低参数量)
    2. Softmax Competition: 跨视图竞争，防止噪声叠加 (缓解多视图负迁移)
    3. Input: 使用 x 生成门控 (Query-Guide), 保持计算高效
    """

    def __init__(self, c_in, c_out, K, num_supports, reduction_ratio=4):
        super().__init__()
        self.K = K
        self.num_supports = num_supports

        # 扩散特征通道: 2方向 * K步 * C
        # 64 * 3 * 2 = 384
        self.diffusion_channels = 2 * K * c_in

        # 瓶颈层维度: 64 / 4 = 16
        hidden_dim = max(4, c_in // reduction_ratio)

        # 门控生成器 (Gate Generators)
        # 这里的输出不再经过 Sigmoid，而是作为 Logits 进 Softmax
        self.gate_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_in, hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, self.diffusion_channels, kernel_size=1)
            )
            for _ in range(num_supports)
        ])

        # 输出投影: 384 -> 64
        self.mlp = nn.Conv2d(self.diffusion_channels, c_out, kernel_size=1)

    def _nconv(self, x, A_rw):
        # x: [B, C, N, T], A: [N, N] -> [B, C, N, T]
        return torch.einsum('bcnt,nm->bcmt', x, A_rw)

    def forward(self, x, supports):
        # x: [B, C, N, T]
        assert len(supports) == len(self.gate_generators)

        # 1. 预计算所有 Logits (Raw Gates)
        # logits_list: List of [B, 2KC, N, T]
        logits_list = [gen(x) for gen in self.gate_generators]

        # 2. 跨视图 Softmax (Competition) ⭐ 关键修改 ⭐
        # 堆叠 -> [S, B, 2KC, N, T]
        stacked_logits = torch.stack(logits_list, dim=0)
        # 在视图维度(dim=0)做 Softmax
        # 含义: 对每一个(batch, channel, node, time)点，所有视图的权重之和为 1
        weights = torch.softmax(stacked_logits, dim=0)

        fused_feature = 0.0

        for i, A in enumerate(supports):
            # --- A. 计算扩散特征 ---
            A_rw = _as_row_rw(A)
            AT_rw = _as_row_rw(A.t())

            x_f, x_b = x, x
            out_i = []
            for _ in range(self.K):
                x_f = self._nconv(x_f, A_rw)
                x_b = self._nconv(x_b, AT_rw)
                out_i.append(x_f)
                out_i.append(x_b)

            # h_i: [B, 2KC, N, T]
            h_i = torch.cat(out_i, dim=1)

            # --- B. 加权融合 ---
            # weights[i]: [B, 2KC, N, T]
            fused_feature = fused_feature + (h_i * weights[i])

        # --- C. 输出投影 ---
        return self.mlp(fused_feature)


class GWBlock_ViewFusion(nn.Module):
    """
    修改版的 GWBlock，使用 ViewGatedDiffusionConv 替代普通的 DiffusionConv
    """

    def __init__(self, c_in, c_res, c_skip, K, num_supports, dilation):
        super().__init__()
        self.dilation = dilation
        self.tconv = nn.Conv2d(c_in, 2 * c_res, kernel_size=(1, 3), dilation=(1, dilation))

        # ⭐ 使用新的视图融合卷积
        self.dconv = ViewGatedDiffusionConv(c_res, c_res, K, num_supports)

        self.res_proj = nn.Conv2d(c_res, c_in, kernel_size=(1, 1))
        self.skip_proj = nn.Conv2d(c_res, c_skip, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(c_res)

    def forward(self, x, supports):
        x_in = x
        pad = (3 - 1) * self.dilation
        x_padded = F.pad(x, (pad, 0, 0, 0))

        t = self.tconv(x_padded)
        P, Q = torch.chunk(t, 2, dim=1)
        h = torch.tanh(P) * torch.sigmoid(Q)

        # 调用带门控融合的图卷积
        h = self.dconv(h, supports)
        h = self.bn(h)

        res = self.res_proj(h)
        skip = self.skip_proj(h)
        return x_in + res, skip


class GraphWaveNet_ViewFusion(nn.Module):
    """
    支持视图门控融合的 GraphWaveNet
    """

    def __init__(self, N, C_in, C_out,
                 C_res=32, C_skip=64,
                 K=2, dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 use_adaptive=True, num_static_supports=1,
                 horizon=1):
        super().__init__()
        self.horizon = horizon
        self.use_adp = use_adaptive
        self.adp = AdaptiveAdjacency(N) if use_adaptive else None

        # 计算总的视图数量 (Static + Adaptive)
        total_supports = num_static_supports + (1 if use_adaptive else 0)

        self.input_proj = nn.Conv2d(C_in, C_res, kernel_size=(1, 1))

        # ⭐ 使用 GWBlock_ViewFusion
        self.blocks = nn.ModuleList([
            GWBlock_ViewFusion(c_in=C_res, c_res=C_res, c_skip=C_skip, K=K,
                               num_supports=total_supports,
                               dilation=d) for d in dilations
        ])

        self.temporal_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.out = nn.Sequential(
            nn.Conv2d(C_skip, C_skip, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(C_skip, C_skip // 2, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(C_skip // 2, C_out * horizon, kernel_size=(1, 1))
        )

    def forward(self, x, static_supports, return_adp_matrix=False):
        supports = list(static_supports)
        adp_matrix = None
        if self.use_adp and self.adp is not None:
            adp_matrix = self.adp()
            supports.append(adp_matrix)

        x = self.input_proj(x)
        skip_total = None
        for blk in self.blocks:
            x, skip = blk(x, supports)
            skip_total = skip if skip_total is None else skip_total + skip

        feature_pooled = self.temporal_pool(skip_total)
        y = self.out(feature_pooled)

        B, _, N, _ = y.shape
        y = y.view(B, -1, self.horizon, N).permute(0, 1, 3, 2)

        if return_adp_matrix:
            return y, adp_matrix
        else:
            return y
