"""Configuration for node-level drainage forecasting experiments."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("NODE_DATA_ROOT", PROJECT_ROOT / "DATA"))
STATIC_ROOT = Path(os.environ.get("NODE_STATIC_ROOT", PROJECT_ROOT / "static_matrix"))
CHECKPOINT_ROOT = Path(
    os.environ.get("NODE_CHECKPOINT_ROOT", PROJECT_ROOT / "node_related" / "checkpoint" / "NODES")
)

class ConfigBase:
    """基础配置"""

    def __init__(self):
        self.task_name = "Base"
        # --- 模型架构 ---
        self.num_blocks = 6
        self.C_res = 64
        self.C_skip = 128
        self.diffusion_steps_K = 2
        self.dilations = [1, 2, 4, 8,1,2]  # 标准 TCN 感受野

        self.model_type = "standard"  # 或 "gated"

        # --- Loss 配置 ---
        self.use_dtw = False
        self.dtw_weight = 0.1
        self.dtw_gamma = 0.1

        # --- 训练参数 ---
        self.epochs = 1000 # 根据数据量调整
        self.batch_size = 64*4
        self.lr = 1e-4
        self.min_lr = 1e-6
        self.weight_decay = 1e-4
        self.patience = 20
        self.clip = 1
        self.seed = 3407
        self.use_amp = False  # 开启混合精度加速

        self.warmup_epochs = 20
        self.horizon = 6

        # 占位符
        self.supports_to_use = []
        self.use_adaptive = False
        self.target_idx = [0, 1, 2, 3, 4]  # 默认为5任务


class ConfigNode(ConfigBase):
    """节点(Node)任务配置"""

    def __init__(self):
        super().__init__()
        self.task_name = "Node_5Task"
        # NODE_DATA_ROOT points to DATA/; override it for HPC runs.
        node_data_root = DATA_ROOT / "npy_out_separated" / "NODES"
        self.train_data_dir = str(node_data_root / "train")
        self.val_data_dir = str(node_data_root / "val")
        self.test_data_dir = str(node_data_root / "test")
        self.scaler_config_path = str(DATA_ROOT / "scalers" / "scalers_config.json")

        # 邻接矩阵路径
        self.topo_matrix_path = str(STATIC_ROOT / "nodes_topo.npz")
        self.rbf_matrix_path = str(STATIC_ROOT / "nodes_rbf.npz")

        self.save_dir = str(CHECKPOINT_ROOT)

        # 5个节点变量: ["invert_depth", "hydraulic_head", "lateral_inflow", "total_inflow", "flooding_losses"]
        self.target_names = ["Depth", "Head", "LatIn", "TotIn", "Flood"]
