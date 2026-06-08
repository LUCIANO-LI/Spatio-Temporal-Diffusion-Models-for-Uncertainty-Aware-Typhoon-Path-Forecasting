from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_ROOT = os.environ.get(
    "TYPHOON_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "Typhoon_data_final"),
)

@dataclass
class DataConfig:
    data_root: str = DEFAULT_DATA_ROOT
    stats_csv: str = ""

    grid_size: int = 40

    pressure_level_vars: List[str] = field(
        default_factory=lambda: ["u", "v", "z"]
    )
    surface_vars: List[str] = field(
        default_factory=lambda: []
    )
    pressure_levels: List[int] = field(
        default_factory=lambda: [850, 500, 250]
    )

    history_steps: int = 5
    forecast_steps: int = 1
    time_interval_hours: int = 3

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    num_workers: int = 16
    prefetch_factor: int = 4
    pin_memory: bool = True

    preprocessed_dir: Optional[str] = None

    norm_stats_path: str = ""

    def __post_init__(self):
        if not self.stats_csv:
            self.stats_csv = os.path.join(
                self.data_root, "typhoon_organization_stats_1950_2021.csv"
            )

    @property
    def num_pressure_level_channels(self) -> int:
        return len(self.pressure_level_vars) * len(self.pressure_levels)

    @property
    def num_surface_channels(self) -> int:
        return len(self.surface_vars)

    @property
    def num_channels(self) -> int:
        return self.num_pressure_level_channels + self.num_surface_channels

    @property
    def condition_channels(self) -> int:
        return self.num_channels * self.history_steps

    @property
    def target_channels(self) -> int:
        return self.num_channels * self.forecast_steps

    def get_wind_channel_indices(self) -> List[Tuple[int, int]]:
        pairs = []
        n_pl = len(self.pressure_levels)

        for t_step in range(self.forecast_steps):
            base = t_step * self.num_channels
            u_idx_in_pl = self.pressure_level_vars.index("u")
            v_idx_in_pl = self.pressure_level_vars.index("v")
            for lev in range(n_pl):
                u_ch = base + u_idx_in_pl * n_pl + lev
                v_ch = base + v_idx_in_pl * n_pl + lev
                pairs.append((u_ch, v_ch))

            if "u10m" in self.surface_vars and "v10m" in self.surface_vars:
                u10_idx = self.surface_vars.index("u10m")
                v10_idx = self.surface_vars.index("v10m")
                u10_ch = base + self.num_pressure_level_channels + u10_idx
                v10_ch = base + self.num_pressure_level_channels + v10_idx
                pairs.append((u10_ch, v10_ch))

        return pairs

@dataclass
class ModelConfig:
    d_model: int = 384
    n_heads: int = 6
    n_dit_layers: int = 12
    n_cond_layers: int = 3
    ff_mult: int = 4
    patch_size: int = 4
    dropout: float = 0.1

    num_diffusion_steps: int = 1000
    noise_schedule: str = "cosine"
    ddim_sampling_steps: int = 50
    prediction_type: str = "eps"

    in_channels: int = 9
    cond_channels: int = 45

@dataclass
class TrainConfig:
    batch_size: int = 48
    gradient_accumulation_steps: int = 1
    max_epochs: int = 2000

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    warmup_steps: int = 200
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6

    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    ema_decay: float = 0.999
    ema_start_step: int = 0

    max_grad_norm: float = 1.0

    physics_loss_weight: float = 0.0

    vorticity_loss_weight: float = 0.0

    use_channel_weights: bool = True
    # z 通道权重大一点，避免高度场被风场 loss 淹掉
    channel_weights: Tuple[float, ...] = (
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.5,
    )

    condition_noise_sigma: float = 0.30
    condition_noise_rampup_epochs: int = 100
    condition_noise_prob: float = 0.5
    condition_noise_spatial_smooth: bool = True
    condition_noise_smooth_kernel: int = 5

    # 训练时少量喂回模型自己的预测
    scheduled_sampling_enabled: bool = True
    scheduled_sampling_start_epoch: int = 50
    scheduled_sampling_max_prob: float = 0.3
    scheduled_sampling_rampup_epochs: int = 100
    scheduled_sampling_max_replace: int = 2
    scheduled_sampling_ddim_steps: int = 10

    eval_every: int = 10
    early_stopping_patience: int = 50
    save_top_k: int = 3

    log_every: int = 20
    use_tensorboard: bool = True

    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    use_compile: bool = False

    cudnn_benchmark: bool = True

    seed: int = 42

@dataclass
class InferenceConfig:
    ddim_steps: int = 50
    clamp_range: Tuple[float, float] = (-5.0, 5.0)
    # z 场单独收紧，滚动预测时更稳
    z_clamp_range: Tuple[float, float] = (-3.0, 3.0)
    autoregressive_steps: int = 24
    autoregressive_noise_sigma: float = 0.05
    ar_ensemble_per_step: int = 5
    ensemble_size: int = 10
    checkpoint_path: str = ""
    output_dir: str = "outputs"
    device: str = "cuda"

def get_config(
    data_root: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[DataConfig, ModelConfig, TrainConfig, InferenceConfig]:
    data_cfg = DataConfig()
    if data_root:
        data_cfg.data_root = data_root
        data_cfg.__post_init__()

    model_cfg = ModelConfig(
        in_channels=data_cfg.target_channels,
        cond_channels=data_cfg.condition_channels,
    )

    train_cfg = TrainConfig()
    if checkpoint_dir:
        train_cfg.checkpoint_dir = checkpoint_dir

    infer_cfg = InferenceConfig()

    return data_cfg, model_cfg, train_cfg, infer_cfg
