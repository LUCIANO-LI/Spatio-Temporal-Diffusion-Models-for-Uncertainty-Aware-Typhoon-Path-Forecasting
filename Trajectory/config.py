from dataclasses import dataclass, field
from typing import List, Tuple
import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.environ.get(
    "TYPHOON_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "Typhoon_data_final"),
)

@dataclass
class DataConfig:
    csv_path: str = "processed_typhoon_tracks.csv"
    era5_dir: str = DEFAULT_DATA_ROOT

    lat_range: Tuple[float, float] = (0.0, 60.0)
    lon_range: Tuple[float, float] = (95.0, 185.0)

    grid_height: int = 40
    grid_width: int = 40

    time_resolution_hours: int = 3

    pressure_levels: List[int] = field(default_factory=lambda: [850, 500, 250])

    era5_3d_vars: List[str] = field(default_factory=lambda: ['u', 'v', 'z'])

    era5_2d_vars: List[str] = field(default_factory=lambda: [])

@dataclass
class ModelConfig:
    t_history: int = 16
    t_future_era5: int = 24
    t_future: int = 24

    coord_dim: int = 2
    output_dim: int = 2

    cond_feature_dim: int = 32

    era5_channels: int = 9
    era5_base_channels: int = 64
    era5_out_dim: int = 256

    coord_embed_dim: int = 128

    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 8
    transformer_ff_dim: int = 2048
    dropout: float = 0.1

    use_heatmap_head: bool = False
    heatmap_loss_weight: float = 0.0
    gaussian_sigma: float = 2.0

    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"

@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    num_epochs: int = 200

    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = False

    use_sample_weights: bool = True
    real_sample_weight: float = 1.0
    interp_sample_weight: float = 0.5

    early_stopping: bool = True
    patience: int = 25

    save_interval: int = 10
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints/"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_by: str = "storm_id"
    
    lr_scheduler: str = "cosine_warmup"
    warmup_epochs: int = 20
    
    gradient_accumulation_steps: int = 1
    
    min_typhoon_duration_hours: int = 120

@dataclass
class SampleConfig:
    num_samples: int = 10
    use_ddim: bool = True
    ddim_steps: int = 50
    eta: float = 0.0
    guidance_scale: float = 1.0

data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
sample_cfg = SampleConfig()
