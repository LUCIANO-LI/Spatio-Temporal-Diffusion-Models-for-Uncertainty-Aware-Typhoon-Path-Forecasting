import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .components import (
    SinusoidalTimeEmbedding,
    AdaLayerNorm,
    PatchEmbed,
    Unpatchify,
    DiTBlock,
)

class ConditionEncoder(nn.Module):
    # 历史 ERA5 条件先过卷积，再切成 token
    def __init__(
        self,
        cond_channels: int = 60,
        d_model: int = 384,
        n_heads: int = 6,
        n_cond_layers: int = 3,
        ff_mult: int = 4,
        patch_size: int = 4,
        grid_size: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        num_patches = (grid_size // patch_size) ** 2

        self.local_conv = nn.Sequential(
            nn.Conv2d(cond_channels, d_model, 3, padding=1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.SiLU(),
        )
        self.patch_embed = PatchEmbed(d_model, d_model, patch_size=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_cond_layers):
            self.layers.append(CondSelfAttnBlock(d_model, n_heads, ff_mult, dropout))

        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        h = self.local_conv(condition)

        tokens = self.patch_embed(h)

        tokens = tokens + self.pos_embed

        for layer in self.layers:
            tokens = layer(tokens)

        return self.norm_out(tokens)

class CondSelfAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.dropout(self.proj(attn_out))

        x = x + self.ffn(self.norm2(x))

        return x

class ERA5DiT(nn.Module):

    def __init__(
        self,
        in_channels: int = 12,
        cond_channels: int = 60,
        d_model: int = 384,
        n_heads: int = 6,
        n_dit_layers: int = 12,
        n_cond_layers: int = 3,
        ff_mult: int = 4,
        patch_size: int = 4,
        grid_size: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_size = patch_size
        num_patches = (grid_size // patch_size) ** 2
        time_emb_dim = d_model

        self.time_emb = SinusoidalTimeEmbedding(d_model, time_emb_dim)

        self.cond_encoder = ConditionEncoder(
            cond_channels=cond_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_cond_layers=n_cond_layers,
            ff_mult=ff_mult,
            patch_size=patch_size,
            grid_size=grid_size,
            dropout=dropout,
        )

        self.patch_embed = PatchEmbed(in_channels, d_model, patch_size=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_mult=ff_mult,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
            )
            for _ in range(n_dit_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * d_model),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.final_linear = nn.Linear(d_model, in_channels * patch_size * patch_size)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

        self.unpatchify = Unpatchify(in_channels, patch_size, grid_size)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_emb(t)

        # 条件分支只编码历史场，不参与加噪
        cond_tokens = self.cond_encoder(condition)

        x = self.patch_embed(x_noisy)
        x = x + self.pos_embed

        for block in self.dit_blocks:
            x = block(x, time_emb, cond_tokens)

        final_params = self.final_adaLN(time_emb).unsqueeze(1)
        gamma, beta = final_params.chunk(2, dim=-1)
        x = (1 + gamma) * self.final_norm(x) + beta
        x = self.final_linear(x)

        x = self.unpatchify(x)

        return x

class DiffusionScheduler:

    def __init__(
        self,
        num_steps: int = 1000,
        schedule: str = "cosine",
        s: float = 0.008,
        ddim_steps: int = 50,
        clamp_range: Tuple[float, float] = (-5.0, 5.0),
        prediction_type: str = "v",
    ):
        self.num_steps = num_steps
        self.ddim_steps = ddim_steps
        self.clamp_range = clamp_range
        self.prediction_type = prediction_type

        if schedule == "cosine":
            # cosine schedule 后期噪声变化更平滑
            steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
            f_t = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = f_t / f_t[0]
            alphas_cumprod = alphas_cumprod.clamp(min=1e-8, max=1.0)
        elif schedule == "linear":
            beta_start, beta_end = 1e-4, 0.02
            betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod])
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas_cumprod = alphas_cumprod.float()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.ddim_timesteps = self._make_ddim_timesteps(ddim_steps, num_steps)

    def _make_ddim_timesteps(self, ddim_steps: int, total_steps: int) -> torch.Tensor:
        step_size = total_steps // ddim_steps
        timesteps = torch.arange(0, total_steps, step_size)
        return timesteps.flip(0)

    def to(self, device: torch.device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        return self

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)

        # 按时间步把真实场混成 x_t
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)

        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        # eps 模式后期会除以 sqrt_alpha，数值更敏感
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        x0 = (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha.clamp(min=1e-8)
        return x0

    def compute_v_target(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        # v 模式恢复 x0 不用做除法，z 场更稳
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_start

    def predict_x0_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v

    def predict_eps_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_one_minus_alpha * x_t + sqrt_alpha * v

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
        eta: float = 0.0,
        z_channel_indices: Optional[List[int]] = None,
        z_clamp_range: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        B = shape[0]
        x = torch.randn(shape, device=device)

        # DDIM 从大噪声往小噪声走
        timesteps = self.ddim_timesteps

        for i in tqdm(range(len(timesteps)), desc="DDIM采样", unit="步", leave=False):
            t_current = timesteps[i]
            t_batch = torch.full((B,), t_current, device=device, dtype=torch.long)

            model_output = model(x, t_batch, condition)

            if self.prediction_type == "v":
                x0_pred = self.predict_x0_from_v(x, t_batch, model_output)
                eps_pred = self.predict_eps_from_v(x, t_batch, model_output)
            else:
                x0_pred = self.predict_x0_from_eps(x, t_batch, model_output)
                eps_pred = model_output
            x0_pred = x0_pred.clamp(*self.clamp_range)
            if z_channel_indices and z_clamp_range:
                # z 场漂移会影响自回归，单独收紧范围
                x0_pred[:, z_channel_indices] = x0_pred[:, z_channel_indices].clamp(*z_clamp_range)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = self.alphas_cumprod[t_current + 1]
                alpha_next = self.alphas_cumprod[t_next + 1]

                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)
                )

                pred_dir = torch.sqrt(1 - alpha_next - sigma ** 2) * eps_pred

                x = torch.sqrt(alpha_next) * x0_pred + pred_dir
                if sigma > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x

class DivergenceLoss(nn.Module):

    def __init__(self, wind_pairs: List[Tuple[int, int]]):
        super().__init__()
        self.wind_pairs = wind_pairs

    def forward(self, x0_pred: torch.Tensor) -> torch.Tensor:
        total_div = 0.0
        count = 0

        for u_idx, v_idx in self.wind_pairs:
            u = x0_pred[:, u_idx]
            v = x0_pred[:, v_idx]

            # 简单有限差分，只当弱正则
            du_dx = u[:, :, 1:] - u[:, :, :-1]
            dv_dy = v[:, 1:, :] - v[:, :-1, :]

            du_dx = du_dx[:, :-1, :]
            dv_dy = dv_dy[:, :, :-1]

            divergence = du_dx + dv_dy
            total_div = total_div + (divergence ** 2).mean()
            count += 1

        return total_div / max(count, 1)

class VorticityCurlLoss(nn.Module):

    def __init__(self, data_cfg):
        super().__init__()
        n_pl = len(data_cfg.pressure_levels)
        pl_vars = data_cfg.pressure_level_vars

        u_idx_in_pl = pl_vars.index("u")
        v_idx_in_pl = pl_vars.index("v")
        vo_idx_in_pl = pl_vars.index("vo")

        triplets = []
        for t_step in range(data_cfg.forecast_steps):
            base = t_step * data_cfg.num_channels
            for lev in range(n_pl):
                u_ch = base + u_idx_in_pl * n_pl + lev
                v_ch = base + v_idx_in_pl * n_pl + lev
                vo_ch = base + vo_idx_in_pl * n_pl + lev
                triplets.append((u_ch, v_ch, vo_ch))

        self.triplets = triplets

    def forward(self, x0_pred: torch.Tensor) -> torch.Tensor:
        # 用风场旋度约束 vo，不反归一化
        total_loss = 0.0
        count = 0

        for u_ch, v_ch, vo_ch in self.triplets:
            u = x0_pred[:, u_ch]
            v = x0_pred[:, v_ch]
            vo = x0_pred[:, vo_ch]

            dv_dx = v[:, :, 1:] - v[:, :, :-1]
            du_dy = u[:, 1:, :] - u[:, :-1, :]

            dv_dx = dv_dx[:, :-1, :]
            du_dy = du_dy[:, :, :-1]

            curl_from_wind = dv_dx - du_dy

            vo_inner = vo[:, :-1, :-1]

            total_loss = total_loss + F.mse_loss(vo_inner, curl_from_wind)
            count += 1

        return total_loss / max(count, 1)

class ChannelWeightedMSE(nn.Module):
    # z 通道权重大一点，避免高度场被风场 loss 淹掉

    def __init__(self, channel_weights: torch.Tensor):
        super().__init__()
        w = channel_weights / channel_weights.mean()
        self.register_buffer("weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = pred.shape[1]
        w = self.weights[:C].reshape(1, C, 1, 1)
        return (w * (pred - target) ** 2).mean()

class ERA5DiffusionModel(nn.Module):

    def __init__(self, model_cfg, data_cfg, train_cfg=None):
        super().__init__()

        self.dit = ERA5DiT(
            in_channels=model_cfg.in_channels,
            cond_channels=model_cfg.cond_channels,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_dit_layers=model_cfg.n_dit_layers,
            n_cond_layers=model_cfg.n_cond_layers,
            ff_mult=model_cfg.ff_mult,
            patch_size=model_cfg.patch_size,
            grid_size=data_cfg.grid_size,
            dropout=model_cfg.dropout,
        )

        self.scheduler = DiffusionScheduler(
            num_steps=model_cfg.num_diffusion_steps,
            schedule=model_cfg.noise_schedule,
            ddim_steps=model_cfg.ddim_sampling_steps,
            prediction_type=getattr(model_cfg, 'prediction_type', 'eps'),
        )

        wind_pairs = data_cfg.get_wind_channel_indices()
        self.div_loss = DivergenceLoss(wind_pairs)

        if "vo" in data_cfg.pressure_level_vars:
            self.curl_loss = VorticityCurlLoss(data_cfg)
        else:
            self.curl_loss = None

        if train_cfg is not None and train_cfg.use_channel_weights:
            weights = torch.tensor(train_cfg.channel_weights, dtype=torch.float32)
            if data_cfg.forecast_steps > 1:
                per_step = weights[:data_cfg.num_channels]
                weights = per_step.repeat(data_cfg.forecast_steps)
            self.channel_mse = ChannelWeightedMSE(weights)
        else:
            self.channel_mse = None

        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

        if "z" in data_cfg.pressure_level_vars:
            z_var_idx = data_cfg.pressure_level_vars.index("z")
            n_levels = len(data_cfg.pressure_levels)
            self.z_channel_indices = list(range(
                z_var_idx * n_levels, (z_var_idx + 1) * n_levels
            ))
        else:
            self.z_channel_indices = []

    def forward(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        device = target.device
        B = target.shape[0]

        self.scheduler.to(device)

        # 每个样本独立抽 t，避免只适应固定噪声强度
        t = torch.randint(0, self.model_cfg.num_diffusion_steps, (B,), device=device)

        x_noisy, noise = self.scheduler.q_sample(target, t)

        model_output = self.dit(x_noisy, t, condition)

        prediction_type = self.scheduler.prediction_type
        if prediction_type == "v":
            # v 模式的训练目标不是 noise 本身
            v_target = self.scheduler.compute_v_target(target, t, noise)
            if self.channel_mse is not None:
                loss_mse = self.channel_mse(model_output, v_target)
            else:
                loss_mse = F.mse_loss(model_output, v_target)
            x0_pred = self.scheduler.predict_x0_from_v(x_noisy, t, model_output)
        else:
            if self.channel_mse is not None:
                loss_mse = self.channel_mse(model_output, noise)
            else:
                loss_mse = F.mse_loss(model_output, noise)
            x0_pred = self.scheduler.predict_x0_from_eps(x_noisy, t, model_output)

        # 物理项只做弱约束，异常尖峰先截断
        loss_div_raw = self.div_loss(x0_pred)
        loss_div = torch.clamp(loss_div_raw, max=10.0)

        loss_curl_raw = self.curl_loss(x0_pred) if self.curl_loss is not None else torch.tensor(0.0, device=device)
        loss_curl = torch.clamp(loss_curl_raw, max=10.0) if self.curl_loss is not None else loss_curl_raw

        return {
            "loss_mse": loss_mse,
            "loss_div": loss_div,
            "loss_curl": loss_curl,
            "eps_pred": model_output,
            "eps_true": noise,
            "x0_pred": x0_pred,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        device: torch.device,
        z_clamp_range: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:

        self.scheduler.to(device)
        B = condition.shape[0]
        shape = (B, self.model_cfg.in_channels, self.data_cfg.grid_size, self.data_cfg.grid_size)
        return self.scheduler.ddim_sample(
            self.dit, condition, shape, device,
            z_channel_indices=self.z_channel_indices if self.z_channel_indices else None,
            z_clamp_range=z_clamp_range,
        )

    @torch.no_grad()
    def fast_sample(
        self,
        condition: torch.Tensor,
        device: torch.device,
        ddim_steps: int = 10,
    ) -> torch.Tensor:

        self.scheduler.to(device)
        B = condition.shape[0]
        H = W = self.data_cfg.grid_size
        in_ch = self.model_cfg.in_channels
        shape = (B, in_ch, H, W)

        total_steps = self.scheduler.num_steps
        step_size = total_steps // ddim_steps
        # scheduled sampling 用快采样，牺牲精度换速度
        timesteps = torch.arange(0, total_steps, step_size, device=device).flip(0)

        x = torch.randn(shape, device=device)

        for i in range(len(timesteps)):
            t_current = timesteps[i]
            t_batch = torch.full((B,), t_current, device=device, dtype=torch.long)

            model_output = self.dit(x, t_batch, condition)

            if self.scheduler.prediction_type == "v":
                x0_pred = self.scheduler.predict_x0_from_v(x, t_batch, model_output)
                eps_pred = self.scheduler.predict_eps_from_v(x, t_batch, model_output)
            else:
                x0_pred = self.scheduler.predict_x0_from_eps(x, t_batch, model_output)
                eps_pred = model_output
            x0_pred = x0_pred.clamp(*self.scheduler.clamp_range)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = self.scheduler.alphas_cumprod[t_current + 1]
                alpha_next = self.scheduler.alphas_cumprod[t_next + 1]
                pred_dir = torch.sqrt(1 - alpha_next) * eps_pred
                x = torch.sqrt(alpha_next) * x0_pred + pred_dir
            else:
                x = x0_pred

        return x
