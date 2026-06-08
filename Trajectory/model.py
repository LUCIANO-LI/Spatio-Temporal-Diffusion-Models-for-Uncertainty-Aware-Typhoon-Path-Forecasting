import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from config import model_cfg, data_cfg

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class LeadTimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_lead_time: int = 24):
        super().__init__()
        self.embedding = nn.Embedding(max_lead_time, d_model)
    
    def forward(self, t_future: int, batch_size: int, device: torch.device) -> torch.Tensor:
        lead_times = torch.arange(t_future, device=device).unsqueeze(0).expand(batch_size, -1)
        return self.embedding(lead_times)

class PhysicsEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        base_channels: int = 64,
        out_dim: int = 256
    ):
        super().__init__()

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
        )
        
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
        )
        
        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
        )
        
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(base_channels * 4, 1, kernel_size=1),
        )

        self.proj = nn.Sequential(
            nn.Linear(base_channels * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, era5: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = era5.shape
        
        x = era5.permute(0, 2, 1, 3, 4)
        
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)

        attn_logits = self.spatial_attn(x)
        attn_weights = torch.softmax(
            attn_logits.flatten(3), dim=-1
        ).unflatten(3, attn_logits.shape[3:])

        x = (x * attn_weights).sum(dim=(-2, -1))

        x = x.permute(0, 2, 1)
        
        x = self.proj(x)
        
        return x

class TrajectoryEncoder(nn.Module):
    def __init__(self, coord_dim: int = 2, embed_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_enc = PositionalEncoding(embed_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.mlp(coords)
        x = self.pos_enc(x)
        return x

class MotionEncoder(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.vel_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.accel_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, history_coords: torch.Tensor) -> torch.Tensor:
        velocity = torch.diff(history_coords, dim=1)
        velocity = torch.cat([velocity[:, :1], velocity], dim=1)

        acceleration = torch.diff(velocity, dim=1)
        acceleration = torch.cat([acceleration[:, :1], acceleration], dim=1)

        vel_embed = self.vel_proj(velocity)
        accel_embed = self.accel_proj(acceleration)

        gate_input = torch.cat([vel_embed, accel_embed], dim=-1)
        gate = self.gate(gate_input)
        motion_embed = gate * vel_embed + (1 - gate) * accel_embed

        return motion_embed

class TrajectoryPredictor(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        t_future: int = 24,
        output_dim: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.t_future = t_future
        self.output_dim = output_dim
        
        self.future_queries = nn.Parameter(torch.randn(1, t_future, d_model) * 0.02)
        
        self.lead_time_enc = LeadTimeEncoding(d_model, max_lead_time=t_future)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )
        
    def forward(
        self,
        history_embed: torch.Tensor,
        physics_embed: torch.Tensor,
    ) -> torch.Tensor:
        B = history_embed.shape[0]
        device = history_embed.device
        
        memory = torch.cat([history_embed, physics_embed], dim=1)
        
        queries = self.future_queries.expand(B, -1, -1)

        lead_time_emb = self.lead_time_enc(self.t_future, B, device)
        queries = queries + lead_time_emb
        
        decoded = self.decoder(queries, memory)
        
        output = self.output_proj(decoded)
        
        return output

class LT3PModel(nn.Module):
    
    def __init__(
        self,
        coord_dim: int = 2,
        output_dim: int = 2,
        era5_channels: int = 9,
        t_history: int = 16,
        t_future: int = 24,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_history = t_history
        self.t_future = t_future
        self.output_dim = output_dim

        coord_embed_dim = d_model // 2
        self.trajectory_encoder = TrajectoryEncoder(coord_dim, coord_embed_dim)

        self.traj_proj = nn.Linear(coord_embed_dim, d_model)

        self.motion_encoder = MotionEncoder(d_model)

        self.physics_encoder = PhysicsEncoder3D(
            in_channels=era5_channels,
            base_channels=64,
            out_dim=d_model
        )

        self.past_physics_encoder = PhysicsEncoder3D(
            in_channels=era5_channels,
            base_channels=64,
            out_dim=d_model
        )

        self.trajectory_predictor = TrajectoryPredictor(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            t_future=t_future,
            output_dim=output_dim
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        history_coords: torch.Tensor,
        future_era5: torch.Tensor,
        target_coords: torch.Tensor = None,
        past_era5: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, T_hist, _ = history_coords.shape
        device = history_coords.device

        history_embed = self.trajectory_encoder(history_coords)
        history_embed = self.traj_proj(history_embed)

        motion_embed = self.motion_encoder(history_coords)
        history_embed = history_embed + motion_embed

        if past_era5 is not None:
            past_physics_embed = self.past_physics_encoder(past_era5)
            history_embed = history_embed + past_physics_embed

        physics_embed = self.physics_encoder(future_era5)

        last_pos = history_coords[:, -1:, :]
        recent_vels = torch.diff(history_coords[:, -4:], dim=1)
        avg_vel = recent_vels.mean(dim=1, keepdim=True)

        steps = torch.arange(1, self.t_future + 1, device=device).float().view(1, -1, 1)
        linear_baseline = last_pos + avg_vel * steps

        residual = self.trajectory_predictor(history_embed, physics_embed)

        predicted_coords = linear_baseline + residual
        predicted_coords = predicted_coords.clamp(0.0, 1.0)

        outputs = {
            'predicted_coords': predicted_coords,
            'linear_baseline': linear_baseline,
            'residual': residual,
        }

        if target_coords is not None:
            T = target_coords.shape[1]

            time_weights = torch.linspace(1.0, 1.5, T, device=device).view(1, T, 1)
            weighted_mse = ((predicted_coords - target_coords) ** 2 * time_weights).mean()

            continuity_loss = F.mse_loss(
                predicted_coords[:, 0], history_coords[:, -1]
            )

            hist_dir = history_coords[:, -1] - history_coords[:, -2]
            pred_dir = predicted_coords[:, 0] - history_coords[:, -1]

            hist_dir_norm = F.normalize(hist_dir, dim=-1, eps=1e-8)
            pred_dir_norm = F.normalize(pred_dir, dim=-1, eps=1e-8)
            cos_sim = (hist_dir_norm * pred_dir_norm).sum(dim=-1)
            direction_loss = (1 - cos_sim).clamp(min=0).mean()

            pred_segments = torch.diff(predicted_coords, dim=1)
            if pred_segments.shape[1] > 1:
                seg1 = F.normalize(pred_segments[:, :-1], dim=-1, eps=1e-8)
                seg2 = F.normalize(pred_segments[:, 1:], dim=-1, eps=1e-8)
                cos_angles = (seg1 * seg2).sum(dim=-1)
                curvature_loss = F.relu(-cos_angles).mean()
            else:
                curvature_loss = torch.tensor(0.0, device=device)

            step_sizes = torch.norm(pred_segments, dim=-1)
            max_step = 0.03
            speed_penalty = F.relu(step_sizes - max_step).mean()

            if residual.shape[1] > 2:
                residual_accel = torch.diff(residual, n=2, dim=1)
                residual_smooth = (residual_accel ** 2).mean()
            else:
                residual_smooth = torch.tensor(0.0, device=device)

            if pred_segments.shape[1] > 2:
                cross_z = (pred_segments[:, :-1, 0] * pred_segments[:, 1:, 1] -
                           pred_segments[:, :-1, 1] * pred_segments[:, 1:, 0])
                sign_product = cross_z[:, :-1] * cross_z[:, 1:]
                oscillation_loss = F.relu(-sign_product).mean()
            else:
                oscillation_loss = torch.tensor(0.0, device=device)

            residual_l2 = (residual ** 2).mean()

            loss = (
                weighted_mse
                + 5.0 * continuity_loss
                + 2.0 * direction_loss
                + 1.5 * curvature_loss
                + 3.0 * speed_penalty
                + 1.0 * residual_smooth
                + 0.5 * oscillation_loss
                + 0.1 * residual_l2
            )

            outputs['loss'] = loss
            outputs['mse_loss'] = weighted_mse
            outputs['continuity_loss'] = continuity_loss
            outputs['direction_loss'] = direction_loss
            outputs['curvature_loss'] = curvature_loss
            outputs['speed_penalty'] = speed_penalty
            outputs['smooth_loss'] = residual_smooth
            outputs['oscillation_loss'] = oscillation_loss
            outputs['residual_l2'] = residual_l2

        return outputs
    
    @torch.no_grad()
    def predict(
        self,
        history_coords: torch.Tensor,
        future_era5: torch.Tensor,
        past_era5: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        outputs = self.forward(history_coords, future_era5, past_era5=past_era5)

        pred_coords = outputs['predicted_coords']

        pred_coords = self._smooth_loops(pred_coords)

        return {
            'predicted_coords': pred_coords,
            'predicted_lat': pred_coords[:, :, 0],
            'predicted_lon': pred_coords[:, :, 1],
        }

    @torch.no_grad()
    def _smooth_loops(self, coords: torch.Tensor) -> torch.Tensor:
        B, T, D = coords.shape
        result = coords.clone()

        result = self._remove_self_intersections(result)

        for _ in range(5):
            segments = torch.diff(result, dim=1)
            if segments.shape[1] < 2:
                break

            dot = (segments[:, :-1] * segments[:, 1:]).sum(dim=-1)
            norm1 = segments[:, :-1].norm(dim=-1) + 1e-8
            norm2 = segments[:, 1:].norm(dim=-1) + 1e-8
            cos_angle = dot / (norm1 * norm2)

            sharp = cos_angle < 0.0
            if not sharp.any():
                break

            new_result = result.clone()
            for t in range(sharp.shape[1]):
                mask = sharp[:, t]
                if mask.any():
                    pt = t + 1
                    avg = 0.5 * (result[:, pt - 1] + result[:, pt + 1])
                    new_result[:, pt] = torch.where(
                        mask.unsqueeze(-1).expand(-1, D), avg, result[:, pt]
                    )
            result = new_result

        return result.clamp(0.0, 1.0)

    @torch.no_grad()
    def _remove_self_intersections(self, coords: torch.Tensor) -> torch.Tensor:
        B, T, D = coords.shape
        result = coords.clone()

        for b in range(B):
            pts = result[b]
            i = 0
            while i < T - 2:
                found = False
                for j in range(i + 2, T - 1):
                    if self._segments_intersect(
                        pts[i], pts[i + 1], pts[j], pts[j + 1]
                    ):
                        loop_len = j - i
                        for k in range(1, loop_len):
                            alpha = k / loop_len
                            result[b, i + k] = (1 - alpha) * pts[i] + alpha * pts[j + 1]
                        pts = result[b]
                        i = j
                        found = True
                        break
                if not found:
                    i += 1

        return result

    @staticmethod
    def _segments_intersect(
        p1: torch.Tensor, p2: torch.Tensor,
        p3: torch.Tensor, p4: torch.Tensor,
    ) -> bool:
        d1 = p2 - p1
        d2 = p4 - p3

        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross.item()) < 1e-10:
            return False

        d3 = p3 - p1
        t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
        u = (d3[0] * d1[1] - d3[1] * d1[0]) / cross

        return 0 < t.item() < 1 and 0 < u.item() < 1

class LT3PDiffusionModel(LT3PModel):
    
    def __init__(self, *args, num_diffusion_steps: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_diffusion_steps = num_diffusion_steps
        
        d_model = kwargs.get('d_model', 256)
        self.noise_pred = nn.Sequential(
            nn.Linear(self.output_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.output_dim),
        )
        
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, num_diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    @torch.no_grad()
    def sample_ensemble(
        self,
        history_coords: torch.Tensor,
        future_era5: torch.Tensor,
        num_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        
        base_pred = self.forward(history_coords, future_era5)['predicted_coords']
        
        all_samples = [base_pred]
        
        for _ in range(num_samples - 1):
            noise = torch.randn_like(base_pred) * 0.01
            perturbed = base_pred + noise
            all_samples.append(perturbed)
        
        samples = torch.stack(all_samples, dim=0)
        mean_pred = samples.mean(dim=0)
        std_pred = samples.std(dim=0)
        
        return {
            'predicted_coords': mean_pred,
            'predicted_lat': mean_pred[:, :, 0],
            'predicted_lon': mean_pred[:, :, 1],
            'uncertainty': std_pred,
            'all_samples': samples,
        }

def create_lt3p_model(use_diffusion: bool = False) -> nn.Module:
    
    common_kwargs = {
        'coord_dim': model_cfg.coord_dim,
        'output_dim': model_cfg.output_dim,
        'era5_channels': model_cfg.era5_channels,
        't_history': model_cfg.t_history,
        't_future': model_cfg.t_future,
        'd_model': model_cfg.transformer_dim,
        'n_heads': model_cfg.transformer_heads,
        'n_layers': model_cfg.transformer_layers,
        'ff_dim': model_cfg.transformer_ff_dim,
        'dropout': model_cfg.dropout,
    }
    
    if use_diffusion:
        return LT3PDiffusionModel(
            **common_kwargs,
            num_diffusion_steps=model_cfg.num_diffusion_steps
        )
    else:
        return LT3PModel(**common_kwargs)
