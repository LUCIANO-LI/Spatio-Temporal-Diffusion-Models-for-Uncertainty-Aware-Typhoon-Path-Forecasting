import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    # 时间步嵌入，给扩散模型提供当前噪声强度信息

    def __init__(self, dim: int = 256, time_emb_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

class AdaLayerNorm(nn.Module):
    # 用时间嵌入调制 LayerNorm，是 DiT 里注入 timestep 的位置

    def __init__(self, d_model: int, time_emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        # 输出两组 scale/shift/gate，分别给 attention 和 FFN 用
        params = self.adaLN_modulation(time_emb).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        return gamma1, beta1, alpha1, gamma2, beta2, alpha2

    def modulate(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return (1 + gamma) * self.norm(x) + beta

class PatchEmbed(nn.Module):
    # 把 40x40 气象场切成 patch token

    def __init__(self, in_channels: int, d_model: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        x = x.reshape(B, D, Hp * Wp).permute(0, 2, 1)
        return x

class Unpatchify(nn.Module):
    # 把 token 还原成二维气象场

    def __init__(self, out_channels: int, patch_size: int = 4, grid_size: int = 40):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.grid_h = grid_size // patch_size
        self.grid_w = grid_size // patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        p = self.patch_size
        C = self.out_channels
        x = x.reshape(B, self.grid_h, self.grid_w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, self.grid_h * p, self.grid_w * p)
        return x

class DiTBlock(nn.Module):
    # DiT 主块：自注意力看当前场，交叉注意力看历史条件

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        time_emb_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, f"d_model {d_model} 必须被 n_heads {n_heads} 整除"

        self.adaln = AdaLayerNorm(d_model, time_emb_dim)

        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_proj = nn.Linear(d_model, d_model)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_kv = nn.Linear(d_model, 2 * d_model)
        self.cross_attn_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(x, time_emb)

        # 当前 noisy field 内部建模
        h = self.adaln.modulate(x, gamma1, beta1)
        qkv = self.self_attn_qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        attn_out = self.self_attn_proj(attn_out)
        x = x + alpha1 * self.dropout(attn_out)

        # 从历史 ERA5 条件里取信息
        h_cross = self.cross_attn_norm(x)
        q_cross = self.cross_attn_q(h_cross).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        N_cond = cond_tokens.shape[1]
        kv_cross = self.cross_attn_kv(cond_tokens).reshape(B, N_cond, 2, self.n_heads, self.head_dim)
        kv_cross = kv_cross.permute(2, 0, 3, 1, 4)
        k_cross, v_cross = kv_cross[0], kv_cross[1]
        cross_out = F.scaled_dot_product_attention(q_cross, k_cross, v_cross)
        cross_out = cross_out.permute(0, 2, 1, 3).reshape(B, N, D)
        cross_out = self.cross_attn_proj(cross_out)
        x = x + self.dropout(cross_out)

        h_ffn = self.adaln.modulate(x, gamma2, beta2)
        ffn_out = self.ffn(h_ffn)
        x = x + alpha2 * ffn_out

        return x
