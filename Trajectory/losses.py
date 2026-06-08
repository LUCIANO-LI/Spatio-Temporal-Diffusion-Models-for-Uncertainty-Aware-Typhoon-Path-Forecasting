from typing import Optional
import torch
import torch.nn.functional as F

from config import data_cfg

def physics_loss(
    pred_deltas: torch.Tensor,
    cond_coords: torch.Tensor,
    cond_features: Optional[torch.Tensor] = None,
    dt_hours: float = 0.5,
) -> torch.Tensor:
    lat_range = data_cfg.lat_range
    lon_range = data_cfg.lon_range

    last_coord = cond_coords[:, -1:, :3]
    pred_coords = last_coord + torch.cumsum(pred_deltas, dim=1)
    pred_lat = pred_coords[..., 0]
    pred_lon = pred_coords[..., 1]
    pred_vmax = pred_coords[..., 2]

    lat_deg = pred_lat * (lat_range[1] - lat_range[0]) + lat_range[0]
    lon_deg = pred_lon * (lon_range[1] - lon_range[0]) + lon_range[0]

    dlat_deg = pred_deltas[..., 0] * (lat_range[1] - lat_range[0])
    dlon_deg = pred_deltas[..., 1] * (lon_range[1] - lon_range[0])

    dist_lat_km = dlat_deg * 111.0
    dist_lon_km = dlon_deg * 111.0 * torch.cos(torch.deg2rad(lat_deg + 1e-6))
    speed_kmh = torch.sqrt(dist_lat_km**2 + dist_lon_km**2) / dt_hours

    speed_penalty = F.relu(speed_kmh - 100.0)

    accel = torch.diff(speed_kmh, dim=1)
    smooth_penalty = torch.abs(accel)

    direction = torch.atan2(dlon_deg, dlat_deg + 1e-6)
    dir_diff = torch.diff(direction, dim=1)
    dir_penalty = torch.abs(torch.sin(dir_diff / 2.0))

    vmax_change = pred_deltas[..., 2] * 100.0
    vmax_penalty = F.relu(torch.abs(vmax_change) - 0.625)

    loss = (
        speed_penalty.mean()
        + smooth_penalty.mean()
        + dir_penalty.mean()
        + vmax_penalty.mean()
    )
    return loss

def geo_distance_loss(
    pred_coords_norm: torch.Tensor,
    target_coords_norm: torch.Tensor,
    normalize_factor_km: float = 500.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lat_range = data_cfg.lat_range
    lon_range = data_cfg.lon_range

    pred_lat = pred_coords_norm[..., 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
    pred_lon = pred_coords_norm[..., 1] * (lon_range[1] - lon_range[0]) + lon_range[0]
    tgt_lat = target_coords_norm[..., 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
    tgt_lon = target_coords_norm[..., 1] * (lon_range[1] - lon_range[0]) + lon_range[0]

    lat_err_km = (pred_lat - tgt_lat) * 111.0
    lon_err_km = (pred_lon - tgt_lon) * 111.0 * torch.cos(torch.deg2rad(tgt_lat))
    dist_km = torch.sqrt(lat_err_km**2 + lon_err_km**2)

    mae_km = dist_km.mean()
    rmse_km = torch.sqrt((dist_km**2).mean())
    loss = mae_km / normalize_factor_km
    return loss, mae_km, rmse_km
