import numpy as np
from typing import Tuple, Optional, List
import pandas as pd

from config import model_cfg, data_cfg
from data_structures import StormSample

def find_nearest_grid_point(
    lat: float,
    lon: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray
) -> Tuple[int, int]:
    i = np.argmin(np.abs(lat_grid - lat))
    j = np.argmin(np.abs(lon_grid - lon))
    return int(i), int(j)

def compute_wind_shear(
    u_850: np.ndarray, v_850: np.ndarray,
    u_200: np.ndarray, v_200: np.ndarray,
    center_i: int, center_j: int,
    window_size: int = 9
) -> Tuple[float, float]:
    H, W = u_850.shape
    half = window_size // 2
    
    i_start = max(0, center_i - half)
    i_end = min(H, center_i + half + 1)
    j_start = max(0, center_j - half)
    j_end = min(W, center_j + half + 1)
    
    u_850_mean = np.mean(u_850[i_start:i_end, j_start:j_end])
    v_850_mean = np.mean(v_850[i_start:i_end, j_start:j_end])
    u_200_mean = np.mean(u_200[i_start:i_end, j_start:j_end])
    v_200_mean = np.mean(v_200[i_start:i_end, j_start:j_end])
    
    du = u_200_mean - u_850_mean
    dv = v_200_mean - v_850_mean
    
    shear_magnitude = np.sqrt(du**2 + dv**2)
    shear_direction = np.arctan2(dv, du)
    
    return float(shear_magnitude), float(shear_direction)

def compute_local_stats(
    field: np.ndarray,
    center_i: int, center_j: int,
    window_size: int = 9
) -> Tuple[float, float, float]:
    H, W = field.shape
    half = window_size // 2
    
    i_start = max(0, center_i - half)
    i_end = min(H, center_i + half + 1)
    j_start = max(0, center_j - half)
    j_end = min(W, center_j + half + 1)
    
    window = field[i_start:i_end, j_start:j_end]
    
    return float(np.mean(window)), float(np.min(window)), float(np.max(window))

def encode_time_features(timestamp: np.datetime64) -> np.ndarray:
    ts = pd.Timestamp(timestamp)
    
    day_of_year = ts.dayofyear
    month = ts.month
    
    day_sin = np.sin(2 * np.pi * day_of_year / 366)
    day_cos = np.cos(2 * np.pi * day_of_year / 366)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    return np.array([day_sin, day_cos, month_sin, month_cos], dtype=np.float32)

def compute_motion_features(
    track_lat: np.ndarray,
    track_lon: np.ndarray,
    track_vmax: np.ndarray,
    t_idx: int,
    dt_hours: float = 0.5
) -> np.ndarray:
    features = np.zeros(6, dtype=np.float32)
    
    if t_idx == 0:
        return features
    
    lat_now, lon_now = track_lat[t_idx], track_lon[t_idx]
    lat_prev, lon_prev = track_lat[t_idx - 1], track_lon[t_idx - 1]
    
    dlat = lat_now - lat_prev
    dlon = lon_now - lon_prev
    
    dist_km = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(lat_now)))**2)
    speed = dist_km / dt_hours
    
    direction = np.arctan2(dlon, dlat)
    
    vmax_now = track_vmax[t_idx]
    vmax_prev = track_vmax[t_idx - 1]
    vmax_change = vmax_now - vmax_prev
    
    acceleration = 0.0
    if t_idx >= 2:
        lat_prev2, lon_prev2 = track_lat[t_idx - 2], track_lon[t_idx - 2]
        dlat_prev = lat_prev - lat_prev2
        dlon_prev = lon_prev - lon_prev2
        dist_km_prev = np.sqrt((dlat_prev * 111)**2 + (dlon_prev * 111 * np.cos(np.radians(lat_prev)))**2)
        speed_prev = dist_km_prev / dt_hours
        acceleration = (speed - speed_prev) / dt_hours
    
    features = np.array([speed, direction, dlat, dlon, vmax_change, acceleration], dtype=np.float32)
    return features

def build_single_step_features(
    storm_sample: StormSample,
    t_idx: int,
    window_size: int = 9
) -> np.ndarray:
    features_list = []
    
    lat = storm_sample.track_lat[t_idx]
    lon = storm_sample.track_lon[t_idx]
    vmax = storm_sample.track_vmax[t_idx]
    
    lat_norm = (lat - 10) / 40
    lon_norm = (lon - 100) / 60
    vmax_norm = vmax / 85
    
    features_list.extend([lat_norm, lon_norm, vmax_norm])
    
    if storm_sample.track_pmin is not None:
        pmin = storm_sample.track_pmin[t_idx]
        pmin_norm = (pmin - 900) / 100
        features_list.append(pmin_norm)
    else:
        features_list.append(0.0)
    
    motion_features = compute_motion_features(
        storm_sample.track_lat, storm_sample.track_lon,
        storm_sample.track_vmax, t_idx
    )
    features_list.extend(motion_features.tolist())
    
    time_features = encode_time_features(storm_sample.times[t_idx])
    features_list.extend(time_features.tolist())
    
    if storm_sample.is_real is not None:
        is_real = float(storm_sample.is_real[t_idx])
    else:
        is_real = 1.0
    features_list.append(is_real)
    
    era5_features = extract_era5_features(storm_sample, t_idx, lat, lon, window_size)
    features_list.extend(era5_features.tolist())

    return np.array(features_list, dtype=np.float32)

def extract_era5_features(
    storm_sample: StormSample,
    t_idx: int,
    lat: float,
    lon: float,
    window_size: int = 9
) -> np.ndarray:
    if storm_sample.era5_dataset is None and storm_sample.era5_array is None:
        return np.zeros(32, dtype=np.float32)

    features = []
    lat_grid = storm_sample.lat_grid
    lon_grid = storm_sample.lon_grid

    center_i, center_j = find_nearest_grid_point(lat, lon, lat_grid, lon_grid)

    if storm_sample.era5_dataset is not None:
        ds = storm_sample.era5_dataset.isel(valid_time=t_idx)

        try:
            u_850 = ds['u'].sel(pressure_level=850).values
            v_850 = ds['v'].sel(pressure_level=850).values
            u_200 = ds['u'].sel(pressure_level=200).values
            v_200 = ds['v'].sel(pressure_level=200).values
            shear_mag, shear_dir = compute_wind_shear(u_850, v_850, u_200, v_200, center_i, center_j)
            features.extend([shear_mag / 30, np.sin(shear_dir), np.cos(shear_dir)])
        except:
            features.extend([0.0, 0.0, 0.0])

        try:
            if 'vo' in ds:
                vo_850 = ds['vo'].sel(pressure_level=850).values
                vo_mean, vo_min, vo_max = compute_local_stats(vo_850, center_i, center_j)
                features.extend([vo_mean * 1e4, vo_max * 1e4])
        except:
            features.extend([0.0, 0.0])

        try:
            if 'msl' in ds:
                msl = ds['msl'].values
                msl_mean, msl_min, msl_max = compute_local_stats(msl, center_i, center_j)
                features.extend([(msl_min - 100000) / 5000, (msl_mean - 100000) / 5000])
        except:
            features.extend([0.0, 0.0])

        try:
            if 'tcwv' in ds:
                tcwv = ds['tcwv'].values
                tcwv_mean, _, _ = compute_local_stats(tcwv, center_i, center_j)
                features.append(tcwv_mean / 60)
        except:
            features.append(0.0)

        try:
            if 't2m' in ds:
                t2m = ds['t2m'].values
                t2m_mean, _, _ = compute_local_stats(t2m, center_i, center_j)
                features.append((t2m_mean - 273) / 30)
        except:
            features.append(0.0)

    while len(features) < 32:
        features.append(0.0)

    return np.array(features[:32], dtype=np.float32)

def build_cond_features_for_storm(storm_sample: StormSample) -> np.ndarray:
    T = len(storm_sample)
    features_list = []

    for t in range(T):
        feat = build_single_step_features(storm_sample, t)
        features_list.append(feat)

    return np.stack(features_list, axis=0)

def normalize_features(features: np.ndarray, stats: dict = None) -> Tuple[np.ndarray, dict]:
    if stats is None:
        mean = features.mean(axis=tuple(range(features.ndim - 1)), keepdims=True)
        std = features.std(axis=tuple(range(features.ndim - 1)), keepdims=True) + 1e-8
        stats = {'mean': mean.squeeze(), 'std': std.squeeze()}

    normalized = (features - stats['mean']) / stats['std']
    return normalized, stats

