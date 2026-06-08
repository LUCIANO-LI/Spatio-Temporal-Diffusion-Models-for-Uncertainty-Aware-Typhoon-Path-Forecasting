import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import random

from config import model_cfg, train_cfg, data_cfg
from data_structures import StormSample

TRAJ_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TRAJ_DIR, ".."))
DIFFUSION_DIR = os.path.join(PROJECT_ROOT, "Diffusion")

def _load_norm_stats():
    expected_channels = model_cfg.era5_channels

    possible_paths = [
        os.environ.get("TYPHOON_NORM_STATS", ""),
        os.path.join(DIFFUSION_DIR, 'norm_stats.pt'),
        os.path.join(TRAJ_DIR, 'norm_stats.pt'),
    ]

    for path in possible_paths:
        if not path:
            continue
        if os.path.exists(path):
            stats = torch.load(path, weights_only=True, map_location='cpu')
            mean = stats['mean'].numpy()
            std = stats['std'].numpy()
            loaded_ch = len(mean)
            if loaded_ch != expected_channels:
                raise ValueError(
                    f"norm_stats.pt 通道数不匹配! 加载到 {loaded_ch}ch, 但配置期望 {expected_channels}ch。"
                    f"请用新的扩散模型重新训练生成 norm_stats.pt (路径: {os.path.abspath(path)})"
                )
            print(f"[dataset.py] 从 {os.path.abspath(path)} 加载归一化统计 ({loaded_ch}ch)")
            return mean, std

    print("[dataset.py] 警告: 未找到 norm_stats.pt，使用硬编码回退值")
    mean = np.array([
        -1.29, -0.28, 0.39,
        1.74, 2.38, 2.60,
        14253.13, 56708.52, 106498.13,
    ], dtype=np.float32)

    std = np.array([
        10.02, 10.24, 12.65,
        9.04, 8.48, 8.56,
        1320.93, 4869.39, 9103.74,
    ], dtype=np.float32)

    return mean, std

ERA5_CHANNEL_MEAN, ERA5_CHANNEL_STD = _load_norm_stats()

def normalize_era5(era5: np.ndarray) -> np.ndarray:
    if era5.ndim == 4:
        C = era5.shape[1]
        mean = ERA5_CHANNEL_MEAN[:C].reshape(1, C, 1, 1)
        std = ERA5_CHANNEL_STD[:C].reshape(1, C, 1, 1)
    elif era5.ndim == 3:
        C = era5.shape[0]
        mean = ERA5_CHANNEL_MEAN[:C].reshape(C, 1, 1)
        std = ERA5_CHANNEL_STD[:C].reshape(C, 1, 1)
    else:
        return era5
    return (era5 - mean) / (std + 1e-8)

def normalize_coords(lat, lon, lat_range=None, lon_range=None):
    if lat_range is None:
        lat_range = data_cfg.lat_range
    if lon_range is None:
        lon_range = data_cfg.lon_range
    lat_norm = (lat - lat_range[0]) / (lat_range[1] - lat_range[0])
    lon_norm = (lon - lon_range[0]) / (lon_range[1] - lon_range[0])
    return lat_norm, lon_norm

def denormalize_coords(lat_norm, lon_norm, lat_range=None, lon_range=None):
    if lat_range is None:
        lat_range = data_cfg.lat_range
    if lon_range is None:
        lon_range = data_cfg.lon_range
    lat = lat_norm * (lat_range[1] - lat_range[0]) + lat_range[0]
    lon = lon_norm * (lon_range[1] - lon_range[0]) + lon_range[0]
    return lat, lon

class LT3PDataset(Dataset):
    
    def __init__(
        self,
        storm_samples: List[StormSample],
        t_history: int = None,
        t_future: int = None,
        stride: int = 1,
        era5_channels: int = None,
        time_resolution_hours: int = None,
    ):
        self.storm_samples = storm_samples
        self.t_history = t_history or model_cfg.t_history
        self.t_future = t_future or model_cfg.t_future
        self.stride = stride
        self.era5_channels = era5_channels or model_cfg.era5_channels
        self.time_resolution_hours = time_resolution_hours or data_cfg.time_resolution_hours
        
        self.total_length = self.t_history + self.t_future
        
        self.samples_index = self._build_samples_index()
        
        print(f"LT3PDataset: t_history={self.t_history}, t_future={self.t_future}, "
              f"total_length={self.total_length}, samples={len(self.samples_index)}")
    
    def _build_samples_index(self) -> List[Tuple[int, int]]:
        index = []
        
        for storm_idx, sample in enumerate(self.storm_samples):
            T = len(sample)
            if T < self.total_length:
                continue
            
            for start in range(0, T - self.total_length + 1, self.stride):
                index.append((storm_idx, start))
        
        return index
    
    def __len__(self) -> int:
        return len(self.samples_index)
    
    def _get_era5_video(self, sample: StormSample, start: int, end: int) -> np.ndarray:
        T = end - start
        
        if sample.era5_array is not None:
            era5 = sample.era5_array[start:end]
            if era5.shape[1] > self.era5_channels:
                era5 = era5[:, :self.era5_channels]
            elif era5.shape[1] < self.era5_channels:
                pad = np.zeros((T, self.era5_channels - era5.shape[1], 
                               era5.shape[2], era5.shape[3]), dtype=np.float32)
                era5 = np.concatenate([era5, pad], axis=1)
            return era5
        elif sample.era5_dataset is not None:
            frames = []
            for t in range(start, end):
                frame = sample.get_era5_at_time(t)
                if frame is not None:
                    if frame.shape[0] > self.era5_channels:
                        frame = frame[:self.era5_channels]
                    frames.append(frame)
                else:
                    frames.append(np.zeros((self.era5_channels, 
                                           data_cfg.grid_height, 
                                           data_cfg.grid_width), dtype=np.float32))
            return np.stack(frames, axis=0)
        else:
            return np.zeros((T, self.era5_channels, 
                           data_cfg.grid_height, data_cfg.grid_width), dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        storm_idx, start_idx = self.samples_index[idx]
        sample = self.storm_samples[storm_idx]
        
        history_start = start_idx
        history_end = start_idx + self.t_history
        future_start = history_end
        future_end = history_end + self.t_future
        
        history_lat = sample.track_lat[history_start:history_end]
        history_lon = sample.track_lon[history_start:history_end]
        
        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)
        
        future_era5 = self._get_era5_video(sample, future_start, future_end)
        future_era5 = normalize_era5(future_era5)

        past_era5 = self._get_era5_video(sample, history_start, history_end)
        past_era5 = normalize_era5(past_era5)
        
        future_lat = sample.track_lat[future_start:future_end]
        future_lon = sample.track_lon[future_start:future_end]
        
        f_lat_n, f_lon_n = normalize_coords(future_lat, future_lon)
        target_coords = np.stack([f_lat_n, f_lon_n], axis=-1)
        
        sample_weight = 1.0
        if sample.is_real is not None:
            real_ratio = sample.is_real[future_start:future_end].mean()
            sample_weight = train_cfg.interp_sample_weight + \
                real_ratio * (train_cfg.real_sample_weight - train_cfg.interp_sample_weight)
        
        return {
            'history_coords': torch.from_numpy(history_coords).float(),
            'past_era5': torch.from_numpy(past_era5).float(),
            'future_era5': torch.from_numpy(future_era5).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'sample_weight': torch.tensor(sample_weight).float(),
            'storm_id': sample.storm_id,
            'target_lat_raw': torch.from_numpy(future_lat).float(),
            'target_lon_raw': torch.from_numpy(future_lon).float(),
            'history_lat_raw': torch.from_numpy(history_lat).float(),
            'history_lon_raw': torch.from_numpy(history_lon).float(),
        }

def split_storms_by_id(
    storm_samples: List[StormSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[StormSample], List[StormSample], List[StormSample]]:
    random.seed(seed)
    samples = storm_samples.copy()
    random.shuffle(samples)
    
    n = len(samples)
    
    if n <= 3:
        return samples, [], []
    elif n <= 6:
        n_train = n - 2
        n_val = 1
    else:
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_val == 0:
            n_val = 1
        if n_train + n_val >= n:
            n_train = n - n_val - 1
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    return train_samples, val_samples, test_samples

def split_storms_by_year(
    storm_samples: List[StormSample],
    train_years: List[int],
    val_years: List[int],
    test_years: List[int]
) -> Tuple[List[StormSample], List[StormSample], List[StormSample]]:
    train_samples = [s for s in storm_samples if s.year in train_years]
    val_samples = [s for s in storm_samples if s.year in val_years]
    test_samples = [s for s in storm_samples if s.year in test_years]
    return train_samples, val_samples, test_samples

def filter_short_storms(
    storm_samples: List[StormSample],
    min_duration_hours: int = 120,
    time_resolution_hours: int = 3
) -> List[StormSample]:
    min_steps = min_duration_hours // time_resolution_hours
    filtered = [s for s in storm_samples if len(s) >= min_steps]
    print(f"Filtered storms: {len(storm_samples)} -> {len(filtered)} "
          f"(min duration: {min_duration_hours}h = {min_steps} steps)")
    return filtered

def filter_out_of_range_storms(
    storm_samples: List[StormSample],
    lat_range: tuple = None,
    lon_range: tuple = None,
) -> List[StormSample]:
    if lat_range is None:
        lat_range = data_cfg.lat_range
    if lon_range is None:
        lon_range = data_cfg.lon_range

    filtered = []
    removed_ids = []
    for s in storm_samples:
        lat_ok = (s.track_lat.min() >= lat_range[0]) and (s.track_lat.max() <= lat_range[1])
        lon_ok = (s.track_lon.min() >= lon_range[0]) and (s.track_lon.max() <= lon_range[1])
        if lat_ok and lon_ok:
            filtered.append(s)
        else:
            removed_ids.append(s.storm_id)

    print(f"Filtered out-of-range storms: {len(storm_samples)} -> {len(filtered)} "
          f"(removed {len(removed_ids)} storms outside lat{lat_range} lon{lon_range})")
    return filtered

def create_dataloaders(
    storm_samples: List[StormSample],
    batch_size: int = None,
    split_by: str = 'storm_id',
    **split_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = batch_size or train_cfg.batch_size
    
    min_duration = train_cfg.min_typhoon_duration_hours
    storm_samples = filter_short_storms(storm_samples, min_duration)

    storm_samples = filter_out_of_range_storms(storm_samples)
    
    if split_by == 'storm_id':
        train_s, val_s, test_s = split_storms_by_id(
            storm_samples,
            train_cfg.train_ratio,
            train_cfg.val_ratio
        )
    else:
        train_s, val_s, test_s = split_storms_by_year(storm_samples, **split_kwargs)
    
    train_ds = LT3PDataset(train_s, stride=1)
    val_ds = LT3PDataset(val_s, stride=model_cfg.t_future)
    test_ds = LT3PDataset(test_s, stride=model_cfg.t_future)
    
    num_workers = train_cfg.num_workers
    pin_memory = train_cfg.pin_memory
    
    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Storms - Train: {len(train_s)}, Val: {len(val_s)}, Test: {len(test_s)}")
    
    return train_loader, val_loader, test_loader
