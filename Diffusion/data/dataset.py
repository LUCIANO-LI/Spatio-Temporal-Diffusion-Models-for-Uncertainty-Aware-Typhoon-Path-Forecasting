import os
import glob
import math
import logging
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

def _extract_timestamp_from_filename(filename: str) -> str:
    basename = os.path.basename(filename)
    parts = basename.replace(".nc", "").split("_")
    return parts[2]

def _load_single_nc_to_array(
    filepath: str,
    pl_vars: List[str],
    sfc_vars: List[str],
    pressure_levels: List[int],
) -> np.ndarray:
    import netCDF4 as nc

    ds = nc.Dataset(filepath, "r")
    n_times = ds.dimensions["valid_time"].size
    H = ds.dimensions["latitude"].size
    W = ds.dimensions["longitude"].size
    n_levels = len(pressure_levels)
    n_channels = len(pl_vars) * n_levels + len(sfc_vars)

    result = np.empty((n_times, n_channels, H, W), dtype=np.float32)
    ch = 0

    available_levels = ds.variables["pressure_level"][:].astype(float)

    level_indices = []
    for target_level in pressure_levels:
        idx = int(np.argmin(np.abs(available_levels - target_level)))
        level_indices.append(idx)

    for var_name in pl_vars:
        data = ds.variables[var_name][:].astype(np.float32)
        for lev_idx in level_indices:
            result[:, ch, :, :] = data[:, lev_idx, :, :]
            ch += 1

    for var_name in sfc_vars:
        data = ds.variables[var_name][:].astype(np.float32)
        result[:, ch, :, :] = data
        ch += 1

    ds.close()
    return result

def _crop_pad_to_target(data: np.ndarray, target_H: int = 40, target_W: int = 40) -> np.ndarray:
    if data.ndim == 3:
        C, H, W = data.shape
        if H == target_H and W == target_W:
            return data
        padded = np.zeros((C, target_H, target_W), dtype=np.float32)
        h = min(H, target_H)
        w = min(W, target_W)
        padded[:, :h, :w] = data[:, :h, :w]
        return padded
    elif data.ndim == 4:
        T, C, H, W = data.shape
        if H == target_H and W == target_W:
            return data
        padded = np.zeros((T, C, target_H, target_W), dtype=np.float32)
        h = min(H, target_H)
        w = min(W, target_W)
        padded[:, :, :h, :w] = data[:, :, :h, :w]
        return padded
    else:
        raise ValueError(f"不支持的数据维度: {data.ndim}")

def preprocess_to_npy(
    data_root: str,
    output_dir: str,
    pl_vars: List[str],
    sfc_vars: List[str],
    pressure_levels: List[int],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    typhoon_dirs = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    skipped = 0
    converted = 0
    failed = 0

    for tid in tqdm(typhoon_dirs, desc="预处理 NC→NPY", unit="台风"):
        out_path = os.path.join(output_dir, f"{tid}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue

        typhoon_dir = os.path.join(data_root, tid)
        nc_files = sorted(glob.glob(os.path.join(typhoon_dir, "era5_merged_*.nc")))
        if not nc_files:
            continue

        steps = []
        for f in nc_files:
            try:
                arr = _load_single_nc_to_array(f, pl_vars, sfc_vars, pressure_levels)
                step = arr[0]
                step = _crop_pad_to_target(step, 40, 40)
                step = np.nan_to_num(step, nan=0.0)
                steps.append(step)
            except Exception as e:
                logger.warning(f"预处理跳过 {f}: {e}")

        if steps:
            data = np.stack(steps, axis=0)
            np.save(out_path, data)
            converted += 1
        else:
            failed += 1

    logger.info(
        f"预处理完成: 转换={converted}, 跳过(已存在)={skipped}, 失败={failed}"
    )

class ERA5TyphoonDataset(Dataset):

    def __init__(
        self,
        typhoon_ids: List[str],
        data_root: str,
        pl_vars: List[str],
        sfc_vars: List[str],
        pressure_levels: List[int],
        history_steps: int = 15,
        forecast_steps: int = 3,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        preload: bool = False,
        preprocessed_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.pl_vars = pl_vars
        self.sfc_vars = sfc_vars
        self.pressure_levels = pressure_levels
        self.n_levels = len(pressure_levels)
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.window_size = history_steps + forecast_steps
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.preprocessed_dir = preprocessed_dir

        self.num_channels = len(pl_vars) * self.n_levels + len(sfc_vars)

        self.samples = []
        self._mmap_cache: Dict[str, np.ndarray] = {}

        if preprocessed_dir:
            self._build_index_preprocessed(typhoon_ids)
        else:
            self._build_index_nc(typhoon_ids)

    def _build_index_preprocessed(self, typhoon_ids: List[str]):
        total_samples = 0
        for tid in tqdm(sorted(typhoon_ids), desc="扫描预处理数据", unit="个"):
            npy_path = os.path.join(self.preprocessed_dir, f"{tid}.npy")
            if not os.path.exists(npy_path):
                continue

            # 只读 shape，不把整条台风加载进内存
            data = np.load(npy_path, mmap_mode='r')
            n_steps = data.shape[0]

            if n_steps < self.window_size:
                continue

            for i in range(n_steps - self.window_size + 1):
                self.samples.append((tid, i))
                total_samples += 1

        logger.info(
            f"数据集构建完成 (NPY模式): {len(typhoon_ids)} 个台风, {total_samples} 个样本"
        )

    def _build_index_nc(self, typhoon_ids: List[str]):
        total_samples = 0
        for tid in tqdm(sorted(typhoon_ids), desc="扫描台风目录", unit="个"):
            typhoon_dir = os.path.join(self.data_root, tid)
            if not os.path.isdir(typhoon_dir):
                continue

            nc_files = sorted(glob.glob(os.path.join(typhoon_dir, "era5_merged_*.nc")))
            if len(nc_files) < self.window_size:
                continue

            for i in range(len(nc_files) - self.window_size + 1):
                window_files = nc_files[i : i + self.window_size]
                self.samples.append((tid, window_files))
                total_samples += 1

        logger.info(
            f"数据集构建完成 (NC模式): {len(typhoon_ids)} 个台风, {total_samples} 个样本"
        )

    def _get_mmap(self, tid: str) -> np.ndarray:
        if tid not in self._mmap_cache:
            npy_path = os.path.join(self.preprocessed_dir, f"{tid}.npy")
            self._mmap_cache[tid] = np.load(npy_path, mmap_mode='r')
        return self._mmap_cache[tid]

    def _load_step_nc(self, filepath: str) -> np.ndarray:
        arr = _load_single_nc_to_array(
            filepath, self.pl_vars, self.sfc_vars, self.pressure_levels
        )
        result = arr[0]
        result = _crop_pad_to_target(result, 40, 40)
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if self.preprocessed_dir:
            return self._getitem_preprocessed(sample)
        else:
            return self._getitem_nc(sample)

    def _getitem_preprocessed(self, sample) -> Dict[str, torch.Tensor]:
        tid, start_idx = sample
        mmap_data = self._get_mmap(tid)

        # mmap 切片后再拷贝成 batch 数据
        all_data = np.array(
            mmap_data[start_idx : start_idx + self.window_size]
        )

        condition_data = all_data[: self.history_steps]
        target_data = all_data[self.history_steps :]

        if self.norm_mean is not None and self.norm_std is not None:
            mean = self.norm_mean[None, :, None, None]
            std = self.norm_std[None, :, None, None]
            std = np.where(std < 1e-8, 1.0, std)
            condition_data = (condition_data - mean) / std
            target_data = (target_data - mean) / std

        condition = condition_data.reshape(-1, *condition_data.shape[2:])
        target = target_data.reshape(-1, *target_data.shape[2:])

        return {
            "condition": torch.from_numpy(condition).float(),
            "target": torch.from_numpy(target).float(),
            "typhoon_id": tid,
        }

    def _getitem_nc(self, sample) -> Dict[str, torch.Tensor]:
        tid, window_files = sample

        steps = []
        for f in window_files:
            try:
                step_data = self._load_step_nc(f)
                steps.append(step_data)
            except Exception as e:
                logger.warning(f"加载失败 {f}: {e}, 使用零填充")
                steps.append(np.zeros((self.num_channels, 40, 40), dtype=np.float32))

        target_shape = (self.num_channels, 40, 40)
        for i, s in enumerate(steps):
            if s.shape != target_shape:
                steps[i] = np.zeros(target_shape, dtype=np.float32)

        all_data = np.stack(steps, axis=0)

        condition_data = all_data[: self.history_steps]
        target_data = all_data[self.history_steps :]

        if self.norm_mean is not None and self.norm_std is not None:
            mean = self.norm_mean[None, :, None, None]
            std = self.norm_std[None, :, None, None]
            std = np.where(std < 1e-8, 1.0, std)
            condition_data = (condition_data - mean) / std
            target_data = (target_data - mean) / std

        condition = condition_data.reshape(-1, *condition_data.shape[2:])
        target = target_data.reshape(-1, *target_data.shape[2:])

        return {
            "condition": torch.from_numpy(condition).float(),
            "target": torch.from_numpy(target).float(),
            "typhoon_id": tid,
        }

def compute_normalization_stats(
    typhoon_ids: List[str],
    data_root: str,
    pl_vars: List[str],
    sfc_vars: List[str],
    pressure_levels: List[int],
    max_files_per_typhoon: int = 50,
    preprocessed_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_levels = len(pressure_levels)
    n_channels = len(pl_vars) * n_levels + len(sfc_vars)
    count = np.zeros(n_channels, dtype=np.float64)
    mean = np.zeros(n_channels, dtype=np.float64)
    M2 = np.zeros(n_channels, dtype=np.float64)

    for tid in tqdm(sorted(typhoon_ids), desc="计算归一化统计", unit="个台风"):

        if preprocessed_dir:
            npy_path = os.path.join(preprocessed_dir, f"{tid}.npy")
            if not os.path.exists(npy_path):
                continue
            data = np.load(npy_path, mmap_mode='r')
            n_steps = data.shape[0]
            if n_steps > max_files_per_typhoon:
                indices = np.linspace(0, n_steps - 1, max_files_per_typhoon, dtype=int)
            else:
                indices = range(n_steps)

            for step_idx in indices:
                step = np.array(data[step_idx])
                for c in range(n_channels):
                    channel_data = step[c].ravel().astype(np.float64)
                    valid = channel_data[~np.isnan(channel_data)]
                    if len(valid) == 0:
                        continue
                    n = len(valid)
                    batch_mean = valid.mean()
                    batch_var = valid.var()
                    delta = batch_mean - mean[c]
                    new_count = count[c] + n
                    mean[c] += delta * n / new_count
                    M2[c] += batch_var * n + delta ** 2 * count[c] * n / new_count
                    count[c] = new_count
        else:
            typhoon_dir = os.path.join(data_root, tid)
            if not os.path.isdir(typhoon_dir):
                continue

            nc_files = sorted(glob.glob(os.path.join(typhoon_dir, "era5_merged_*.nc")))
            if len(nc_files) > max_files_per_typhoon:
                indices = np.linspace(0, len(nc_files) - 1, max_files_per_typhoon, dtype=int)
                nc_files = [nc_files[j] for j in indices]

            for filepath in nc_files:
                try:
                    arr = _load_single_nc_to_array(filepath, pl_vars, sfc_vars, pressure_levels)
                    step = arr[0]
                    step = _crop_pad_to_target(step, 40, 40)

                    for c in range(n_channels):
                        channel_data = step[c].ravel().astype(np.float64)
                        valid = channel_data[~np.isnan(channel_data)]
                        if len(valid) == 0:
                            continue
                        n = len(valid)
                        batch_mean = valid.mean()
                        batch_var = valid.var()
                        delta = batch_mean - mean[c]
                        new_count = count[c] + n
                        mean[c] += delta * n / new_count
                        M2[c] += batch_var * n + delta ** 2 * count[c] * n / new_count
                        count[c] = new_count
                except Exception as e:
                    logger.warning(f"统计时跳过 {filepath}: {e}")

    std = np.sqrt(M2 / np.maximum(count - 1, 1))
    return mean.astype(np.float32), std.astype(np.float32)

def split_typhoon_ids(
    data_root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    all_dirs = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    rng = np.random.RandomState(seed)
    rng.shuffle(all_dirs)

    n = len(all_dirs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = all_dirs[:n_train]
    val_ids = all_dirs[n_train : n_train + n_val]
    test_ids = all_dirs[n_train + n_val :]

    logger.info(
        f"数据集划分: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    return train_ids, val_ids, test_ids

def _extract_year_from_id(typhoon_id: str) -> Optional[int]:
    try:
        return int(typhoon_id[:4])
    except (ValueError, IndexError):
        return None

def split_typhoon_ids_by_year(
    data_root: str,
    train_years: Tuple[int, int] = (1950, 2016),
    val_years: Tuple[int, int] = (2017, 2018),
    test_years: Tuple[int, int] = (2019, 2021),
) -> Tuple[List[str], List[str], List[str]]:
    all_dirs = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    train_ids, val_ids, test_ids = [], [], []
    skipped = 0

    for tid in all_dirs:
        year = _extract_year_from_id(tid)
        if year is None:
            skipped += 1
            continue
        if train_years[0] <= year <= train_years[1]:
            train_ids.append(tid)
        elif val_years[0] <= year <= val_years[1]:
            val_ids.append(tid)
        elif test_years[0] <= year <= test_years[1]:
            test_ids.append(tid)

    logger.info(
        f"数据集按年份划分: "
        f"train={len(train_ids)} ({train_years[0]}-{train_years[1]}), "
        f"val={len(val_ids)} ({val_years[0]}-{val_years[1]}), "
        f"test={len(test_ids)} ({test_years[0]}-{test_years[1]})"
    )
    if skipped:
        logger.info(f"  跳过 {skipped} 个无法解析年份的目录")
    return train_ids, val_ids, test_ids

def build_dataloaders(
    data_cfg,
    train_cfg=None,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    split_by_year: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    preprocessed_dir = getattr(data_cfg, 'preprocessed_dir', None)

    if split_by_year:
        train_ids, val_ids, test_ids = split_typhoon_ids_by_year(
            data_cfg.data_root,
        )
    else:
        train_ids, val_ids, test_ids = split_typhoon_ids(
            data_cfg.data_root,
            data_cfg.train_ratio,
            data_cfg.val_ratio,
            seed=train_cfg.seed if train_cfg else 42,
        )

    if norm_mean is None or norm_std is None:
        stats_path = data_cfg.norm_stats_path
        if stats_path and os.path.exists(stats_path):
            logger.info(f"加载归一化统计: {stats_path}")
            stats = torch.load(stats_path, weights_only=True)
            norm_mean = stats["mean"].numpy()
            norm_std = stats["std"].numpy()
        else:
            logger.info("计算归一化统计（仅使用训练集）...")
            norm_mean, norm_std = compute_normalization_stats(
                train_ids,
                data_cfg.data_root,
                data_cfg.pressure_level_vars,
                data_cfg.surface_vars,
                data_cfg.pressure_levels,
                preprocessed_dir=preprocessed_dir,
            )
            if stats_path:
                os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
                torch.save(
                    {"mean": torch.from_numpy(norm_mean), "std": torch.from_numpy(norm_std)},
                    stats_path,
                )
                logger.info(f"归一化统计已保存: {stats_path}")

    batch_size = train_cfg.batch_size if train_cfg else 16

    common_kwargs = dict(
        data_root=data_cfg.data_root,
        pl_vars=data_cfg.pressure_level_vars,
        sfc_vars=data_cfg.surface_vars,
        pressure_levels=data_cfg.pressure_levels,
        history_steps=data_cfg.history_steps,
        forecast_steps=data_cfg.forecast_steps,
        norm_mean=norm_mean,
        norm_std=norm_std,
        preprocessed_dir=preprocessed_dir,
    )

    train_dataset = ERA5TyphoonDataset(train_ids, **common_kwargs)
    val_dataset = ERA5TyphoonDataset(val_ids, **common_kwargs)
    test_dataset = ERA5TyphoonDataset(test_ids, **common_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=data_cfg.prefetch_factor,
        drop_last=True,
        persistent_workers=True if data_cfg.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=False,
        persistent_workers=True if data_cfg.num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, norm_mean, norm_std
