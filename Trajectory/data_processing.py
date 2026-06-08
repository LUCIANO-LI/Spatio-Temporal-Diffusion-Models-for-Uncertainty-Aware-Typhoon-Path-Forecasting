import os
import glob
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime

from config import data_cfg
from data_structures import StormSample

def load_typhoon_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    column_mapping = {
        'typhoon_id': 'storm_id',
        'wind': 'vmax',
        'pressure': 'pmin'
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    if 'time' in df.columns:
        time_sample = df['time'].iloc[0]
        if isinstance(time_sample, (int, np.integer)) or (isinstance(time_sample, str) and time_sample.isdigit()):
            df = _convert_tyc_time_format(df)
        else:
            df['time'] = pd.to_datetime(df['time'])

    df = df.sort_values(['storm_id'], kind='mergesort').reset_index(drop=True)

    return df

def _convert_tyc_time_format(df: pd.DataFrame) -> pd.DataFrame:
    def parse_tyc_time(row):
        time_val = int(row['time'])
        year = int(row['year']) if 'year' in row else 2020

        hour = time_val % 100
        time_val //= 100
        month = time_val % 100
        day = time_val // 100

        if day == 0:
            day = 1
        if month == 0:
            month = 1

        try:
            result = pd.Timestamp(year=year, month=month, day=day, hour=hour)
            return result
        except:
            return pd.NaT

    df['time'] = df.apply(parse_tyc_time, axis=1)

    return df

def load_era5_for_storm(storm_id: str, era5_dir: str) -> Optional[xr.Dataset]:
    nc_path = Path(era5_dir) / f"{storm_id}.nc"
    zarr_path = Path(era5_dir) / f"{storm_id}.zarr"
    
    if nc_path.exists():
        return xr.open_dataset(nc_path)
    elif zarr_path.exists():
        return xr.open_zarr(zarr_path)
    else:
        print(f"Warning: ERA5 data not found for storm {storm_id}")
        return None

def mark_real_vs_interpolated(times: np.ndarray, original_resolution_hours: float = 3.0) -> np.ndarray:
    is_real = np.zeros(len(times), dtype=bool)
    
    for i, t in enumerate(times):
        ts = pd.Timestamp(t)
        if ts.hour % 3 == 0 and ts.minute == 0:
            is_real[i] = True
    
    return is_real

def align_track_and_era5(
    track_df: pd.DataFrame,
    era5_ds: xr.Dataset,
    time_tolerance_minutes: int = 5
) -> Tuple[pd.DataFrame, xr.Dataset]:
    # 时间对齐
    track_times = track_df['time'].values
    era5_times = era5_ds['valid_time'].values
    
    common_start = max(track_times.min(), era5_times.min())
    common_end = min(track_times.max(), era5_times.max())
    
    track_df = track_df[
        (track_df['time'] >= common_start) & 
        (track_df['time'] <= common_end)
    ].copy()
    
    era5_ds = era5_ds.sel(
        valid_time=slice(common_start, common_end)
    )
    
    return track_df, era5_ds

def create_storm_sample(
    storm_id: str,
    track_df: pd.DataFrame,
    era5_ds: Optional[xr.Dataset] = None
) -> StormSample:
    storm_data = track_df[track_df['storm_id'] == storm_id].copy()

    times = storm_data['time'].values
    track_lat = storm_data['lat'].values.astype(np.float32)
    track_lon = storm_data['lon'].values.astype(np.float32)
    track_vmax = storm_data['vmax'].values.astype(np.float32)
    track_pmin = storm_data['pmin'].values.astype(np.float32) if 'pmin' in storm_data else None
    
    is_real = mark_real_vs_interpolated(times)
    
    lat_grid = None
    lon_grid = None
    if era5_ds is not None:
        lat_grid = era5_ds['latitude'].values
        lon_grid = era5_ds['longitude'].values
    
    year = pd.Timestamp(times[0]).year if len(times) > 0 else None
    
    basin = storm_data['basin'].iloc[0] if 'basin' in storm_data else None
    
    return StormSample(
        storm_id=storm_id,
        times=times,
        track_lat=track_lat,
        track_lon=track_lon,
        track_vmax=track_vmax,
        track_pmin=track_pmin,
        era5_dataset=era5_ds,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        is_real=is_real,
        basin=basin,
        year=year
    )

def load_all_storms(
    csv_path: str = None,
    era5_dir: str = None,
    storm_ids: Optional[List[str]] = None
) -> List[StormSample]:
    csv_path = csv_path or data_cfg.csv_path
    era5_dir = era5_dir or data_cfg.era5_dir
    
    track_df = load_typhoon_csv(csv_path)
    
    if storm_ids is None:
        storm_ids = track_df['storm_id'].unique().tolist()
    
    samples = []
    for sid in storm_ids:
        era5_ds = load_era5_for_storm(sid, era5_dir)
        
        storm_track = track_df[track_df['storm_id'] == sid]
        if era5_ds is not None:
            storm_track, era5_ds = align_track_and_era5(storm_track, era5_ds)
        
        sample = create_storm_sample(sid, storm_track, era5_ds)
        
        if len(sample) > 0:
            samples.append(sample)
    
    print(f"Loaded {len(samples)} storm samples")
    return samples

def load_tyc_storms(
    csv_path: str = None,
    era5_base_dir: str = None,
    storm_ids: Optional[List[str]] = None,
) -> List[StormSample]:
    csv_path = csv_path or data_cfg.csv_path
    era5_base_dir = era5_base_dir or data_cfg.era5_dir

    track_df = load_typhoon_csv(csv_path)
    print(f"Loaded CSV with {len(track_df)} records, {track_df['storm_id'].nunique()} unique storms")

    if storm_ids is None:
        storm_ids = []
        era5_base = Path(era5_base_dir)
        if era5_base.exists():
            for folder in era5_base.iterdir():
                if folder.is_dir():
                    sid = folder.name.replace('_chazhi_finetuned', '')
                    if sid in track_df['storm_id'].values:
                        storm_ids.append(sid)
        print(f"Found {len(storm_ids)} storms with ERA5 data: {storm_ids[:10]}..." if len(storm_ids) > 10 else f"Found {len(storm_ids)} storms with ERA5 data: {storm_ids}")

    if not storm_ids:
        print("No storms found with ERA5 data, falling back to CSV-only mode")
        storm_ids = track_df['storm_id'].unique().tolist()[:5]

    samples = []
    skipped = 0
    from tqdm import tqdm
    for sid in tqdm(storm_ids, desc="Loading storms"):
        sample = load_single_tyc_storm(sid, track_df, era5_base_dir, verbose=False)
        if sample is not None and len(sample) >= 24:
            samples.append(sample)
        else:
            skipped += 1

    print(f"Total loaded: {len(samples)} storm samples (skipped {skipped} due to insufficient data)")
    return samples

def load_single_tyc_storm(
    storm_id: str,
    track_df: pd.DataFrame,
    era5_base_dir: str,
    verbose: bool = True,
    npy_dir: str = "preprocessed_era5",
) -> Optional[StormSample]:
    storm_data = track_df[track_df['storm_id'] == storm_id].copy()
    if len(storm_data) == 0:
        if verbose:
            print(f"  Warning: No track data for {storm_id}")
        return None

    era5_array = None
    lat_grid = None
    lon_grid = None

    npy_path = Path(npy_dir) / f"{storm_id}.npy"
    times_npy_path = Path(npy_dir) / f"{storm_id}_times.npy"

    if npy_path.exists() and times_npy_path.exists():
        era5_all = np.load(npy_path)
        era5_times_ns = np.load(times_npy_path)
        era5_timestamps = [pd.Timestamp(t) for t in era5_times_ns]

        # 时间对齐
        matched_data = []
        matched_era5_idx = []

        for idx, row in storm_data.iterrows():
            track_time = pd.Timestamp(row['time'])
            best_idx = None
            min_diff = timedelta(minutes=31)

            for i, era5_t in enumerate(era5_timestamps):
                diff = abs(track_time - era5_t)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i

            if best_idx is not None and min_diff <= timedelta(minutes=30):
                matched_data.append(row)
                matched_era5_idx.append(best_idx)

        if matched_data:
            storm_data = pd.DataFrame(matched_data)
            era5_array = era5_all[matched_era5_idx]

    else:
        era5_folder = Path(era5_base_dir) / storm_id
        if not era5_folder.exists():
            era5_folder = Path(era5_base_dir) / f"{storm_id}_chazhi_finetuned"

        if era5_folder.exists():
            nc_files = sorted(glob.glob(str(era5_folder / "era5_merged_*.nc")))

            if nc_files:
                nc_times = []
                for nc_file in nc_files:
                    nc_time = parse_nc_filename_time(nc_file)
                    if nc_time is not None:
                        nc_times.append((nc_time, nc_file))

                if nc_times:
                    nc_times.sort(key=lambda x: x[0])

                    matched_data = []
                    matched_files = []

                    for idx, row in storm_data.iterrows():
                        track_time = pd.Timestamp(row['time'])
                        best_match = None
                        min_diff = timedelta(minutes=31)

                        for nc_time, nc_file in nc_times:
                            diff = abs(track_time - nc_time)
                            if diff < min_diff:
                                min_diff = diff
                                best_match = (nc_time, nc_file)

                        if best_match is not None and min_diff <= timedelta(minutes=30):
                            matched_data.append(row)
                            matched_files.append(best_match[1])

                    if matched_data:
                        storm_data = pd.DataFrame(matched_data)

                        era5_frames = []
                        for nc_file in matched_files:
                            frame = load_era5_frame(nc_file)
                            if frame is not None:
                                era5_frames.append(frame)

                        if era5_frames and len(era5_frames) == len(matched_files):
                            era5_array = np.stack(era5_frames, axis=0)
                            ds = xr.open_dataset(matched_files[0])
                            lat_grid = ds['latitude'].values
                            lon_grid = ds['longitude'].values
                            ds.close()

    if len(storm_data) == 0:
        return None

    times = storm_data['time'].values
    track_lat = storm_data['lat'].values.astype(np.float32)
    track_lon = storm_data['lon'].values.astype(np.float32)
    track_vmax = storm_data['vmax'].values.astype(np.float32)
    track_pmin = storm_data['pmin'].values.astype(np.float32) if 'pmin' in storm_data.columns else None

    is_real = mark_real_vs_interpolated(times)

    year = pd.Timestamp(times[0]).year if len(times) > 0 else None

    return StormSample(
        storm_id=storm_id,
        times=times,
        track_lat=track_lat,
        track_lon=track_lon,
        track_vmax=track_vmax,
        track_pmin=track_pmin,
        era5_array=era5_array,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        is_real=is_real,
        year=year
    )

def parse_nc_filename_time(nc_path: str) -> Optional[pd.Timestamp]:
    filename = Path(nc_path).name
    try:
        core = filename.replace('era5_merged_', '').replace('.nc', '')
        parts = core.split('_')

        if 'fused' in parts:
            date_str = parts[0]
            time_str = parts[2]

            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4]) if len(time_str) >= 4 else 0
        else:
            datetime_str = parts[0]
            year = int(datetime_str[:4])
            month = int(datetime_str[4:6])
            day = int(datetime_str[6:8])
            hour = int(datetime_str[8:10])
            minute = 0

        return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
    except Exception as e:
        pass
    return None

def load_era5_frame(nc_path: str, target_size: tuple = None) -> Optional[np.ndarray]:
    if target_size is None:
        target_size = (data_cfg.grid_height, data_cfg.grid_width)

    try:
        ds = xr.open_dataset(nc_path)
        channels = []

        for var in data_cfg.era5_3d_vars:
            if var in ds:
                for level in data_cfg.pressure_levels:
                    try:
                        if 'pressure_level' in ds[var].dims:
                            data = ds[var].sel(pressure_level=level).values
                        elif 'level' in ds[var].dims:
                            data = ds[var].sel(level=level).values
                        else:
                            continue

                        if data.ndim == 3:
                            data = data[0]
                        channels.append(data[np.newaxis].astype(np.float32))
                    except (KeyError, ValueError):
                        try:
                            if 'pressure_level' in ds[var].dims:
                                avail = ds[var].pressure_level.values
                            elif 'level' in ds[var].dims:
                                avail = ds[var].level.values
                            else:
                                continue
                            nearest = avail[np.argmin(np.abs(avail - level))]
                            data = ds[var].sel(
                                **{('pressure_level' if 'pressure_level' in ds[var].dims else 'level'): nearest}
                            ).values
                            if data.ndim == 3:
                                data = data[0]
                            channels.append(data[np.newaxis].astype(np.float32))
                        except Exception:
                            pass

        for var in data_cfg.era5_2d_vars:
            if var in ds:
                data = ds[var].values
                if data.ndim == 3:
                    data = data[0]
                channels.append(data[np.newaxis].astype(np.float32))

        ds.close()

        if channels:
            result = np.concatenate(channels, axis=0)
            if result.shape[-2:] != target_size:
                result = _resize_era5_array(result, target_size)
            return result
        else:
            return None
    except Exception as e:
        print(f"  Error loading {nc_path}: {e}")
    return None

def _resize_era5_array(arr: np.ndarray, target_size: tuple) -> np.ndarray:
    c, h, w = arr.shape
    th, tw = target_size

    result = np.zeros((c, th, tw), dtype=arr.dtype)

    src_h_start = max(0, (h - th) // 2)
    src_w_start = max(0, (w - tw) // 2)
    dst_h_start = max(0, (th - h) // 2)
    dst_w_start = max(0, (tw - w) // 2)

    copy_h = min(h, th)
    copy_w = min(w, tw)

    result[:, dst_h_start:dst_h_start+copy_h, dst_w_start:dst_w_start+copy_w] = \
        arr[:, src_h_start:src_h_start+copy_h, src_w_start:src_w_start+copy_w]

    return result

