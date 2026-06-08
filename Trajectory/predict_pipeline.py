import os
import sys
import argparse
import importlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAJ_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TRAJ_DIR, ".."))
DIFFUSION_DIR = os.path.join(PROJECT_ROOT, "Diffusion")
DEFAULT_DATA_ROOT = os.environ.get(
    "TYPHOON_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "Typhoon_data_final"),
)
DEFAULT_PREPROCESS_DIR = os.environ.get("TYPHOON_PREPROCESS_DIR")
DEFAULT_NORM_STATS = os.environ.get(
    "TYPHOON_NORM_STATS",
    os.path.join(DIFFUSION_DIR, "norm_stats.pt"),
)
sys.path.insert(0, TRAJ_DIR)

from config import model_cfg as traj_model_cfg, data_cfg as traj_data_cfg
from model import LT3PModel
from dataset import normalize_coords, denormalize_coords

def load_state_dict_allow_missing_past_encoder(model, state_dict, model_name: str):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        'past_physics_encoder.',
        'base_model.past_physics_encoder.',
    )
    bad_missing = [
        key for key in missing
        if not key.startswith(allowed_missing_prefixes)
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model_name}:\n"
            f"  Missing key(s): {bad_missing}\n"
            f"  Unexpected key(s): {unexpected}"
        )
    if missing:
        print(
            f"[Trajectory model] checkpoint lacks new past_physics_encoder "
            f"({len(missing)} keys); demo inference does not pass past_era5, so loading continues."
        )

def load_trajectory_model(checkpoint_path: str, era5_channels: int = 9, device: str = 'cuda',
                          bias_path: str = None):
    # 偏差校正
    model = LT3PModel(
        coord_dim=traj_model_cfg.coord_dim,
        output_dim=traj_model_cfg.output_dim,
        era5_channels=era5_channels,
        t_history=traj_model_cfg.t_history,
        t_future=traj_model_cfg.t_future,
        d_model=traj_model_cfg.transformer_dim,
        n_heads=traj_model_cfg.transformer_heads,
        n_layers=traj_model_cfg.transformer_layers,
        ff_dim=traj_model_cfg.transformer_ff_dim,
        dropout=traj_model_cfg.dropout,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_key = 'ema_model_state_dict' if 'ema_model_state_dict' in ckpt else 'model_state_dict'
    state_dict = ckpt[state_key]
    is_finetune = any(k.startswith('channel_scale') or k.startswith('channel_bias')
                      or k.startswith('adapter.') or k.startswith('base_model.')
                      for k in state_dict.keys())

    if is_finetune:
        has_conv_adapter = any(k.startswith('adapter.') for k in state_dict.keys())

        if has_conv_adapter:
            from finetune_train import ERA5ConvAdaptedModel
            adapted_model = ERA5ConvAdaptedModel(model, era5_channels=era5_channels).to(device)
            print(f"[轨迹模型] 检测到 Conv 适配器 (1×1 Conv bottleneck)")
        else:
            from finetune_train import ERA5AdaptedModel
            adapted_model = ERA5AdaptedModel(model, era5_channels=era5_channels).to(device)
            print(f"[轨迹模型] 检测到 Affine 适配器 (channel scale+bias)")

        load_state_dict_allow_missing_past_encoder(
            adapted_model, state_dict, adapted_model.__class__.__name__
        )
        print(f"[轨迹模型] 已加载微调模型 (epoch {ckpt.get('epoch', '?')}, stage={ckpt.get('stage', '?')})")
        adapted_model.eval()
        n_params = sum(p.numel() for p in adapted_model.parameters())
        print(f"[轨迹模型] 参数量: {n_params:,}")
        final_model = adapted_model
    else:
        load_state_dict_allow_missing_past_encoder(model, state_dict, model.__class__.__name__)
        print(f"[轨迹模型] 已加载{'EMA' if 'ema' in state_key else ''}参数 (epoch {ckpt.get('epoch', '?')})")
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[轨迹模型] 参数量: {n_params:,}")
        final_model = model

    # 偏差校正
    if bias_path and os.path.exists(bias_path):
        from finetune_train import BiasCorrector
        bias = torch.load(bias_path, map_location=device, weights_only=True)
        final_model = BiasCorrector(final_model, bias.to(device))
        final_model.eval()
        print(f"[轨迹模型] 已加载 MOS 偏差校正 ({bias_path})")
        print(f"  最大偏差: lat={bias[:,0].abs().max():.5f}, lon={bias[:,1].abs().max():.5f} (归一化坐标)")

    return final_model

def denormalize_diffusion_output(
    pred_norm: torch.Tensor,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> torch.Tensor:
    mean = torch.from_numpy(norm_mean).float().to(pred_norm.device)
    std = torch.from_numpy(norm_std).float().to(pred_norm.device)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    if pred_norm.ndim == 5:
        mean = mean.reshape(1, 1, -1, 1, 1)
        std = std.reshape(1, 1, -1, 1, 1)
    elif pred_norm.ndim == 4:
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)

    return pred_norm * std + mean

def load_track_data(csv_path: str, storm_id: str) -> Optional[pd.DataFrame]:
    df = pd.read_csv(csv_path)

    col_map = {'typhoon_id': 'storm_id', 'wind': 'vmax', 'pressure': 'pmin'}
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

    track = df[df['storm_id'] == storm_id].copy()
    if len(track) == 0:
        return None

    track = track.sort_index().reset_index(drop=True)
    return track

def extract_track_history(
    track_df: pd.DataFrame,
    forecast_start_idx: int,
    t_history: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    start = forecast_start_idx - t_history
    end = forecast_start_idx

    if start < 0:
        return None, None

    history_lat = track_df['lat'].values[start:end].astype(np.float32)
    history_lon = track_df['lon'].values[start:end].astype(np.float32)
    return history_lat, history_lon

def compute_track_error_km(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    gt_lat: np.ndarray,
    gt_lon: np.ndarray,
) -> np.ndarray:
    lat_err_km = (pred_lat - gt_lat) * 111.0
    lon_err_km = (pred_lon - gt_lon) * 111.0 * np.cos(np.radians(gt_lat))
    return np.sqrt(lat_err_km ** 2 + lon_err_km ** 2)

def plot_track_prediction(
    history_lat, history_lon,
    gt_lat, gt_lon,
    pred_lat, pred_lon,
    storm_id: str,
    error_km: np.ndarray,
    save_path: str,
    era5_source: str = "diffusion",
):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    if has_cartopy:
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())

        all_lats = np.concatenate([history_lat, gt_lat, pred_lat])
        all_lons = np.concatenate([history_lon, gt_lon, pred_lon])
        lat_margin = max(3, (all_lats.max() - all_lats.min()) * 0.2)
        lon_margin = max(3, (all_lons.max() - all_lons.min()) * 0.2)
        extent = [
            max(100, all_lons.min() - lon_margin),
            min(180, all_lons.max() + lon_margin),
            max(0, all_lats.min() - lat_margin),
            min(60, all_lats.max() + lat_margin),
        ]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=(12, 9))
        # cartopy 可选
        transform = None

    plot_kwargs = {'transform': transform} if has_cartopy else {}

    ax.plot(history_lon, history_lat, 'b-', linewidth=2.5, label='History (48h)', zorder=5, **plot_kwargs)
    ax.scatter(history_lon, history_lat, c='blue', s=30, zorder=6, **plot_kwargs)

    full_gt_lon = np.concatenate([[history_lon[-1]], gt_lon])
    full_gt_lat = np.concatenate([[history_lat[-1]], gt_lat])
    ax.plot(full_gt_lon, full_gt_lat, 'g-', linewidth=2.5, label='Ground Truth (72h)', zorder=7, **plot_kwargs)
    ax.scatter(gt_lon, gt_lat, c='green', s=30, zorder=8, **plot_kwargs)

    full_pred_lon = np.concatenate([[history_lon[-1]], pred_lon])
    full_pred_lat = np.concatenate([[history_lat[-1]], pred_lat])
    ax.plot(full_pred_lon, full_pred_lat, 'r--', linewidth=2.5, label='Prediction (72h)', zorder=9, **plot_kwargs)
    ax.scatter(pred_lon, pred_lat, c='red', s=30, marker='x', zorder=10, **plot_kwargs)

    ax.scatter(history_lon[-1], history_lat[-1], c='black', s=100, marker='*',
               label='Forecast Start', zorder=11, **plot_kwargs)

    mean_err = error_km.mean()
    err_24h = error_km[7] if len(error_km) > 7 else error_km[-1]
    err_72h = error_km[-1]

    source_label = "ERA5 from Diffusion Model" if era5_source == "diffusion" else "ERA5 Ground Truth"
    plt.title(
        f"Typhoon {storm_id} — {source_label}\n"
        f"Mean: {mean_err:.1f} km | +24h: {err_24h:.1f} km | +72h: {err_72h:.1f} km",
        fontsize=14, fontweight='bold',
    )
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_error_comparison(
    errors_diffusion: Dict[str, np.ndarray],
    errors_gt: Dict[str, np.ndarray],
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = np.arange(1, 25) * 3

    if errors_gt:
        gt_means = []
        for sid, err in errors_gt.items():
            gt_means.append(err)
        gt_mean = np.mean(gt_means, axis=0)
        ax.plot(hours[:len(gt_mean)], gt_mean, 'g-o', linewidth=2, label='ERA5 GT Input', markersize=4)

    if errors_diffusion:
        diff_means = []
        for sid, err in errors_diffusion.items():
            diff_means.append(err)
        diff_mean = np.mean(diff_means, axis=0)
        ax.plot(hours[:len(diff_mean)], diff_mean, 'r-s', linewidth=2, label='Diffusion Predicted Input', markersize=4)

    ax.set_xlabel('Lead Time (hours)', fontsize=12)
    ax.set_ylabel('Track Error (km)', fontsize=12)
    ax.set_title('Trajectory Prediction Error: GT ERA5 vs Diffusion ERA5', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(hours[::2])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def run_end_to_end(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n[1/5] 加载扩散模型...")
    diff_code = args.diffusion_code
    if diff_code not in sys.path:
        sys.path.insert(0, diff_code)

    from configs import get_config as diff_get_config
    from models import ERA5DiffusionModel
    from train import EMA as DiffEMA
    from data.dataset import ERA5TyphoonDataset, split_typhoon_ids_by_year

    diff_data_cfg, diff_model_cfg, _, diff_infer_cfg = diff_get_config(data_root=args.data_root)

    stats = torch.load(args.norm_stats, weights_only=True, map_location='cpu')
    norm_mean = stats['mean'].numpy()
    norm_std = stats['std'].numpy()

    diff_model = ERA5DiffusionModel(diff_model_cfg, diff_data_cfg).to(device)
    ckpt = torch.load(args.diffusion_ckpt, map_location=device, weights_only=False)
    if 'ema_state_dict' in ckpt:
        ema = DiffEMA(diff_model, decay=0.9999)
        ema.load_state_dict(ckpt['ema_state_dict'])
        ema.apply_shadow(diff_model)
        print("  已加载 EMA 参数")
    else:
        diff_model.load_state_dict(ckpt['model_state_dict'])
    diff_model.eval()

    from inference import ERA5Predictor
    diff_infer_cfg.ddim_steps = args.ddim_steps
    predictor = ERA5Predictor(diff_model, diff_data_cfg, diff_infer_cfg, norm_mean, norm_std, torch.device(device))

    print("\n[2/5] 加载轨迹模型...")
    traj_model = load_trajectory_model(
        args.trajectory_ckpt, era5_channels=9, device=device,
        bias_path=args.bias_path,
    )

    print("\n[3/5] 加载测试数据...")
    _, _, test_ids = split_typhoon_ids_by_year(
        diff_data_cfg.data_root,
        train_years=(1950, 2016),
        val_years=(2017, 2018),
        test_years=(2019, 2021),
    )
    test_ids = test_ids[:args.num_typhoons]

    test_dataset = ERA5TyphoonDataset(
        typhoon_ids=test_ids,
        data_root=diff_data_cfg.data_root,
        pl_vars=diff_data_cfg.pressure_level_vars,
        sfc_vars=diff_data_cfg.surface_vars,
        pressure_levels=diff_data_cfg.pressure_levels,
        history_steps=diff_data_cfg.history_steps,
        forecast_steps=diff_data_cfg.forecast_steps,
        norm_mean=norm_mean,
        norm_std=norm_std,
        preprocessed_dir=args.preprocess_dir,
    )
    print(f"  测试集: {len(test_dataset)} 个样本")

    track_csv_path = args.track_csv
    if not os.path.isabs(track_csv_path):
        track_csv_path = os.path.join(TRAJ_DIR, track_csv_path)

    print("\n[4/5] 运行端到端推理...")
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    typhoon_samples = {}
    for i in range(len(test_dataset)):
        tid = test_dataset[i]['typhoon_id']
        typhoon_samples.setdefault(tid, []).append(i)

    all_typhoon_ids = list(typhoon_samples.keys())
    target_ids = getattr(args, 'target_typhoon_ids', None)
    if target_ids:
        selected_typhoons = [tid for tid in all_typhoon_ids if tid in target_ids]
        print(f"  目标台风: {len(selected_typhoons)} / {len(all_typhoon_ids)}")
    else:
        selected_typhoons = all_typhoon_ids
    print(f"  共 {len(selected_typhoons)} 个测试台风")

    diff_history_steps = diff_data_cfg.history_steps

    for count, typhoon_id in enumerate(selected_typhoons, 1):
        print(f"\n  [{count}/{len(selected_typhoons)}] 台风 {typhoon_id}")

        # 时间对齐
        track_df = load_track_data(track_csv_path, typhoon_id)
        if track_df is None or len(track_df) < traj_model_cfg.t_history + traj_model_cfg.t_future:
            print(f"    轨迹数据不足, 跳过")
            continue

        t_hist = traj_model_cfg.t_history
        t_fut = traj_model_cfg.t_future

        # 时间对齐

        forecast_start_idx = min(t_hist + len(track_df) // 3, len(track_df) - t_fut)
        if forecast_start_idx < t_hist:
            print(f"    轨迹太短, 跳过")
            continue

        available_samples = typhoon_samples[typhoon_id]
        best_sample_idx = available_samples[0]
        target_sample_pos = max(0, forecast_start_idx - diff_history_steps)
        if len(available_samples) > 1:
            best_dist = float('inf')
            for si in available_samples:
                sample_pos = si - available_samples[0]
                dist = abs(sample_pos - target_sample_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_sample_idx = si

        sample_idx = best_sample_idx
        sample = test_dataset[sample_idx]
        era5_sample_offset = sample_idx - available_samples[0]
        print(f"    ERA5样本偏移: {era5_sample_offset}, 轨迹起点: {forecast_start_idx}")

        cond = sample['condition'].unsqueeze(0).to(device)
        with torch.no_grad():
            preds = predictor.predict_autoregressive(
                cond, num_steps=24,
                noise_sigma=diff_infer_cfg.autoregressive_noise_sigma,
            )
        # 归一化空间保持一致

        # 归一化空间保持一致
        future_era5_for_traj = torch.stack(preds, dim=1)

        history_lat, history_lon = extract_track_history(track_df, forecast_start_idx, t_hist)
        if history_lat is None:
            continue

        gt_lat = track_df['lat'].values[forecast_start_idx:forecast_start_idx + t_fut].astype(np.float32)
        gt_lon = track_df['lon'].values[forecast_start_idx:forecast_start_idx + t_fut].astype(np.float32)
        n_gt = min(len(gt_lat), t_fut)
        gt_lat, gt_lon = gt_lat[:n_gt], gt_lon[:n_gt]

        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)
        history_coords_t = torch.from_numpy(history_coords).float().unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = traj_model.predict(history_coords_t, future_era5_for_traj)
        pred_coords = outputs['predicted_coords'].cpu().numpy()[0]

        lat_range = traj_data_cfg.lat_range
        lon_range = traj_data_cfg.lon_range
        pred_lat = pred_coords[:n_gt, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
        pred_lon = pred_coords[:n_gt, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]

        error_km = compute_track_error_km(pred_lat, pred_lon, gt_lat, gt_lon)

        results.append({
            'storm_id': typhoon_id,
            'error_km': error_km,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            'gt_lat': gt_lat,
            'gt_lon': gt_lon,
            'history_lat': history_lat,
            'history_lon': history_lon,
        })

        print(f"    平均误差: {error_km.mean():.1f} km | +24h: {error_km[min(7,n_gt-1)]:.1f} km | +72h: {error_km[-1]:.1f} km")

    print_summary(results)
    top_k = getattr(args, 'top_k', 20)
    visualize_top_k(results, args.output_dir, top_k=top_k)
    return results

def run_from_saved(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n[1/4] 加载归一化统计...")
    stats = torch.load(args.norm_stats, weights_only=True, map_location='cpu')
    norm_mean = stats['mean'].numpy()
    norm_std = stats['std'].numpy()

    print("\n[2/4] 加载轨迹模型...")
    traj_model = load_trajectory_model(
        args.trajectory_ckpt, era5_channels=9, device=device,
        bias_path=args.bias_path,
    )

    print("\n[3/4] 加载扩散模型预测...")
    output_dir = args.diffusion_output_dir
    ar_files = sorted(Path(output_dir).glob("ar_pred_*.pt"))
    if not ar_files:
        print(f"  未找到 ar_pred_*.pt 文件于 {output_dir}")
        return []
    print(f"  找到 {len(ar_files)} 个预测文件")

    track_csv_path = args.track_csv
    if not os.path.isabs(track_csv_path):
        track_csv_path = os.path.join(TRAJ_DIR, track_csv_path)

    print("\n[4/4] 运行轨迹预测...")
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for pt_file in ar_files:
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        typhoon_id = data['typhoon_id']
        # 归一化空间保持一致
        preds_norm = data['predictions']

        print(f"\n  台风: {typhoon_id}, 预测步数: {preds_norm.shape[1]}")

        # 归一化空间保持一致
        future_era5_for_traj = preds_norm.to(device)

        T = future_era5_for_traj.shape[1]
        t_fut = traj_model_cfg.t_future
        if T < t_fut:
            pad = future_era5_for_traj[:, -1:].expand(-1, t_fut - T, -1, -1, -1)
            future_era5_for_traj = torch.cat([future_era5_for_traj, pad], dim=1)
        elif T > t_fut:
            future_era5_for_traj = future_era5_for_traj[:, :t_fut]

        track_df = load_track_data(track_csv_path, typhoon_id)
        if track_df is None:
            print(f"    未找到轨迹数据, 跳过")
            continue

        t_hist = traj_model_cfg.t_history
        total_needed = t_hist + t_fut
        if len(track_df) < total_needed:
            print(f"    轨迹太短 ({len(track_df)} < {total_needed}), 跳过")
            continue

        forecast_start_idx = min(t_hist + len(track_df) // 3, len(track_df) - t_fut)
        if forecast_start_idx < t_hist:
            forecast_start_idx = t_hist

        history_lat, history_lon = extract_track_history(track_df, forecast_start_idx, t_hist)
        if history_lat is None:
            continue

        gt_lat = track_df['lat'].values[forecast_start_idx:forecast_start_idx + t_fut].astype(np.float32)
        gt_lon = track_df['lon'].values[forecast_start_idx:forecast_start_idx + t_fut].astype(np.float32)
        n_gt = min(len(gt_lat), t_fut)
        gt_lat, gt_lon = gt_lat[:n_gt], gt_lon[:n_gt]

        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)
        history_coords_t = torch.from_numpy(history_coords).float().unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = traj_model.predict(history_coords_t, future_era5_for_traj)
        pred_coords = outputs['predicted_coords'].cpu().numpy()[0]

        lat_range = traj_data_cfg.lat_range
        lon_range = traj_data_cfg.lon_range
        pred_lat = pred_coords[:n_gt, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
        pred_lon = pred_coords[:n_gt, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]

        error_km = compute_track_error_km(pred_lat, pred_lon, gt_lat, gt_lon)

        results.append({
            'storm_id': typhoon_id,
            'error_km': error_km,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            'gt_lat': gt_lat,
            'gt_lon': gt_lon,
            'history_lat': history_lat,
            'history_lon': history_lon,
        })

        print(f"    平均误差: {error_km.mean():.1f} km | +24h: {error_km[min(7,n_gt-1)]:.1f} km | +72h: {error_km[-1]:.1f} km")

    print_summary(results)
    top_k = getattr(args, 'top_k', 20)
    visualize_top_k(results, args.output_dir, top_k=top_k)
    return results

def visualize_top_k(results: list, output_dir: str, top_k: int = 20):
    if not results:
        print("\n无结果可可视化")
        return

    sorted_results = sorted(results, key=lambda r: r['error_km'].mean())
    top_results = sorted_results[:top_k]

    print(f"\n{'='*60}")
    print(f"Top {len(top_results)} 预测最准的台风 (按平均误差排序)")
    print(f"{'='*60}")

    top_dir = os.path.join(output_dir, f"top{top_k}")
    os.makedirs(top_dir, exist_ok=True)

    for rank, r in enumerate(top_results, 1):
        mean_err = r['error_km'].mean()
        n_gt = len(r['error_km'])
        err_24h = r['error_km'][min(7, n_gt-1)]
        err_72h = r['error_km'][-1]
        print(f"  #{rank:2d} {r['storm_id']}: 平均={mean_err:.1f}km, +24h={err_24h:.1f}km, +72h={err_72h:.1f}km")

        plot_track_prediction(
            r['history_lat'], r['history_lon'],
            r['gt_lat'], r['gt_lon'],
            r['pred_lat'], r['pred_lon'],
            f"#{rank} {r['storm_id']}",
            r['error_km'],
            os.path.join(top_dir, f"top{rank:02d}_{r['storm_id']}.png"),
            era5_source="diffusion",
        )

    _plot_top_k_summary(top_results, top_dir, top_k)

    _save_ranking_csv(sorted_results, output_dir)

    import pickle
    pkl_path = os.path.join(output_dir, 'pipeline_results.pkl')
    pickle.dump(sorted_results, open(pkl_path, 'wb'))
    print(f"  已保存结果到 {pkl_path}（可用于 typhoon_demo.py）")

def _plot_top_k_summary(top_results: list, top_dir: str, top_k: int):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    n = len(top_results)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    if has_cartopy:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
            subplot_kw={'projection': ccrs.PlateCarree()},
        )
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, r in enumerate(top_results):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        all_lats = np.concatenate([r['history_lat'], r['gt_lat'], r['pred_lat']])
        all_lons = np.concatenate([r['history_lon'], r['gt_lon'], r['pred_lon']])
        lat_margin = max(2, (all_lats.max() - all_lats.min()) * 0.15)
        lon_margin = max(2, (all_lons.max() - all_lons.min()) * 0.15)
        extent = [
            max(95, all_lons.min() - lon_margin), min(185, all_lons.max() + lon_margin),
            max(0, all_lats.min() - lat_margin), min(60, all_lats.max() + lat_margin),
        ]

        plot_kw = {}
        if has_cartopy:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            ax.gridlines(linewidth=0.3, alpha=0.5)
            plot_kw = {'transform': ccrs.PlateCarree()}
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.grid(True, alpha=0.3)

        ax.plot(r['history_lon'], r['history_lat'], 'b-', linewidth=1.5, **plot_kw)
        full_gt_lon = np.concatenate([[r['history_lon'][-1]], r['gt_lon']])
        full_gt_lat = np.concatenate([[r['history_lat'][-1]], r['gt_lat']])
        ax.plot(full_gt_lon, full_gt_lat, 'g-', linewidth=1.5, **plot_kw)
        full_pred_lon = np.concatenate([[r['history_lon'][-1]], r['pred_lon']])
        full_pred_lat = np.concatenate([[r['history_lat'][-1]], r['pred_lat']])
        ax.plot(full_pred_lon, full_pred_lat, 'r--', linewidth=1.5, **plot_kw)
        ax.scatter(r['history_lon'][-1], r['history_lat'][-1], c='black', s=50, marker='*', zorder=11, **plot_kw)

        mean_err = r['error_km'].mean()
        err_72h = r['error_km'][-1]
        ax.set_title(f"#{idx+1} {r['storm_id']}\nMean:{mean_err:.0f}km 72h:{err_72h:.0f}km", fontsize=9)

    for idx in range(n, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='History (48h)'),
        Line2D([0], [0], color='green', linewidth=2, label='Ground Truth (72h)'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Prediction (72h)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)

    plt.suptitle(f'Top {len(top_results)} Best Predictions (by Mean Error)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(top_dir, f'top{len(top_results)}_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  汇总大图已保存: {save_path}")

def _save_ranking_csv(sorted_results: list, output_dir: str):
    rows = []
    for rank, r in enumerate(sorted_results, 1):
        n_gt = len(r['error_km'])
        rows.append({
            'rank': rank,
            'storm_id': r['storm_id'],
            'mean_error_km': round(r['error_km'].mean(), 1),
            'error_24h_km': round(r['error_km'][min(7, n_gt-1)], 1),
            'error_48h_km': round(r['error_km'][min(15, n_gt-1)], 1),
            'error_72h_km': round(r['error_km'][-1], 1),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'all_typhoons_ranking.csv')
    df.to_csv(csv_path, index=False)
    print(f"  排名CSV已保存: {csv_path} ({len(rows)} 个台风)")

def print_summary(results: list):
    if not results:
        print("\n无有效结果")
        return

    print(f"\n{'='*60}")
    print("Pipeline 误差汇总 (ERA5-Diffusion → LT3P Trajectory)")
    print(f"{'='*60}")
    print(f"样本数: {len(results)}")

    all_errors = []
    for r in results:
        all_errors.append(r['error_km'])

    min_len = min(len(e) for e in all_errors)
    all_errors = np.array([e[:min_len] for e in all_errors])

    print(f"\n{'Lead Time':>10} {'Mean (km)':>12} {'Std (km)':>10}")
    print("-" * 36)
    for t in range(min_len):
        hours = (t + 1) * 3
        mean_err = all_errors[:, t].mean()
        std_err = all_errors[:, t].std()
        print(f"  +{hours:2d}h      {mean_err:>10.1f}   {std_err:>8.1f}")

    print(f"\n总体平均误差: {all_errors.mean():.1f} km")

    print("\n关键时间点:")
    for key_h in [24, 48, 72]:
        t_idx = key_h // 3 - 1
        if t_idx < min_len:
            print(f"  +{key_h}h: {all_errors[:, t_idx].mean():.1f} ± {all_errors[:, t_idx].std():.1f} km")

def main():
    parser = argparse.ArgumentParser(
        description="端到端台风轨迹预测: ERA5-Diffusion → LT3P",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["end_to_end", "from_saved"], default="from_saved",
                        help="运行模式: end_to_end=完整pipeline, from_saved=加载预计算扩散预测")

    parser.add_argument("--trajectory_ckpt", type=str, default="checkpoints/best.pt",
                        help="轨迹模型 checkpoint 路径")
    parser.add_argument("--norm_stats", type=str, default=DEFAULT_NORM_STATS,
                        help="扩散模型归一化统计文件 norm_stats.pt")
    parser.add_argument("--track_csv", type=str, default=os.path.join(TRAJ_DIR, "processed_typhoon_tracks.csv"),
                        help="台风轨迹 CSV 文件路径")
    parser.add_argument("--output_dir", type=str, default="pipeline_outputs",
                        help="输出目录")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="测试样本数")
    parser.add_argument("--bias_path", type=str, default=None,
                        help="MOS偏差校正文件路径 (lead_time_bias.pt)，不指定则不校正")
    parser.add_argument("--top_k", type=int, default=20,
                        help="可视化预测最准的前K个台风 (默认20)")

    parser.add_argument("--diffusion_code", type=str, default=DIFFUSION_DIR,
                        help="(end_to_end) 扩散模型代码目录")
    parser.add_argument("--diffusion_ckpt", type=str, default=os.path.join(DIFFUSION_DIR, "checkpoints", "best.pt"),
                        help="(end_to_end) 扩散模型 checkpoint")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                        help="(end_to_end) ERA5 数据根目录；也可通过 TYPHOON_DATA_ROOT 设置")
    parser.add_argument("--preprocess_dir", type=str, default=DEFAULT_PREPROCESS_DIR,
                        help="(end_to_end) 预处理 NPY 目录")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="(end_to_end) DDIM 采样步数")
    parser.add_argument("--num_typhoons", type=int, default=10,
                        help="(end_to_end) 测试台风数")

    parser.add_argument("--diffusion_output_dir", type=str, default=os.path.join(DIFFUSION_DIR, "pipeline_outputs"),
                        help="(from_saved) 扩散模型预测输出目录 (含 ar_pred_*.pt)")

    args = parser.parse_args()

    print("=" * 60)
    print("端到端台风轨迹预测 Pipeline")
    print("ERA5-Diffusion 风场预测 → LT3P 轨迹预测")
    print(f"模式: {args.mode}")
    print("=" * 60)

    if args.mode == "end_to_end":
        assert args.diffusion_code, "--diffusion_code 必须指定"
        assert args.diffusion_ckpt, "--diffusion_ckpt 必须指定"
        assert args.data_root, "--data_root 必须指定"
        run_end_to_end(args)
    elif args.mode == "from_saved":
        assert args.diffusion_output_dir, "--diffusion_output_dir 必须指定"
        run_from_saved(args)

if __name__ == "__main__":
    main()
