from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from paper_eval_common import (
    add_common_arguments,
    aggregate_table2_predictions,
    apply_report_view,
    build_report_indices,
    collect_saved_prediction_samples,
    heuristic_forecast_start_idx,
    infer_end_to_end_era5_channels,
    infer_era5_channels_from_saved_entries,
    load_diffusion_runtime,
    load_track_data,
    load_trajectory_model,
    prepare_future_era5_for_traj,
    print_summary,
    resolve_case_key_for_saved,
    resolve_track_csv,
    run_trajectory_prediction_once,
    save_results,
    select_typhoon_ids,
    _extract_year,
    traj_data_cfg,
    traj_model_cfg,
    normalize_coords,
    validate_args,
)

from dataset import normalize_era5

def build_history_tensor(history_lat: np.ndarray, history_lon: np.ndarray, device: str) -> torch.Tensor:
    h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
    history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)
    return torch.from_numpy(history_coords).float().unsqueeze(0).to(device)

def build_past_era5_tensor(
    storm_id: str,
    forecast_start_idx: int,
    t_history: int,
    era5_channels: int,
    preprocess_dir: str,
    device: str,
) -> torch.Tensor:
    past_era5 = np.zeros((t_history, era5_channels, 40, 40), dtype=np.float32)

    if preprocess_dir is not None:
        npy_path = os.path.join(preprocess_dir, f"{storm_id}.npy")
        if os.path.exists(npy_path):
            data = np.load(npy_path, mmap_mode='r')
            history_start = max(0, forecast_start_idx - t_history)
            history_end = forecast_start_idx
            raw = np.array(data[history_start:history_end])
            if raw.shape[1] > era5_channels:
                raw = raw[:, :era5_channels]
            n_frames = min(len(raw), t_history)
            past_era5[-n_frames:] = raw[-n_frames:]

    past_era5 = normalize_era5(past_era5)
    return torch.from_numpy(past_era5).float().unsqueeze(0).to(device)

def select_matching_sample_idx(
    available_samples: Sequence[int],
    forecast_start_idx: int,
    diff_history_steps: int,
) -> int:
    best_sample_idx = available_samples[0]
    target_sample_pos = max(0, forecast_start_idx - diff_history_steps)
    if len(available_samples) == 1:
        return best_sample_idx

    best_dist = float("inf")
    first_idx = available_samples[0]
    for sample_idx in available_samples:
        sample_pos = sample_idx - first_idx
        dist = abs(sample_pos - target_sample_pos)
        if dist < best_dist:
            best_dist = dist
            best_sample_idx = sample_idx
    return best_sample_idx

def run_end_to_end(args) -> List[Dict[str, np.ndarray]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff_data_cfg, diff_infer_cfg, predictor, ERA5TyphoonDataset, split_typhoon_ids, split_typhoon_ids_by_year = load_diffusion_runtime(args, device)

    if getattr(args, 'split_by_year', False):
        _, _, test_ids = split_typhoon_ids_by_year(
            diff_data_cfg.data_root,
            train_years=(1950, 2016),
            val_years=(2017, 2018),
            test_years=(2019, 2021),
        )
        print(f"[数据划分] 按年份: test=2019-2021, {len(test_ids)} 个台风")
    else:
        _, _, test_ids = split_typhoon_ids(diff_data_cfg.data_root, seed=42)
    if args.num_typhoons and args.num_typhoons > 0:
        test_ids = test_ids[: args.num_typhoons]

    test_dataset = ERA5TyphoonDataset(
        typhoon_ids=test_ids,
        data_root=diff_data_cfg.data_root,
        pl_vars=diff_data_cfg.pressure_level_vars,
        sfc_vars=diff_data_cfg.surface_vars,
        pressure_levels=diff_data_cfg.pressure_levels,
        history_steps=diff_data_cfg.history_steps,
        forecast_steps=diff_data_cfg.forecast_steps,
        norm_mean=torch.load(args.norm_stats, weights_only=True, map_location="cpu")["mean"].numpy(),
        norm_std=torch.load(args.norm_stats, weights_only=True, map_location="cpu")["std"].numpy(),
        preprocessed_dir=args.preprocess_dir,
    )

    era5_channels = infer_end_to_end_era5_channels(diff_data_cfg)
    traj_model = load_trajectory_model(
        checkpoint_path=args.trajectory_ckpt,
        era5_channels=era5_channels,
        device=device,
        bias_path=args.bias_path,
    )

    typhoon_samples: Dict[str, List[int]] = {}
    for idx in range(len(test_dataset)):
        storm_id = test_dataset[idx]["typhoon_id"]
        typhoon_samples.setdefault(storm_id, []).append(idx)

    selected_typhoons = select_typhoon_ids(sorted(typhoon_samples.keys()), args.target_typhoon_ids, min_year=args.min_year)

    if args.exclude_file and os.path.exists(args.exclude_file):
        with open(args.exclude_file, 'r') as f:
            exclude_ids = set(line.strip() for line in f if line.strip())
        before = len(selected_typhoons)
        selected_typhoons = [t for t in selected_typhoons if t not in exclude_ids]
        print(f"[排除] 已排除 {before - len(selected_typhoons)} 个台风 (来自 {args.exclude_file}), 剩余 {len(selected_typhoons)}")

    track_csv_path = resolve_track_csv(args.track_csv)
    report_indices = build_report_indices(
        total_steps=traj_model_cfg.t_future,
        base_resolution_hours=traj_data_cfg.time_resolution_hours,
        report_every_hours=args.report_every_hours,
    )

    results: List[Dict[str, np.ndarray]] = []
    lat_range = traj_data_cfg.lat_range
    lon_range = traj_data_cfg.lon_range

    print(f"[Table 2] 开始 end_to_end 评估，台风数: {len(selected_typhoons)}，每个 case 采样: {args.num_samples}")
    for count, storm_id in enumerate(selected_typhoons, 1):
        print(f"\n[{count}/{len(selected_typhoons)}] 台风 {storm_id}")
        track_df = load_track_data(track_csv_path, storm_id)
        if track_df is None or len(track_df) < traj_model_cfg.t_history + traj_model_cfg.t_future:
            print("  跳过：轨迹数据不足")
            continue

        forecast_start_idx = heuristic_forecast_start_idx(track_df, traj_model_cfg.t_history, traj_model_cfg.t_future)
        if forecast_start_idx < traj_model_cfg.t_history:
            print("  跳过：起报点无效")
            continue

        best_sample_idx = select_matching_sample_idx(
            available_samples=typhoon_samples[storm_id],
            forecast_start_idx=forecast_start_idx,
            diff_history_steps=diff_data_cfg.history_steps,
        )
        sample = test_dataset[best_sample_idx]
        print(f"  使用 ERA5 sample index: {best_sample_idx}")

        history_lat, history_lon = sample_history = (
            track_df["lat"].values[forecast_start_idx - traj_model_cfg.t_history : forecast_start_idx].astype(np.float32),
            track_df["lon"].values[forecast_start_idx - traj_model_cfg.t_history : forecast_start_idx].astype(np.float32),
        )
        if len(sample_history[0]) != traj_model_cfg.t_history:
            print("  跳过：历史窗口不足")
            continue
        history_coords_t = build_history_tensor(history_lat, history_lon, device)

        gt_lat = track_df["lat"].values[forecast_start_idx : forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lon = track_df["lon"].values[forecast_start_idx : forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lat = gt_lat[: traj_model_cfg.t_future]
        gt_lon = gt_lon[: traj_model_cfg.t_future]

        train_lat_range = (0.0, 60.0)
        train_lon_range = (100.0, 180.0)
        window_lat = np.concatenate([history_lat, gt_lat])
        window_lon = np.concatenate([history_lon, gt_lon])
        if (window_lat.min() < train_lat_range[0] or window_lat.max() > train_lat_range[1]
                or window_lon.min() < train_lon_range[0] or window_lon.max() > train_lon_range[1]):
            print(f"  跳过：预测窗口超出训练域 "
                  f"lat[{window_lat.min():.1f},{window_lat.max():.1f}] "
                  f"lon[{window_lon.min():.1f},{window_lon.max():.1f}] "
                  f"(训练域 lat{train_lat_range} lon{train_lon_range})")
            continue

        cond = sample["condition"].unsqueeze(0).to(device)
        sample_results = []

        past_era5_t = build_past_era5_tensor(
            storm_id=storm_id,
            forecast_start_idx=forecast_start_idx,
            t_history=traj_model_cfg.t_history,
            era5_channels=era5_channels,
            preprocess_dir=args.preprocess_dir,
            device=device,
        )

        sample_batch = min(args.num_samples, getattr(args, 'sample_batch_size', 4))
        num_done = 0
        while num_done < args.num_samples:
            n_this_batch = min(sample_batch, args.num_samples - num_done)
            cond_batched = cond.repeat(n_this_batch, 1, 1, 1)

            with torch.no_grad():
                preds = predictor.predict_autoregressive(
                    cond_batched,
                    num_steps=traj_model_cfg.t_future,
                    noise_sigma=diff_infer_cfg.autoregressive_noise_sigma,
                    ensemble_per_step=diff_infer_cfg.ar_ensemble_per_step,
                )

            for b in range(n_this_batch):
                preds_b = [p[b:b+1] for p in preds]
                future_era5 = prepare_future_era5_for_traj(
                    torch.stack(preds_b, dim=1),
                    t_future=traj_model_cfg.t_future,
                    era5_channels=era5_channels,
                    device=device,
                )
                sample_results.append(
                    run_trajectory_prediction_once(
                        traj_model=traj_model,
                        history_coords_t=history_coords_t,
                        future_era5_for_traj=future_era5,
                        gt_lat=gt_lat,
                        gt_lon=gt_lon,
                        lat_range=lat_range,
                        lon_range=lon_range,
                        past_era5_for_traj=past_era5_t,
                    )
                )

            num_done += n_this_batch
            print(f"  采样 {num_done:02d}/{args.num_samples} 完成 (batch={n_this_batch})")

        aggregated = aggregate_table2_predictions(sample_results, gt_lat=gt_lat, gt_lon=gt_lon)
        aggregated = apply_report_view(aggregated, report_indices, args.report_every_hours)
        aggregated["storm_id"] = storm_id
        aggregated["case_key"] = f"{storm_id}::forecast_start={forecast_start_idx}"
        results.append(aggregated)
        print(
            f"  [Table 2] mean={aggregated['error_km'].mean():.2f}km | "
            f"+24h={aggregated['error_km'][aggregated['report_hours'].tolist().index(24)] if 24 in aggregated['report_hours'] else float('nan'):.2f}km | "
            f"+72h={aggregated['error_km'][-1]:.2f}km"
        )

    return results

def run_from_saved(args) -> List[Dict[str, np.ndarray]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ar_files = sorted(Path(args.diffusion_output_dir).glob("ar_pred_*.pt"))
    if not ar_files:
        raise FileNotFoundError(f"未在 {args.diffusion_output_dir} 找到 ar_pred_*.pt")

    grouped_entries: Dict[str, List[Tuple[Path, dict]]] = {}
    fallback_case_count = 0
    for pt_file in ar_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        case_key, _, fallback_only_typhoon = resolve_case_key_for_saved(pt_file, data)
        grouped_entries.setdefault(case_key, []).append((pt_file, data))
        if fallback_only_typhoon:
            fallback_case_count += 1

    if fallback_case_count:
        print(
            "[警告] 保存样本缺少 forecast_start_idx/case_id 等字段；"
            "from_saved 将按 typhoon_id 分组。若同一台风含多个起报点文件，请先补元数据。"
        )

    first_entries = next(iter(grouped_entries.values()))
    era5_channels = infer_era5_channels_from_saved_entries(first_entries)
    traj_model = load_trajectory_model(
        checkpoint_path=args.trajectory_ckpt,
        era5_channels=era5_channels,
        device=device,
        bias_path=args.bias_path,
    )

    selected_case_keys = sorted(grouped_entries.keys())
    if args.target_typhoon_ids:
        target_set = set(args.target_typhoon_ids)
        selected_case_keys = [
            case_key for case_key in selected_case_keys
            if any(str(data.get("typhoon_id") or data.get("storm_id") or "") in target_set for _, data in grouped_entries[case_key])
        ]

    if args.min_year is not None:
        before = len(selected_case_keys)
        filtered = []
        for case_key in selected_case_keys:
            entries = grouped_entries[case_key]
            storm_id = str(entries[0][1].get("typhoon_id") or entries[0][1].get("storm_id") or case_key)
            year = _extract_year(storm_id)
            if year is not None and year >= args.min_year:
                filtered.append(case_key)
        selected_case_keys = filtered
        print(f"[年份过滤] min_year={args.min_year}: {before} -> {len(selected_case_keys)} 个 case (排除 {before - len(selected_case_keys)} 个)")

    track_csv_path = resolve_track_csv(args.track_csv)
    report_indices = build_report_indices(
        total_steps=traj_model_cfg.t_future,
        base_resolution_hours=traj_data_cfg.time_resolution_hours,
        report_every_hours=args.report_every_hours,
    )
    results: List[Dict[str, np.ndarray]] = []
    lat_range = traj_data_cfg.lat_range
    lon_range = traj_data_cfg.lon_range

    print(f"[Table 2] 开始 from_saved 评估，case 数: {len(selected_case_keys)}，每个 case 采样: {args.num_samples}")
    for count, case_key in enumerate(selected_case_keys, 1):
        entries = grouped_entries[case_key]
        first_data = entries[0][1]
        storm_id = str(first_data.get("typhoon_id") or first_data.get("storm_id") or case_key)
        print(f"\n[{count}/{len(selected_case_keys)}] case={case_key}")

        track_df = load_track_data(track_csv_path, storm_id)
        if track_df is None or len(track_df) < traj_model_cfg.t_history + traj_model_cfg.t_future:
            print("  跳过：轨迹数据不足")
            continue

        forecast_start_idx = int(
            first_data.get("forecast_start_idx")
            or first_data.get("start_idx")
            or heuristic_forecast_start_idx(track_df, traj_model_cfg.t_history, traj_model_cfg.t_future)
        )

        history_lat = track_df["lat"].values[forecast_start_idx - traj_model_cfg.t_history : forecast_start_idx].astype(np.float32)
        history_lon = track_df["lon"].values[forecast_start_idx - traj_model_cfg.t_history : forecast_start_idx].astype(np.float32)
        if len(history_lat) != traj_model_cfg.t_history:
            print("  跳过：历史窗口不足")
            continue
        history_coords_t = build_history_tensor(history_lat, history_lon, device)

        gt_lat = track_df["lat"].values[forecast_start_idx : forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lon = track_df["lon"].values[forecast_start_idx : forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lat = gt_lat[: traj_model_cfg.t_future]
        gt_lon = gt_lon[: traj_model_cfg.t_future]

        saved_samples: List[torch.Tensor] = []
        for pt_file, data in entries:
            file_samples = collect_saved_prediction_samples(data)
            saved_samples.extend(file_samples)
            print(f"  从 {pt_file.name} 读取 {len(file_samples)} 个样本")
            if len(saved_samples) >= args.num_samples:
                break

        if not saved_samples:
            print("  跳过：没有可用保存样本")
            continue
        if len(saved_samples) < args.num_samples:
            print(f"  [警告] 仅有 {len(saved_samples)} 个样本，将使用可用子集")

        sample_results = []
        for sample_no, preds_norm in enumerate(saved_samples[: args.num_samples], 1):
            future_era5 = prepare_future_era5_for_traj(
                preds_norm,
                t_future=traj_model_cfg.t_future,
                era5_channels=era5_channels,
                device=device,
            )
            sample_results.append(
                run_trajectory_prediction_once(
                    traj_model=traj_model,
                    history_coords_t=history_coords_t,
                    future_era5_for_traj=future_era5,
                    gt_lat=gt_lat,
                    gt_lon=gt_lon,
                    lat_range=lat_range,
                    lon_range=lon_range,
                )
            )
            print(f"  样本 {sample_no:02d}/{min(args.num_samples, len(saved_samples))} 完成")

        aggregated = aggregate_table2_predictions(sample_results, gt_lat=gt_lat, gt_lon=gt_lon)
        aggregated = apply_report_view(aggregated, report_indices, args.report_every_hours)
        aggregated["storm_id"] = storm_id
        aggregated["case_key"] = case_key
        results.append(aggregated)
        print(
            f"  [Table 2] mean={aggregated['error_km'].mean():.2f}km | "
            f"+72h={aggregated['error_km'][-1]:.2f}km"
        )

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="论文 Table 2 风格评估：20 条轨迹先求均值，再算 FDE")
    add_common_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    validate_args(args)

    if args.mode == "end_to_end":
        results = run_end_to_end(args)
    else:
        results = run_from_saved(args)

    print_summary(results, mode_label="Table 2")
    save_results(args.output_dir, mode_name="table2", results=results)

if __name__ == "__main__":
    main()
