from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

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
if TRAJ_DIR not in sys.path:
    sys.path.insert(0, TRAJ_DIR)

from config import data_cfg as traj_data_cfg  # noqa: E402
from config import model_cfg as traj_model_cfg  # noqa: E402
from dataset import normalize_coords  # noqa: E402
from model import LT3PModel  # noqa: E402

def load_trajectory_model(
    checkpoint_path: str,
    era5_channels: int,
    device: str,
    bias_path: Optional[str] = None,
):
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
    state_key = "ema_model_state_dict" if "ema_model_state_dict" in ckpt else "model_state_dict"
    state_dict = ckpt[state_key]

    is_finetune = any(
        key.startswith("channel_scale")
        or key.startswith("channel_bias")
        or key.startswith("adapter.")
        or key.startswith("base_model.")
        for key in state_dict.keys()
    )

    if is_finetune:
        has_conv_adapter = any(key.startswith("adapter.") for key in state_dict.keys())
        if has_conv_adapter:
            from finetune_train import ERA5ConvAdaptedModel

            wrapped_model = ERA5ConvAdaptedModel(model, era5_channels=era5_channels).to(device)
            adapter_name = "Conv adapter"
        else:
            from finetune_train import ERA5AdaptedModel

            wrapped_model = ERA5AdaptedModel(model, era5_channels=era5_channels).to(device)
            adapter_name = "Affine adapter"
        missing, unexpected = wrapped_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[轨迹模型] 忽略缺失的 key ({len(missing)} 个，如 past_physics_encoder 等新增模块)")
        wrapped_model.eval()
        final_model = wrapped_model
        print(f"[轨迹模型] 已加载微调模型 ({adapter_name}, epoch={ckpt.get('epoch', '?')})")
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[轨迹模型] 忽略缺失的 key ({len(missing)} 个)")
        model.eval()
        final_model = model
        print(f"[轨迹模型] 已加载基础模型 (epoch={ckpt.get('epoch', '?')})")

    if bias_path and os.path.exists(bias_path):
        from finetune_train import BiasCorrector

        bias = torch.load(bias_path, map_location=device, weights_only=True)
        final_model = BiasCorrector(final_model, bias.to(device))
        final_model.eval()
        print(f"[轨迹模型] 已加载 MOS 偏差校正: {bias_path}")

    n_params = sum(param.numel() for param in final_model.parameters())
    print(f"[轨迹模型] 参数量: {n_params:,}")
    return final_model

def load_track_data(csv_path: str, storm_id: str) -> Optional[pd.DataFrame]:
    df = pd.read_csv(csv_path)

    col_map = {"typhoon_id": "storm_id", "wind": "vmax", "pressure": "pmin"}
    for old_name, new_name in col_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    track = df[df["storm_id"] == storm_id].copy()
    if len(track) == 0:
        return None
    track = track.sort_index().reset_index(drop=True)

    return track

def extract_track_history(
    track_df: pd.DataFrame,
    forecast_start_idx: int,
    t_history: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    start_idx = forecast_start_idx - t_history
    end_idx = forecast_start_idx
    if start_idx < 0:
        return None, None
    history_lat = track_df["lat"].values[start_idx:end_idx].astype(np.float32)
    history_lon = track_df["lon"].values[start_idx:end_idx].astype(np.float32)
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

def resolve_track_csv(track_csv: str) -> str:
    if os.path.isabs(track_csv):
        return track_csv
    return os.path.join(TRAJ_DIR, track_csv)

def prepare_future_era5_for_traj(
    preds_norm,
    t_future: int,
    era5_channels: Optional[int],
    device: str,
) -> torch.Tensor:
    if isinstance(preds_norm, (list, tuple)):
        preds_norm = torch.stack(list(preds_norm), dim=1)

    future_era5 = preds_norm.to(device)
    if future_era5.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got shape={tuple(future_era5.shape)}")

    if future_era5.shape[1] < t_future:
        pad = future_era5[:, -1:].expand(-1, t_future - future_era5.shape[1], -1, -1, -1)
        future_era5 = torch.cat([future_era5, pad], dim=1)
    elif future_era5.shape[1] > t_future:
        future_era5 = future_era5[:, :t_future]

    if era5_channels is not None and future_era5.shape[2] != era5_channels:
        if future_era5.shape[2] > era5_channels:
            future_era5 = future_era5[:, :, :era5_channels]
        else:
            pad_channels = era5_channels - future_era5.shape[2]
            pad = torch.zeros(
                future_era5.shape[0],
                future_era5.shape[1],
                pad_channels,
                future_era5.shape[3],
                future_era5.shape[4],
                dtype=future_era5.dtype,
                device=future_era5.device,
            )
            future_era5 = torch.cat([future_era5, pad], dim=2)

    return future_era5

def run_trajectory_prediction_once(
    traj_model,
    history_coords_t: torch.Tensor,
    future_era5_for_traj: torch.Tensor,
    gt_lat: np.ndarray,
    gt_lon: np.ndarray,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    past_era5_for_traj: torch.Tensor = None,
) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        outputs = traj_model.predict(history_coords_t, future_era5_for_traj, past_era5=past_era5_for_traj)

    pred_coords = outputs["predicted_coords"].detach().cpu().numpy()[0]
    n_gt = len(gt_lat)
    pred_lat = pred_coords[:n_gt, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
    pred_lon = pred_coords[:n_gt, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]
    error_km = compute_track_error_km(pred_lat, pred_lon, gt_lat, gt_lon)

    return {
        "pred_lat": pred_lat.astype(np.float32),
        "pred_lon": pred_lon.astype(np.float32),
        "error_km_full": error_km.astype(np.float32),
    }

def build_report_indices(
    total_steps: int,
    base_resolution_hours: int,
    report_every_hours: int,
) -> List[int]:
    if report_every_hours % base_resolution_hours != 0:
        raise ValueError(
            f"report_every_hours={report_every_hours} is not divisible by "
            f"base_resolution_hours={base_resolution_hours}"
        )
    stride = report_every_hours // base_resolution_hours
    return list(range(stride - 1, total_steps, stride))

def apply_report_view(
    result: Dict[str, np.ndarray],
    report_indices: Sequence[int],
    report_every_hours: int,
) -> Dict[str, np.ndarray]:
    report_hours = np.array([(idx + 1) * traj_data_cfg.time_resolution_hours for idx in report_indices], dtype=np.int32)
    result = dict(result)
    result["error_km"] = result["error_km_full"][list(report_indices)].astype(np.float32)
    result["report_hours"] = report_hours
    result["report_every_hours"] = int(report_every_hours)
    return result

def aggregate_table2_predictions(
    sample_results: List[Dict[str, np.ndarray]],
    gt_lat: np.ndarray,
    gt_lon: np.ndarray,
) -> Dict[str, np.ndarray]:
    if not sample_results:
        raise ValueError("sample_results must not be empty")

    pred_lat_stack = np.stack([item["pred_lat"] for item in sample_results], axis=0)
    pred_lon_stack = np.stack([item["pred_lon"] for item in sample_results], axis=0)

    pred_lat = pred_lat_stack.mean(axis=0).astype(np.float32)
    pred_lon = pred_lon_stack.mean(axis=0).astype(np.float32)
    error_km = compute_track_error_km(pred_lat, pred_lon, gt_lat, gt_lon).astype(np.float32)

    return {
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "error_km_full": error_km,
        "sample_count": len(sample_results),
        "aggregation": "ensemble_average",
    }

def aggregate_table3_predictions(
    sample_results: List[Dict[str, np.ndarray]],
    selection_strategy: str = "per_lead_min",
) -> Dict[str, np.ndarray]:
    if not sample_results:
        raise ValueError("sample_results must not be empty")

    error_stack = np.stack([item["error_km_full"] for item in sample_results], axis=0)

    if selection_strategy == "per_lead_min":
        representative_idx = int(np.argmin(error_stack[:, -1]))
        representative = dict(sample_results[representative_idx])
        representative["error_km_full"] = error_stack.min(axis=0).astype(np.float32)
        representative["aggregation"] = "oracle_per_lead_min"
        representative["representative_idx"] = representative_idx
        representative["sample_count"] = len(sample_results)
        return representative

    if selection_strategy == "best_72h_traj":
        representative_idx = int(np.argmin(error_stack[:, -1]))
        representative = dict(sample_results[representative_idx])
        representative["aggregation"] = "oracle_best_72h_traj"
        representative["representative_idx"] = representative_idx
        representative["sample_count"] = len(sample_results)
        return representative

    raise ValueError(f"Unsupported Table 3 selection_strategy: {selection_strategy}")

def infer_era5_channels_from_saved_entries(entries: Sequence[Tuple[Path, dict]]) -> int:
    for _, data in entries:
        for key in ("prediction_samples", "predictions_samples", "predictions"):
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, torch.Tensor):
                if value.ndim == 5:
                    return int(value.shape[2])
                if value.ndim == 6:
                    return int(value.shape[3])
            if isinstance(value, (list, tuple)) and value:
                item = value[0]
                if isinstance(item, torch.Tensor):
                    if item.ndim == 4:
                        return int(item.shape[1])
                    if item.ndim == 5:
                        return int(item.shape[2])
    return traj_model_cfg.era5_channels

def load_diffusion_runtime(args, device: str):
    diff_code = args.diffusion_code
    if diff_code not in sys.path:
        sys.path.insert(0, diff_code)
    elif sys.path[0] != diff_code:
        sys.path.remove(diff_code)
        sys.path.insert(0, diff_code)

    from configs import get_config as diff_get_config
    from data.dataset import ERA5TyphoonDataset, split_typhoon_ids, split_typhoon_ids_by_year
    from inference import ERA5Predictor
    from models import ERA5DiffusionModel
    from train import EMA as DiffEMA

    diff_data_cfg, diff_model_cfg, _, diff_infer_cfg = diff_get_config(data_root=args.data_root)
    stats = torch.load(args.norm_stats, weights_only=True, map_location="cpu")
    norm_mean = stats["mean"].numpy()
    norm_std = stats["std"].numpy()

    diff_model = ERA5DiffusionModel(diff_model_cfg, diff_data_cfg).to(device)
    ckpt = torch.load(args.diffusion_ckpt, map_location=device, weights_only=False)
    if "ema_state_dict" in ckpt:
        ema = DiffEMA(diff_model, decay=0.9999)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply_shadow(diff_model)
        print("[扩散模型] 已加载 EMA 参数")
    else:
        diff_model.load_state_dict(ckpt["model_state_dict"])
        print("[扩散模型] 已加载基础参数")
    diff_model.eval()

    diff_infer_cfg.ddim_steps = args.ddim_steps
    predictor = ERA5Predictor(diff_model, diff_data_cfg, diff_infer_cfg, norm_mean, norm_std, torch.device(device))
    return diff_data_cfg, diff_infer_cfg, predictor, ERA5TyphoonDataset, split_typhoon_ids, split_typhoon_ids_by_year

def infer_end_to_end_era5_channels(diff_data_cfg) -> int:
    num_channels = getattr(diff_data_cfg, "num_channels", None)
    if num_channels is not None:
        return int(num_channels)
    pl_vars = getattr(diff_data_cfg, "pressure_level_vars", [])
    sfc_vars = getattr(diff_data_cfg, "surface_vars", [])
    pressure_levels = getattr(diff_data_cfg, "pressure_levels", [])
    if pl_vars and pressure_levels:
        return int(len(pl_vars) * len(pressure_levels) + len(sfc_vars))
    return traj_model_cfg.era5_channels

def resolve_case_key_for_saved(pt_file: Path, data: dict) -> Tuple[str, str, bool]:
    storm_id = str(data.get("typhoon_id") or data.get("storm_id") or pt_file.stem)
    for meta_key in ("case_id", "forecast_start_idx", "start_idx", "sample_idx", "window_start"):
        if meta_key in data:
            return f"{storm_id}::{meta_key}={data[meta_key]}", storm_id, False
    return storm_id, storm_id, True

def collect_saved_prediction_samples(data: dict) -> List[torch.Tensor]:
    raw = None
    for key in ("prediction_samples", "predictions_samples", "predictions"):
        if key in data:
            raw = data[key]
            break

    if raw is None:
        raise KeyError("No prediction tensor found in saved diffusion output")

    samples: List[torch.Tensor] = []
    if isinstance(raw, torch.Tensor):
        if raw.ndim == 5:
            if raw.shape[0] == 1:
                samples.append(raw)
            else:
                samples.extend([raw[i : i + 1] for i in range(raw.shape[0])])
        elif raw.ndim == 6 and raw.shape[1] == 1:
            samples.extend([raw[i] for i in range(raw.shape[0])])
        else:
            raise ValueError(f"Unsupported saved prediction tensor shape: {tuple(raw.shape)}")
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, torch.Tensor):
                raise TypeError("Saved prediction samples must be tensors")
            if item.ndim == 4:
                samples.append(item.unsqueeze(0))
            elif item.ndim == 5:
                samples.append(item if item.shape[0] == 1 else item[:1])
            else:
                raise ValueError(f"Unsupported saved prediction tensor shape: {tuple(item.shape)}")
    else:
        raise TypeError(f"Unsupported saved prediction type: {type(raw)}")
    return samples

def heuristic_forecast_start_idx(track_df: pd.DataFrame, t_hist: int, t_fut: int) -> int:
    return min(t_hist + len(track_df) // 3, len(track_df) - t_fut)

def generate_forecast_starts(
    track_length: int,
    t_history: int,
    t_future: int,
    stride: int = None,
) -> List[int]:
    if stride is None:
        stride = t_future
    return list(range(t_history, track_length - t_future + 1, stride))

def case_result_to_row(result: Dict[str, np.ndarray]) -> Dict[str, object]:
    row = {
        "case_key": result["case_key"],
        "storm_id": result["storm_id"],
        "aggregation": result["aggregation"],
        "sample_count": int(result["sample_count"]),
        "mean_error_km": round(float(result["error_km"].mean()), 3),
    }
    for hour, error in zip(result["report_hours"], result["error_km"]):
        row[f"fde_{int(hour)}h_km"] = round(float(error), 3)
    return row

def save_results(output_dir: str, mode_name: str, results: Sequence[Dict[str, np.ndarray]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if not results:
        summary_path = os.path.join(output_dir, f"{mode_name}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "mode": mode_name,
                    "num_cases": 0,
                    "samples_per_case": 0,
                    "report_hours": [],
                    "overall_mean_error_km": None,
                    "error_by_hour": {},
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[保存] {summary_path}")
        return

    rows = [case_result_to_row(item) for item in results]
    csv_path = os.path.join(output_dir, f"{mode_name}_case_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    all_errors = np.array([item["error_km"] for item in results], dtype=np.float32)
    report_hours = [int(hour) for hour in results[0]["report_hours"]]
    summary = {
        "mode": mode_name,
        "num_cases": int(len(results)),
        "samples_per_case": int(results[0]["sample_count"]),
        "report_hours": report_hours,
        "overall_mean_error_km": float(all_errors.mean()),
        "error_by_hour": {
            f"{hour}h": {
                "mean_km": float(all_errors[:, idx].mean()),
                "std_km": float(all_errors[:, idx].std()),
            }
            for idx, hour in enumerate(report_hours)
        },
    }
    summary_path = os.path.join(output_dir, f"{mode_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[保存] {csv_path}")
    print(f"[保存] {summary_path}")

def print_summary(results: Sequence[Dict[str, np.ndarray]], mode_label: str) -> None:
    if not results:
        print(f"\n[{mode_label}] 没有可用结果。")
        return

    all_errors = np.array([item["error_km"] for item in results], dtype=np.float32)
    report_hours = [int(hour) for hour in results[0]["report_hours"]]

    print("\n" + "=" * 72)
    print(f"{mode_label} | Paper-style FDE summary")
    print("=" * 72)
    print(f"Cases: {len(results)}")
    print(f"Samples per case: {results[0]['sample_count']}")
    print(f"Reported every: {results[0]['report_every_hours']}h")
    print(f"{'Lead Time':>10} {'Mean (km)':>12} {'Std (km)':>10}")
    print("-" * 36)
    for idx, hour in enumerate(report_hours):
        mean_err = float(all_errors[:, idx].mean())
        std_err = float(all_errors[:, idx].std())
        print(f"  +{hour:2d}h      {mean_err:>10.2f}   {std_err:>8.2f}")
    print(f"\nOverall mean over reported horizons: {float(all_errors.mean()):.2f} km")

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["end_to_end", "from_saved"], default="from_saved")
    parser.add_argument("--trajectory_ckpt", type=str, required=True, help="轨迹模型 checkpoint")
    parser.add_argument("--norm_stats", type=str, default=DEFAULT_NORM_STATS, help="扩散模型归一化统计文件")
    parser.add_argument("--track_csv", type=str, default=os.path.join(TRAJ_DIR, "processed_typhoon_tracks.csv"))
    parser.add_argument("--output_dir", type=str, required=True, help="结果输出目录")
    parser.add_argument("--num_samples", type=int, default=20, help="每个 case 的采样数，论文默认 20")
    parser.add_argument("--bias_path", type=str, default=None, help="可选的 MOS 偏差校正文件")
    parser.add_argument(
        "--report_every_hours",
        type=int,
        default=6,
        help="按多少小时报告一次误差；论文 Table 2/3 为 6",
    )
    parser.add_argument(
        "--num_typhoons",
        type=int,
        default=0,
        help="end_to_end 模式评估的测试台风数，0 表示全部",
    )
    parser.add_argument(
        "--target_typhoon_ids",
        nargs="*",
        default=None,
        help="只评估指定台风 ID，可传多个",
    )

    parser.add_argument("--diffusion_code", type=str, default=DIFFUSION_DIR, help="(end_to_end) 扩散模型代码目录")
    parser.add_argument("--diffusion_ckpt", type=str, default=os.path.join(DIFFUSION_DIR, "checkpoints", "best.pt"), help="(end_to_end) 扩散模型 checkpoint")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="(end_to_end) ERA5 数据目录")
    parser.add_argument("--preprocess_dir", type=str, default=DEFAULT_PREPROCESS_DIR, help="(end_to_end) 预处理目录")
    parser.add_argument("--ddim_steps", type=int, default=50, help="(end_to_end) DDIM 采样步数")
    parser.add_argument("--sample_batch_size", type=int, default=4,
                        help="(end_to_end) 采样批量大小, 多个样本并行 DDIM (默认4, OOM就降低)")
    parser.add_argument("--exclude_file", type=str, default=None,
                        help="排除台风列表文件 (每行一个台风ID, 如 excluded_typhoons.txt)")
    parser.add_argument("--min_year", type=int, default=None,
                        help="只评估 >= 该年份的台风 (如 1980，排除卫星时代前数据质量差的早期台风)")

    parser.add_argument("--diffusion_output_dir", type=str, default=None, help="(from_saved) ar_pred_*.pt 所在目录")

    parser.add_argument("--sliding_window", action="store_true", default=False,
                        help="滑动窗口模式：每个台风生成多个起报点进行评估（更充分的统计）")
    parser.add_argument("--window_stride", type=int, default=None,
                        help="滑动窗口步长（时间步数），默认 = t_future（不重叠窗口）")
    parser.add_argument("--split_by_year", action="store_true", default=False,
                        help="按年份划分测试集 (test=2019-2021)，与训练时 --split_by_year 一致")

def validate_args(args, needs_table3_strategy: bool = False) -> None:
    if args.mode == "end_to_end":
        missing = [name for name in ("diffusion_code", "diffusion_ckpt", "data_root") if not getattr(args, name)]
        if missing:
            raise ValueError(f"end_to_end 模式缺少参数: {', '.join('--' + item for item in missing)}")
    elif args.mode == "from_saved" and not args.diffusion_output_dir:
        raise ValueError("from_saved 模式必须提供 --diffusion_output_dir")

    if args.num_samples < 1:
        raise ValueError("--num_samples 必须 >= 1")
    if args.report_every_hours < traj_data_cfg.time_resolution_hours:
        raise ValueError("--report_every_hours 不能小于模型输出时间分辨率")
    if needs_table3_strategy and args.selection_strategy not in ("per_lead_min", "best_72h_traj"):
        raise ValueError("Table 3 的 selection_strategy 只能是 per_lead_min 或 best_72h_traj")

def _extract_year(storm_id: str) -> Optional[int]:
    try:
        return int(storm_id[:4])
    except (ValueError, IndexError):
        return None

def select_typhoon_ids(
    all_ids: Sequence[str],
    target_ids: Optional[Iterable[str]],
    min_year: Optional[int] = None,
) -> List[str]:
    if not target_ids:
        result = list(all_ids)
    else:
        target_set = set(target_ids)
        result = [storm_id for storm_id in all_ids if storm_id in target_set]

    if min_year is not None:
        before = len(result)
        result = [sid for sid in result if (_extract_year(sid) or 0) >= min_year]
        print(f"[年份过滤] min_year={min_year}: {before} -> {len(result)} 个台风 (排除 {before - len(result)} 个)")

    return result
