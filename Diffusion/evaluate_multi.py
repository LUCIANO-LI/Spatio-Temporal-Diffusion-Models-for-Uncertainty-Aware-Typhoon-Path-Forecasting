import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIFFUSION_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREPROCESS_DIR = os.environ.get("TYPHOON_PREPROCESS_DIR")

sys.path.insert(0, DIFFUSION_DIR)
from configs import get_config
from data.dataset import ERA5TyphoonDataset, split_typhoon_ids, split_typhoon_ids_by_year
from models import ERA5DiffusionModel
from train import EMA
from inference import ERA5Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VAR_NAMES = [
    'u_850', 'u_500', 'u_250',
    'v_850', 'v_500', 'v_250',
    'z_850', 'z_500', 'z_250',
]

def denormalize_field(data_norm, mean, std):
    std = np.where(std < 1e-8, 1.0, std)
    return data_norm * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)

def main():
    parser = argparse.ArgumentParser(description="多台风自回归评估")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--work_dir", type=str, default=DIFFUSION_DIR)
    parser.add_argument("--num_typhoons", type=int, default=20,
                        help="测试台风数量")
    parser.add_argument("--num_ar_steps", type=int, default=24,
                        help="自回归步数 (24步=72h)")
    parser.add_argument("--save_dir", type=str, default="eval_results")
    parser.add_argument("--preprocess_dir", type=str, default=DEFAULT_PREPROCESS_DIR)
    parser.add_argument("--ar_ensemble", type=int, default=None,
                        help="逐步集合数 (默认用config值)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="并行推理台风数 (默认8, 根据显存调整)")
    parser.add_argument("--exclude_file", type=str, default=None,
                        help="排除台风列表文件 (每行一个台风ID, 如 excluded_typhoons.txt)")
    parser.add_argument("--split_by_year", action="store_true",
                        help="按年份划分 (train=1950-2016, val=2017-2018, test=2019-2021)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    data_cfg, model_cfg, _, infer_cfg = get_config(data_root=args.data_root)
    if args.ar_ensemble is not None:
        infer_cfg.ar_ensemble_per_step = args.ar_ensemble

    norm_path = os.path.join(args.work_dir, "norm_stats.pt")
    stats = torch.load(norm_path, weights_only=True)
    norm_mean = stats["mean"].numpy()
    norm_std = stats["std"].numpy()
    C = data_cfg.num_channels

    logger.info(f"加载模型: {args.checkpoint}")
    model = ERA5DiffusionModel(model_cfg, data_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema_state_dict" in ckpt:
        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply_shadow(model)
        logger.info("已加载 EMA 参数")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    predictor = ERA5Predictor(model, data_cfg, infer_cfg, norm_mean, norm_std, device)

    if args.split_by_year:
        _, _, test_ids = split_typhoon_ids_by_year(
            data_cfg.data_root,
            train_years=(1950, 2016),
            val_years=(2017, 2018),
            test_years=(2019, 2021),
        )
        logger.info("使用按年份划分: test=2019-2021")
    else:
        _, _, test_ids = split_typhoon_ids(data_cfg.data_root, seed=42)
    n_typhoons = min(args.num_typhoons, len(test_ids))
    logger.info(f"测试台风: {n_typhoons}/{len(test_ids)}")

    test_dataset = ERA5TyphoonDataset(
        typhoon_ids=test_ids[:n_typhoons],
        data_root=data_cfg.data_root,
        pl_vars=data_cfg.pressure_level_vars,
        sfc_vars=data_cfg.surface_vars,
        pressure_levels=data_cfg.pressure_levels,
        history_steps=data_cfg.history_steps,
        forecast_steps=data_cfg.forecast_steps,
        norm_mean=norm_mean[:C],
        norm_std=norm_std[:C],
        preprocessed_dir=args.preprocess_dir,
    )
    logger.info(f"数据集样本数: {len(test_dataset)}")

    tid_to_indices = {}
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        tid = sample["typhoon_id"]
        if tid not in tid_to_indices:
            tid_to_indices[tid] = []
        tid_to_indices[tid].append(i)

    logger.info(f"实际覆盖台风: {len(tid_to_indices)}")
    for tid, indices in tid_to_indices.items():
        logger.info(f"  {tid}: {len(indices)} 个样本")

    if args.exclude_file and os.path.exists(args.exclude_file):
        with open(args.exclude_file, 'r') as f:
            exclude_ids = set(line.strip() for line in f if line.strip())
        before = len(tid_to_indices)
        tid_to_indices = {tid: idx for tid, idx in tid_to_indices.items() if tid not in exclude_ids}
        excluded = before - len(tid_to_indices)
        logger.info(f"已排除 {excluded} 个台风 (来自 {args.exclude_file}), 剩余 {len(tid_to_indices)}")

    ar_steps = args.num_ar_steps
    all_rmse = []
    all_tids = []
    batch_size = args.batch_size

    typhoon_jobs = []
    for tid, indices in tid_to_indices.items():
        start_idx = indices[len(indices) // 2]
        typhoon_jobs.append((tid, start_idx, indices))

    logger.info(f"批量推理: {len(typhoon_jobs)} 个台风, batch_size={batch_size}, "
                f"集合={infer_cfg.ar_ensemble_per_step}")

    for batch_start in range(0, len(typhoon_jobs), batch_size):
        batch_jobs = typhoon_jobs[batch_start:batch_start + batch_size]
        B_actual = len(batch_jobs)
        batch_tids = [job[0] for job in batch_jobs]
        batch_start_indices = [job[1] for job in batch_jobs]

        logger.info(f"批次 {batch_start // batch_size + 1}/"
                    f"{(len(typhoon_jobs) + batch_size - 1) // batch_size}: "
                    f"{B_actual} 个台风 {batch_tids}")

        batch_conds = []
        for tid, start_idx, indices in batch_jobs:
            sample = test_dataset[start_idx]
            batch_conds.append(sample["condition"])
        batch_cond = torch.stack(batch_conds, dim=0).to(device)

        with torch.no_grad():
            preds = predictor.predict_autoregressive(
                batch_cond, num_steps=ar_steps,
                noise_sigma=infer_cfg.autoregressive_noise_sigma,
                ensemble_per_step=infer_cfg.ar_ensemble_per_step,
            )

        for b_idx, (tid, start_idx, indices) in enumerate(batch_jobs):
            rmse = np.full((ar_steps, C), np.nan)
            n_valid = 0

            for t in range(ar_steps):
                gt_idx = start_idx + t
                if gt_idx >= len(test_dataset):
                    break
                gt_sample = test_dataset[gt_idx]
                if gt_sample["typhoon_id"] != tid:
                    break

                pred_phys = denormalize_field(
                    preds[t][b_idx].cpu().numpy()[:C], norm_mean[:C], norm_std[:C]
                )
                gt_phys = denormalize_field(
                    gt_sample["target"][:C].numpy(), norm_mean[:C], norm_std[:C]
                )

                for v in range(C):
                    rmse[t, v] = np.sqrt(np.mean((pred_phys[v] - gt_phys[v]) ** 2))
                n_valid += 1

            if n_valid > 0:
                all_rmse.append(rmse)
                all_tids.append(tid)
                logger.info(f"  {tid}: 有效步={n_valid}/{ar_steps}, "
                            f"3h u850={rmse[0,0]:.2f}, "
                            f"{min(n_valid,24)*3}h u850={rmse[min(23,n_valid-1),0]:.2f}")

    if not all_rmse:
        print("没有有效结果!")
        return

    stacked = np.stack(all_rmse, axis=0)
    mean_rmse = np.nanmean(stacked, axis=0)
    median_rmse = np.nanmedian(stacked, axis=0)
    count = np.sum(~np.isnan(stacked[:, :, 0]), axis=0)

    cache_path = os.path.join(args.save_dir, "per_typhoon_rmse.npz")
    np.savez(cache_path, stacked=stacked, tids=np.array(all_tids, dtype=object))
    logger.info(f"逐台风 RMSE 缓存已保存: {cache_path}")

    print(f"\n{'='*100}")
    print(f"  [Mean RMSE] {len(all_rmse)} typhoons, {ar_steps} steps = {ar_steps*3}h")
    print(f"{'='*100}")
    print(f"{'时效':<8}", end="")
    for vn in VAR_NAMES[:C]:
        print(f" {vn:>8}", end="")
    print()
    print("-" * (8 + 9 * C))

    for t in range(ar_steps):
        if count[t] == 0:
            continue
        print(f"+{(t+1)*3:>3}h    ", end="")
        for v in range(C):
            print(f" {mean_rmse[t, v]:>8.2f}", end="")
        print()

    print(f"\n{'平均':>7} ", end="")
    for v in range(C):
        print(f" {np.nanmean(mean_rmse[:, v]):>8.2f}", end="")
    print()

    print(f"\n{'='*100}")
    print(f"  [Median RMSE] {len(all_rmse)} typhoons (robust to outliers)")
    print(f"{'='*100}")
    print(f"{'时效':<8}", end="")
    for vn in VAR_NAMES[:C]:
        print(f" {vn:>8}", end="")
    print()
    print("-" * (8 + 9 * C))

    for t in range(ar_steps):
        if count[t] == 0:
            continue
        print(f"+{(t+1)*3:>3}h    ", end="")
        for v in range(C):
            print(f" {median_rmse[t, v]:>8.2f}", end="")
        print()

    print(f"\n{'平均':>7} ", end="")
    for v in range(C):
        print(f" {np.nanmean(median_rmse[:, v]):>8.2f}", end="")
    print()

    print(f"\n提示: 逐台风 RMSE 缓存已保存, 可运行离线诊断:")
    print(f"  python diagnose_z_explosion.py --cache {cache_path}")

    csv_path = os.path.join(args.save_dir, "rmse_mean.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("lead_time_h," + ",".join(VAR_NAMES[:C]) + "\n")
        for t in range(ar_steps):
            if count[t] == 0:
                continue
            f.write(f"{(t+1)*3}")
            for v in range(C):
                f.write(f",{mean_rmse[t, v]:.4f}")
            f.write("\n")
    print(f"\nMean CSV: {csv_path}")

    csv_path_med = os.path.join(args.save_dir, "rmse_median.csv")
    with open(csv_path_med, 'w', encoding='utf-8') as f:
        f.write("lead_time_h," + ",".join(VAR_NAMES[:C]) + "\n")
        for t in range(ar_steps):
            if count[t] == 0:
                continue
            f.write(f"{(t+1)*3}")
            for v in range(C):
                f.write(f",{median_rmse[t, v]:.4f}")
            f.write("\n")
    print(f"Median CSV: {csv_path_med}")

    hours = np.array([(t+1)*3 for t in range(ar_steps)])
    valid = count > 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'RMSE vs Lead Time ({len(all_rmse)} typhoons)', fontsize=14, fontweight='bold')

    std_rmse = np.nanstd(stacked, axis=0)

    for v in range(min(6, C)):
        axes[0,0].plot(hours[valid], mean_rmse[valid, v], marker='o', markersize=3,
                       label=VAR_NAMES[v], linewidth=1.5)
    axes[0,0].set_ylabel('RMSE (m/s)')
    axes[0,0].set_title('Mean — Wind (u, v)')
    axes[0,0].legend(fontsize=7, ncol=2)
    axes[0,0].grid(True, alpha=0.3)

    for v in range(6, C):
        axes[0,1].plot(hours[valid], mean_rmse[valid, v], marker='s', markersize=3,
                       label=VAR_NAMES[v], linewidth=1.5)
    axes[0,1].set_ylabel(u'RMSE (m\u00b2/s\u00b2)')
    axes[0,1].set_title('Mean — Geopotential (z)')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    for v in range(min(6, C)):
        axes[1,0].plot(hours[valid], median_rmse[valid, v], marker='o', markersize=3,
                       label=VAR_NAMES[v], linewidth=1.5)
    axes[1,0].set_xlabel('Lead time (hours)')
    axes[1,0].set_ylabel('RMSE (m/s)')
    axes[1,0].set_title('Median — Wind (u, v)')
    axes[1,0].legend(fontsize=7, ncol=2)
    axes[1,0].grid(True, alpha=0.3)

    for v in range(6, C):
        axes[1,1].plot(hours[valid], median_rmse[valid, v], marker='s', markersize=3,
                       label=VAR_NAMES[v], linewidth=1.5)
    axes[1,1].set_xlabel('Lead time (hours)')
    axes[1,1].set_ylabel(u'RMSE (m\u00b2/s\u00b2)')
    axes[1,1].set_title('Median — Geopotential (z)')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_p = os.path.join(args.save_dir, "rmse_vs_leadtime.png")
    plt.savefig(save_p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_p}")

    fig, ax = plt.subplots(figsize=(12, 5))
    norm_rmse = mean_rmse.copy()
    for v in range(C):
        norm_rmse[:, v] /= (norm_std[v] + 1e-8)

    im = ax.imshow(norm_rmse[valid].T, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=min(2.0, np.nanmax(norm_rmse[valid])))
    ax.set_yticks(range(C))
    ax.set_yticklabels(VAR_NAMES[:C])
    step = max(1, int(valid.sum()) // 8)
    xticks = list(range(0, int(valid.sum()), step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"+{hours[valid][i]}h" for i in xticks])
    ax.set_xlabel('Lead time')
    ax.set_title(f'Normalized RMSE (RMSE/std) — {len(all_rmse)} typhoons', fontsize=13)
    plt.colorbar(im, ax=ax, label='RMSE / std')

    for row in range(C):
        for ci, col in enumerate(np.where(valid)[0]):
            val = norm_rmse[col, row]
            if not np.isnan(val):
                color = 'white' if val > 1.0 else 'black'
                ax.text(ci, row, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

    plt.tight_layout()
    save_p = os.path.join(args.save_dir, "rmse_heatmap.png")
    plt.savefig(save_p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_p}")

    print(f"\n评估完成! 结果保存在: {args.save_dir}/")

if __name__ == "__main__":
    main()
