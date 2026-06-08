import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VAR_NAMES = [
    'u_850', 'u_500', 'u_250',
    'v_850', 'v_500', 'v_250',
    'z_850', 'z_500', 'z_250',
]

VAR_UNITS = {
    'u_850': 'm/s', 'u_500': 'm/s', 'u_250': 'm/s',
    'v_850': 'm/s', 'v_500': 'm/s', 'v_250': 'm/s',
    'z_850': 'm²/s²', 'z_500': 'm²/s²', 'z_250': 'm²/s²',
}

def plot_single_sample(pred, gt, typhoon_id, save_path):
    n_vars = pred.shape[0]
    fig, axes = plt.subplots(n_vars, 2, figsize=(10, n_vars * 3.2))
    fig.suptitle(f'Typhoon: {typhoon_id}', fontsize=16, fontweight='bold', y=1.01)

    for i in range(n_vars):
        gt_i = gt[i]
        pred_i = pred[i]
        rmse = np.sqrt(np.mean((pred_i - gt_i) ** 2))

        vmin = min(gt_i.min(), pred_i.min())
        vmax = max(gt_i.max(), pred_i.max())

        ax0 = axes[i, 0]
        im0 = ax0.imshow(gt_i, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax0.set_title(f'{VAR_NAMES[i]} - GT', fontsize=10)
        ax0.set_xticks([])
        ax0.set_yticks([])
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = axes[i, 1]
        im1 = ax1.imshow(pred_i, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax1.set_title(f'{VAR_NAMES[i]} - Pred (RMSE={rmse:.2f} {VAR_UNITS.get(VAR_NAMES[i], "")})', fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_summary(pred, gt, typhoon_id, save_path):
    key_indices = [0, 3, 6, 7, 8]
    key_names = [VAR_NAMES[i] for i in key_indices]

    fig, axes = plt.subplots(len(key_indices), 2, figsize=(10, len(key_indices) * 3.5))
    fig.suptitle(f'Typhoon {typhoon_id} - Key Variables', fontsize=14, fontweight='bold')

    for row, idx in enumerate(key_indices):
        gt_i = gt[idx]
        pred_i = pred[idx]
        rmse = np.sqrt(np.mean((pred_i - gt_i) ** 2))

        vmin = min(gt_i.min(), pred_i.min())
        vmax = max(gt_i.max(), pred_i.max())

        ax0 = axes[row, 0]
        im0 = ax0.imshow(gt_i, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax0.set_title(f'{key_names[row]} - Ground Truth')
        ax0.set_xticks([]); ax0.set_yticks([])
        plt.colorbar(im0, ax=ax0, fraction=0.046)

        ax1 = axes[row, 1]
        im1 = ax1.imshow(pred_i, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax1.set_title(f'{key_names[row]} - Pred (RMSE={rmse:.2f})')
        ax1.set_xticks([]); ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="ERA5 prediction visualization")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sample_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="visualizations")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    pt_path = os.path.join(args.output_dir, f"single_pred_{args.sample_id}.pt")
    if not os.path.exists(pt_path):
        print(f"File not found: {pt_path}")
        sys.exit(1)

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    pred = data["prediction"][0].numpy()
    gt = data["ground_truth"][0].numpy()
    tid = data["typhoon_id"]

    print(f"Typhoon: {tid}")
    print(f"Pred shape: {pred.shape}, GT shape: {gt.shape}")

    plot_single_sample(
        pred, gt, tid,
        os.path.join(args.save_dir, f"full_{args.sample_id}.png"),
    )

    plot_summary(
        pred, gt, tid,
        os.path.join(args.save_dir, f"summary_{args.sample_id}.png"),
    )

    print(f"\n{'Variable':<12} {'RMSE':>10} {'GT std':>10} {'Ratio':>8}")
    print("=" * 44)
    for i in range(pred.shape[0]):
        rmse = np.sqrt(np.mean((pred[i] - gt[i]) ** 2))
        std = gt[i].std()
        ratio = rmse / (std + 1e-8)
        marker = " <--" if ratio < 1.5 else ""
        print(f"{VAR_NAMES[i]:<12} {rmse:>10.2f} {std:>10.2f} {ratio:>7.2f}x{marker}")

if __name__ == "__main__":
    main()
