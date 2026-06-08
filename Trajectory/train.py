import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from config import train_cfg, model_cfg, data_cfg
from model import LT3PModel, create_lt3p_model
from dataset import LT3PDataset, create_dataloaders, denormalize_coords, filter_out_of_range_storms
from data_processing import load_all_storms, load_tyc_storms

class LT3PTrainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = None,
        learning_rate: float = None,
        num_epochs: int = None,
        checkpoint_dir: str = None
    ):
        self.device = device or train_cfg.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs or train_cfg.num_epochs
        self.checkpoint_dir = Path(checkpoint_dir or train_cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        learning_rate = learning_rate or train_cfg.learning_rate
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.999)
        )

        self.warmup_epochs = getattr(train_cfg, 'warmup_epochs', 10)
        self.scheduler = self._create_scheduler(learning_rate)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'tb_logs'))
        self.global_step = 0

        self.early_stopping = train_cfg.early_stopping
        self.patience = train_cfg.patience
        self.patience_counter = 0

        self.use_amp = train_cfg.use_amp and self.device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.use_ema = True
        self.ema_decay = 0.9999
        self.ema_model = None
        if self.use_ema:
            self._init_ema()
    
    def _create_scheduler(self, learning_rate: float):
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _init_ema(self):
        import copy
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def _update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_cont = 0.0
        total_dir = 0.0
        total_curv = 0.0
        total_speed = 0.0
        total_smooth = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            history_coords = batch['history_coords'].to(self.device, non_blocking=True)
            past_era5 = batch['past_era5'].to(self.device, non_blocking=True)
            future_era5 = batch['future_era5'].to(self.device, non_blocking=True)
            target_coords = batch['target_coords'].to(self.device, non_blocking=True)
            sample_weight = batch['sample_weight'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(
                    history_coords, future_era5, target_coords, past_era5=past_era5
                )
                loss = outputs['loss']
                if train_cfg.use_sample_weights:
                    loss = (loss * sample_weight.mean())

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if self.use_ema:
                self._update_ema()

            total_loss += loss.item()
            total_mse += outputs['mse_loss'].item()
            if 'continuity_loss' in outputs:
                total_cont += outputs['continuity_loss'].item()
            if 'direction_loss' in outputs:
                total_dir += outputs['direction_loss'].item()
            if 'curvature_loss' in outputs:
                total_curv += outputs['curvature_loss'].item()
            if 'speed_penalty' in outputs:
                total_speed += outputs['speed_penalty'].item()
            if 'smooth_loss' in outputs:
                total_smooth += outputs['smooth_loss'].item()
            num_batches += 1

            if self.global_step % train_cfg.log_interval == 0:
                self.writer.add_scalar('step/loss', loss.item(), self.global_step)
                self.writer.add_scalar('step/mse_loss', outputs['mse_loss'].item(), self.global_step)
                if 'continuity_loss' in outputs:
                    self.writer.add_scalar('step/continuity_loss', outputs['continuity_loss'].item(), self.global_step)
                if 'direction_loss' in outputs:
                    self.writer.add_scalar('step/direction_loss', outputs['direction_loss'].item(), self.global_step)
                if 'curvature_loss' in outputs:
                    self.writer.add_scalar('step/curvature_loss', outputs['curvature_loss'].item(), self.global_step)
                if 'speed_penalty' in outputs:
                    self.writer.add_scalar('step/speed_penalty', outputs['speed_penalty'].item(), self.global_step)
                if 'oscillation_loss' in outputs:
                    self.writer.add_scalar('step/oscillation_loss', outputs['oscillation_loss'].item(), self.global_step)
            self.global_step += 1

            postfix = {
                'loss': f"{loss.item():.4f}",
                'mse': f"{outputs['mse_loss'].item():.4f}"
            }
            if 'continuity_loss' in outputs:
                postfix['cont'] = f"{outputs['continuity_loss'].item():.4f}"
            if 'direction_loss' in outputs:
                postfix['dir'] = f"{outputs['direction_loss'].item():.4f}"
            if 'curvature_loss' in outputs:
                postfix['curv'] = f"{outputs['curvature_loss'].item():.4f}"
            if 'smooth_loss' in outputs:
                postfix['smooth'] = f"{outputs['smooth_loss'].item():.6f}"
            pbar.set_postfix(postfix)

        avg_loss = total_loss / num_batches

        self.writer.add_scalar('train/loss', avg_loss, epoch)
        self.writer.add_scalar('train/mse_loss', total_mse / num_batches, epoch)
        self.writer.add_scalar('train/continuity_loss', total_cont / num_batches, epoch)
        self.writer.add_scalar('train/direction_loss', total_dir / num_batches, epoch)
        self.writer.add_scalar('train/curvature_loss', total_curv / num_batches, epoch)
        self.writer.add_scalar('train/speed_penalty', total_speed / num_batches, epoch)
        self.writer.add_scalar('train/smooth_loss', total_smooth / num_batches, epoch)

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> float:
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_cont = 0.0
        total_dir = 0.0
        total_curv = 0.0
        total_speed = 0.0
        num_batches = 0

        for batch in self.val_loader:
            history_coords = batch['history_coords'].to(self.device)
            past_era5 = batch['past_era5'].to(self.device)
            future_era5 = batch['future_era5'].to(self.device)
            target_coords = batch['target_coords'].to(self.device)

            outputs = self.model(history_coords, future_era5, target_coords, past_era5=past_era5)
            total_loss += outputs['loss'].item()
            total_mse += outputs['mse_loss'].item()
            if 'continuity_loss' in outputs:
                total_cont += outputs['continuity_loss'].item()
            if 'direction_loss' in outputs:
                total_dir += outputs['direction_loss'].item()
            if 'curvature_loss' in outputs:
                total_curv += outputs['curvature_loss'].item()
            if 'speed_penalty' in outputs:
                total_speed += outputs['speed_penalty'].item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if num_batches > 0:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            self.writer.add_scalar('val/mse_loss', total_mse / num_batches, epoch)
            self.writer.add_scalar('val/continuity_loss', total_cont / num_batches, epoch)
            self.writer.add_scalar('val/direction_loss', total_dir / num_batches, epoch)
            self.writer.add_scalar('val/curvature_loss', total_curv / num_batches, epoch)
            self.writer.add_scalar('val/speed_penalty', total_speed / num_batches, epoch)

        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not is_best:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        if self.use_ema and self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
        print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint.get('epoch', 0)
    
    def train(self, resume_from: str = None):
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed from epoch {start_epoch}")

        print(f"Training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"TensorBoard logs: {self.checkpoint_dir / 'tb_logs'}")

        for epoch in range(start_epoch, self.num_epochs):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)

            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            self.writer.add_scalars('compare/loss', {
                'train': train_loss,
                'val': val_loss,
            }, epoch)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                  f"LR = {current_lr:.6f}")

            compare_loss = val_loss if val_loss > 0 else train_loss
            is_best = compare_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = compare_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, is_best)

            self.writer.add_scalar('val/best_loss', self.best_val_loss, epoch)

            if self.early_stopping and self.patience_counter >= self.patience:
                print(f"Early stopping triggered! No improvement for {self.patience} epochs.")
                break

        self.save_loss_plot()
        self.writer.close()
        print("Training complete!")
        return self.train_losses, self.val_losses

    def save_config(self, config_dict: dict):
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"Config saved to {config_path}")

    def save_loss_plot(self):
        if not self.train_losses or not self.val_losses:
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)

        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)

        best_epoch = np.argmin(self.val_losses) + 1
        best_val = min(self.val_losses)
        plt.scatter([best_epoch], [best_val], c='green', s=100, zorder=5, 
                   label=f'Best (epoch {best_epoch})')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('LT3P Training: 48h History + 72h ERA5 → 72h Trajectory', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plot_path = self.checkpoint_dir / 'loss_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to {plot_path}")

def evaluate_on_test(model, test_loader, device):
    model.eval()

    if len(test_loader.dataset) == 0:
        print("  No test samples available!")
        return {}

    all_errors_km = []
    all_storm_ids = []
    lat_range = data_cfg.lat_range
    lon_range = data_cfg.lon_range

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            history_coords = batch['history_coords'].to(device)
            past_era5 = batch.get('past_era5')
            if past_era5 is not None:
                past_era5 = past_era5.to(device)
            future_era5 = batch['future_era5'].to(device)
            target_lat = batch['target_lat_raw'].numpy()
            target_lon = batch['target_lon_raw'].numpy()
            storm_ids = batch['storm_id']

            outputs = model.predict(history_coords, future_era5, past_era5=past_era5)
            pred_coords = outputs['predicted_coords'].cpu().numpy()
            
            pred_lat = pred_coords[:, :, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
            pred_lon = pred_coords[:, :, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]

            lat_err = (pred_lat - target_lat) * 111
            lon_err = (pred_lon - target_lon) * 111 * np.cos(np.radians(target_lat))
            dist_err = np.sqrt(lat_err**2 + lon_err**2)
            all_errors_km.append(dist_err)
            all_storm_ids.extend(storm_ids)

    all_errors = np.concatenate(all_errors_km, axis=0)
    
    sample_mean_errors = all_errors.mean(axis=1)
    outlier_threshold = np.percentile(sample_mean_errors, 95)
    outlier_mask = sample_mean_errors > outlier_threshold
    n_outliers = outlier_mask.sum()
    
    print(f"\n{'='*60}")
    print(f"Test Results: {all_errors.shape[0]} samples, {all_errors.shape[1]} timesteps")
    print(f"{'='*60}")
    print(f"\n[Outlier Analysis]")
    print(f"  Outlier threshold (95th percentile): {outlier_threshold:.2f} km")
    print(f"  Number of outliers: {n_outliers} ({100*n_outliers/len(sample_mean_errors):.1f}%)")
    if n_outliers > 0:
        print(f"  Max error sample: {sample_mean_errors.max():.2f} km")
    
    filtered_errors = all_errors[~outlier_mask]
    
    print(f"\n[Full Results (all samples)]")
    print(f"Overall Mean Error: {all_errors.mean():.2f} km")
    
    print(f"\n[Filtered Results (excluding top 5% outliers)]")
    print(f"Overall Mean Error: {filtered_errors.mean():.2f} km")
    
    print(f"\nError by forecast lead time (filtered):")

    results = {
        "num_samples": int(all_errors.shape[0]),
        "num_outliers": int(n_outliers),
        "mean_error_km": float(all_errors.mean()),
        "mean_error_km_filtered": float(filtered_errors.mean()),
        "std_error_km": float(all_errors.std()),
        "error_by_hour": {},
        "error_by_hour_filtered": {}
    }

    for t in range(all_errors.shape[1]):
        hours = (t + 1) * 3
        mean_err = all_errors[:, t].mean()
        std_err = all_errors[:, t].std()
        mean_err_filtered = filtered_errors[:, t].mean()
        std_err_filtered = filtered_errors[:, t].std()
        print(f"  +{hours:2d}h: {mean_err_filtered:6.2f} ± {std_err_filtered:5.2f} km (full: {mean_err:.0f}±{std_err:.0f})")
        results["error_by_hour"][f"{hours}h"] = {
            "mean_km": float(mean_err),
            "std_km": float(std_err)
        }
        results["error_by_hour_filtered"][f"{hours}h"] = {
            "mean_km": float(mean_err_filtered),
            "std_km": float(std_err_filtered)
        }

    print(f"\nKey benchmarks:")
    for key_hour in [24, 48, 72]:
        t_idx = key_hour // 3 - 1
        if t_idx < all_errors.shape[1]:
            print(f"  +{key_hour}h: {all_errors[:, t_idx].mean():.2f} km")

    return results

def main():
    parser = argparse.ArgumentParser(description="LT3P Typhoon Trajectory Training")
    parser.add_argument("--split_by_year", action="store_true", default=False,
                        help="按年份划分 (train=1950-2016, val=2017-2018, test=2019-2021)")
    args = parser.parse_args()

    print("=" * 60)
    print("LT3P-Style Typhoon Trajectory Prediction - Training")
    print("48h History + 72h ERA5 -> 72h Trajectory @ 3h resolution")
    if args.split_by_year:
        print("  Split: by year (1950-2016 / 2017-2018 / 2019-2021)")
    else:
        print("  Split: by storm ID (random 70/15/15)")
    print("=" * 60)

    print("\n[1] Loading data...")
    try:
        storm_samples = load_tyc_storms(
            csv_path=data_cfg.csv_path,
            era5_base_dir=data_cfg.era5_dir
        )
    except Exception as e:
        print(f"Error loading TYC data: {e}")
        print("Trying standard loader...")
        try:
            storm_samples = load_all_storms()
        except FileNotFoundError as e2:
            print(f"Data files not found: {e2}")
            print("Creating dummy data for testing...")
            storm_samples = create_dummy_data()

    if len(storm_samples) == 0:
        print("No data available. Please check your data paths in config.py")
        return

    storm_samples = filter_out_of_range_storms(storm_samples)

    print("\n[2] Creating dataloaders...")
    if args.split_by_year:
        train_years = list(range(1950, 2017))
        val_years = list(range(2017, 2019))
        test_years = list(range(2019, 2022))
        train_loader, val_loader, test_loader = create_dataloaders(
            storm_samples, split_by='year',
            train_years=train_years, val_years=val_years, test_years=test_years,
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(storm_samples)
    
    if len(train_loader) == 0:
        print("No training samples! Check data or reduce min_typhoon_duration_hours in config.")
        return

    print("\n[3] Creating LT3P model...")
    
    sample_batch = next(iter(train_loader))
    era5_channels = sample_batch['future_era5'].shape[2]
    print(f"ERA5 channels: {era5_channels}")
    print(f"History coords shape: {sample_batch['history_coords'].shape}")
    print(f"Future ERA5 shape: {sample_batch['future_era5'].shape}")
    print(f"Target coords shape: {sample_batch['target_coords'].shape}")

    model = LT3PModel(
        coord_dim=model_cfg.coord_dim,
        output_dim=model_cfg.output_dim,
        era5_channels=era5_channels,
        t_history=model_cfg.t_history,
        t_future=model_cfg.t_future,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    print("\n[4] Starting training...")
    trainer = LT3PTrainer(model, train_loader, val_loader)

    config = {
        "model": {
            "type": "LT3P",
            "coord_dim": model_cfg.coord_dim,
            "output_dim": model_cfg.output_dim,
            "era5_channels": era5_channels,
            "t_history": model_cfg.t_history,
            "t_future": model_cfg.t_future,
            "transformer_dim": model_cfg.transformer_dim,
            "transformer_heads": model_cfg.transformer_heads,
            "transformer_layers": model_cfg.transformer_layers,
            "num_params": num_params
        },
        "training": {
            "batch_size": train_cfg.batch_size,
            "learning_rate": train_cfg.learning_rate,
            "num_epochs": train_cfg.num_epochs,
            "device": train_cfg.device
        },
        "data": {
            "time_resolution_hours": data_cfg.time_resolution_hours,
            "history_hours": model_cfg.t_history * data_cfg.time_resolution_hours,
            "future_hours": model_cfg.t_future * data_cfg.time_resolution_hours,
            "num_storms": len(storm_samples),
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset)
        }
    }
    trainer.save_config(config)

    trainer.train()

    print("\n[5] Evaluating on test set...")
    eval_model = trainer.ema_model if (trainer.use_ema and trainer.ema_model is not None) else model
    test_results = evaluate_on_test(eval_model, test_loader, trainer.device)

    config["test_results"] = test_results
    trainer.save_config(config)

    print("\nDone!")

def create_dummy_data():
    from data_structures import StormSample
    
    dummy_samples = []
    
    for i in range(10):
        T = 60
        storm_id = f"STORM_{i:04d}"
        
        times = np.array([
            np.datetime64('2020-01-01') + np.timedelta64(j * 3, 'h') 
            for j in range(T)
        ])
        
        track_lat = 15 + i + np.linspace(0, 20, T) + np.random.randn(T) * 0.3
        track_lon = 120 + np.linspace(0, 30, T) + np.random.randn(T) * 0.3
        track_vmax = 30 + 20 * np.sin(np.linspace(0, 2*np.pi, T))
        
        era5_array = np.random.randn(T, 6, data_cfg.grid_height, data_cfg.grid_width).astype(np.float32) * 0.1
        
        lat_grid = np.linspace(5, 45, data_cfg.grid_height)
        lon_grid = np.linspace(100, 180, data_cfg.grid_width)
        
        sample = StormSample(
            storm_id=storm_id,
            times=times,
            track_lat=track_lat.astype(np.float32),
            track_lon=track_lon.astype(np.float32),
            track_vmax=track_vmax.astype(np.float32),
            era5_array=era5_array,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            is_real=np.ones(T, dtype=bool),
            year=2020 + i % 3
        )
        dummy_samples.append(sample)
    
    print(f"Created {len(dummy_samples)} dummy storm samples")
    return dummy_samples

if __name__ == "__main__":
    main()
