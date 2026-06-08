import os
import sys
import copy
import time
import math
import logging
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import DataConfig, ModelConfig, TrainConfig, get_config
from data import build_dataloaders
from data.dataset import preprocess_to_npy
from models import ERA5DiffusionModel

DIFFUSION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DIFFUSION_DIR, ".."))
DEFAULT_DATA_ROOT = os.environ.get(
    "TYPHOON_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "Typhoon_data_final"),
)
DEFAULT_PREPROCESS_DIR = os.environ.get("TYPHOON_PREPROCESS_DIR")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]

class DataPrefetcher:

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self._next_batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                for key in self._next_batch:
                    if isinstance(self._next_batch[key], torch.Tensor):
                        self._next_batch[key] = self._next_batch[key].to(
                            self.device, non_blocking=True
                        )

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)

        batch = self._next_batch
        if batch is None:
            raise StopIteration

        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)

class Trainer:
    def __init__(
        self,
        model: ERA5DiffusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainConfig,
        data_cfg: DataConfig,
        work_dir: str = ".",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.data_cfg = data_cfg
        self.work_dir = work_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"训练设备: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model = self.model.to(self.device)

        if train_cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        if train_cfg.use_compile and sys.platform != "win32":
            try:
                logger.info("正在编译模型 (torch.compile)...")
                self.model.dit = torch.compile(self.model.dit)
                logger.info("模型编译完成")
            except Exception as e:
                logger.warning(f"torch.compile 失败，使用 eager 模式: {e}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=train_cfg.betas,
        )

        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=train_cfg.warmup_start_lr / train_cfg.learning_rate,
            end_factor=1.0,
            total_iters=train_cfg.warmup_steps,
        )
        steps_per_epoch = max(len(train_loader) // train_cfg.gradient_accumulation_steps, 1)
        total_optim_steps = steps_per_epoch * train_cfg.max_epochs
        cosine_steps = max(total_optim_steps - train_cfg.warmup_steps, 1)

        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=train_cfg.min_lr,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[train_cfg.warmup_steps],
        )

        self.amp_dtype = (
            torch.bfloat16 if train_cfg.amp_dtype == "bfloat16" else torch.float16
        )
        self.use_amp = train_cfg.use_amp and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )

        self.ema = EMA(self.model, decay=train_cfg.ema_decay)

        self.writer = None
        if train_cfg.use_tensorboard:
            log_dir = os.path.join(work_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        self.ckpt_dir = os.path.join(work_dir, train_cfg.checkpoint_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.global_step = 0
        self.optim_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        if train_cfg.resume_from:
            self._load_checkpoint(train_cfg.resume_from)

    def train(self):
        logger.info(f"开始训练: max_epochs={self.cfg.max_epochs}")
        logger.info(
            f"batch_size={self.cfg.batch_size} × "
            f"grad_accum={self.cfg.gradient_accumulation_steps} = "
            f"effective_batch={self.cfg.batch_size * self.cfg.gradient_accumulation_steps}"
        )
        if self.cfg.condition_noise_sigma > 0:
            logger.info(
                f"条件噪声增强: sigma={self.cfg.condition_noise_sigma}, "
                f"prob={self.cfg.condition_noise_prob}, "
                f"rampup={self.cfg.condition_noise_rampup_epochs} epochs"
            )
        if getattr(self.cfg, 'scheduled_sampling_enabled', False):
            logger.info(
                f"Scheduled Sampling: start_epoch={self.cfg.scheduled_sampling_start_epoch}, "
                f"max_prob={self.cfg.scheduled_sampling_max_prob}, "
                f"rampup={self.cfg.scheduled_sampling_rampup_epochs} epochs, "
                f"max_replace={self.cfg.scheduled_sampling_max_replace}, "
                f"ddim_steps={self.cfg.scheduled_sampling_ddim_steps}"
            )

        epoch_pbar = tqdm(
            range(self.epoch, self.cfg.max_epochs),
            desc="训练进度",
            unit="epoch",
            initial=self.epoch,
            total=self.cfg.max_epochs,
        )
        for epoch in epoch_pbar:
            self.epoch = epoch
            train_loss = self._train_one_epoch()

            epoch_pbar.set_postfix(
                loss=f"{train_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                best=f"{self.best_val_loss:.4f}",
            )
            logger.info(
                f"Epoch {epoch+1}/{self.cfg.max_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            ss_enabled = getattr(self.cfg, 'scheduled_sampling_enabled', False)
            ss_start = getattr(self.cfg, 'scheduled_sampling_start_epoch', 100)
            if ss_enabled and epoch >= ss_start:
                ss_rampup = getattr(self.cfg, 'scheduled_sampling_rampup_epochs', 100)
                ss_max_prob = getattr(self.cfg, 'scheduled_sampling_max_prob', 0.5)
                ss_progress = min(1.0, (epoch - ss_start) / max(ss_rampup, 1))
                ss_prob = ss_max_prob * ss_progress
                n_replace = max(1, int(ss_progress * getattr(self.cfg, 'scheduled_sampling_max_replace', 3)))
                if self.writer:
                    self.writer.add_scalar("train/ss_prob", ss_prob, epoch + 1)
                    self.writer.add_scalar("train/ss_n_replace", n_replace, epoch + 1)
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Scheduled Sampling: prob={ss_prob:.2f}, n_replace={n_replace}")

            if (epoch + 1) % self.cfg.eval_every == 0:
                val_loss = self._validate()
                logger.info(f"  验证 loss={val_loss:.6f} (best={self.best_val_loss:.6f})")

                if self.writer:
                    self.writer.add_scalar("val/loss", val_loss, epoch + 1)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(f"best.pt", is_best=True)
                    logger.info(f"  ✓ 新的最佳模型已保存")
                else:
                    self.patience_counter += 1
                    logger.info(
                        f"  ✗ 无改善 ({self.patience_counter}/{self.cfg.early_stopping_patience})"
                    )

                if self.patience_counter >= self.cfg.early_stopping_patience:
                    logger.info(f"Early Stopping: 连续 {self.patience_counter} 次验证无改善")
                    break

            if (epoch + 1) % (self.cfg.eval_every * 2) == 0:
                old_ckpt = os.path.join(self.ckpt_dir, "latest.pt")
                self._save_checkpoint("latest.pt")

        self._save_checkpoint("final.pt")
        if self.writer:
            self.writer.close()
        logger.info("训练完成!")

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        prefetcher = DataPrefetcher(self.train_loader, self.device)

        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        batch_pbar = tqdm(
            enumerate(prefetcher),
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch+1}",
            unit="batch",
            leave=False,
        )
        for batch_idx, batch in batch_pbar:
            condition = batch["condition"]
            target = batch["target"]

            if not condition.is_cuda:
                condition = condition.to(self.device)
                target = target.to(self.device)

            if self.cfg.condition_noise_sigma > 0:
                rampup = min(1.0, self.epoch / max(self.cfg.condition_noise_rampup_epochs, 1))
                noise_sigma = self.cfg.condition_noise_sigma * rampup
                noise_mask = (torch.rand(condition.shape[0], 1, 1, 1, device=condition.device)
                              < self.cfg.condition_noise_prob).float()

                white_noise = torch.randn_like(condition)

                if getattr(self.cfg, 'condition_noise_spatial_smooth', False):
                    # 平滑噪声模拟自回归里的结构性偏差
                    ks = getattr(self.cfg, 'condition_noise_smooth_kernel', 5)
                    pad = ks // 2
                    smooth_noise = F.avg_pool2d(
                        white_noise, kernel_size=ks, stride=1, padding=pad
                    )
                    smooth_std = smooth_noise.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                    smooth_noise = smooth_noise / smooth_std
                    noise = 0.5 * white_noise + 0.5 * smooth_noise
                else:
                    noise = white_noise

                condition = condition + noise_mask * noise_sigma * noise

            ss_enabled = getattr(self.cfg, 'scheduled_sampling_enabled', False)
            ss_start = getattr(self.cfg, 'scheduled_sampling_start_epoch', 50)
            if ss_enabled and self.epoch >= ss_start:
                # 让模型提前见到自己的预测结果
                ss_rampup = getattr(self.cfg, 'scheduled_sampling_rampup_epochs', 100)
                ss_max_prob = getattr(self.cfg, 'scheduled_sampling_max_prob', 0.7)
                ss_max_replace = getattr(self.cfg, 'scheduled_sampling_max_replace', 4)
                ss_ddim_steps = getattr(self.cfg, 'scheduled_sampling_ddim_steps', 25)

                ss_progress = min(1.0, (self.epoch - ss_start) / max(ss_rampup, 1))
                ss_prob = ss_max_prob * ss_progress

                if torch.rand(1).item() < ss_prob:
                    n_replace = max(1, int(ss_progress * ss_max_replace))
                    C = self.data_cfg.num_channels
                    T_hist = self.data_cfg.history_steps
                    H = W = self.data_cfg.grid_size
                    B_ss = condition.shape[0]

                    cond_5d = condition.reshape(B_ss, T_hist, C, H, W)

                    # 这里只生成替换帧，不更新采样链梯度
                    self.model.eval()
                    ss_window = cond_5d.clone()

                    for k in range(n_replace):
                        ss_cond = ss_window.reshape(B_ss, -1, H, W)
                        pred_frame = self.model.fast_sample(
                            ss_cond, self.device, ddim_steps=ss_ddim_steps
                        )
                        pred_step = pred_frame[:, :C]

                        ss_window = torch.cat([
                            ss_window[:, 1:],
                            pred_step.unsqueeze(1),
                        ], dim=1)

                    self.model.train()

                    condition = ss_window.reshape(B_ss, -1, H, W)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target)
                loss = (outputs["loss_mse"]
                        + self.cfg.physics_loss_weight * outputs["loss_div"]
                        + self.cfg.vorticity_loss_weight * outputs["loss_curl"])
                loss = loss / self.cfg.gradient_accumulation_steps

            loss_val = loss.item()
            if not torch.isfinite(loss) or loss_val > 10.0:
                logger.warning(
                    f"[Step {self.global_step}] 异常 loss={loss_val:.4f}, 跳过此 batch"
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                continue

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            self.global_step += 1

            if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                self.ema.update(self.model)
                self.optim_step += 1

                total_loss += accum_loss
                num_batches += 1

                if self.optim_step % self.cfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    mse_val = outputs["loss_mse"].item()
                    div_val = outputs["loss_div"].item()
                    curl_val = outputs["loss_curl"].item()
                    batch_pbar.set_postfix(
                        loss=f"{accum_loss:.4f}",
                        mse=f"{mse_val:.4f}",
                        div=f"{div_val:.4f}",
                        curl=f"{curl_val:.4f}",
                        lr=f"{lr:.2e}",
                    )
                    if self.writer:
                        self.writer.add_scalar("train/loss_total", accum_loss, self.optim_step)
                        self.writer.add_scalar("train/loss_mse", mse_val, self.optim_step)
                        self.writer.add_scalar("train/loss_div", div_val, self.optim_step)
                        self.writer.add_scalar("train/loss_curl", curl_val, self.optim_step)
                        self.writer.add_scalar("train/lr", lr, self.optim_step)

                        if self.optim_step % (self.cfg.log_every * 10) == 0:
                            with torch.no_grad():
                                eps_pred = outputs["eps_pred"]
                                eps_true = outputs["eps_true"]
                                C = eps_pred.shape[1]
                                per_ch_mse = ((eps_pred - eps_true) ** 2).mean(dim=(0, 2, 3))
                                ch_names = []
                                for var in self.data_cfg.pressure_level_vars:
                                    for lev in self.data_cfg.pressure_levels:
                                        ch_names.append(f"{var}_{lev}")
                                for var in self.data_cfg.surface_vars:
                                    ch_names.append(var)
                                for ci in range(min(C, len(ch_names))):
                                    self.writer.add_scalar(
                                        f"channel_mse/{ch_names[ci]}",
                                        per_ch_mse[ci].item(),
                                        self.optim_step,
                                    )
                    logger.info(
                        f"  step={self.optim_step} | "
                        f"loss={accum_loss:.6f} | "
                        f"mse={mse_val:.6f} | div={div_val:.6f} | curl={curl_val:.6f} | "
                        f"lr={lr:.2e}"
                    )

                accum_loss = 0.0

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.ema.apply_shadow(self.model)
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        val_pbar = tqdm(self.val_loader, desc="验证中", unit="batch", leave=False)
        for batch in val_pbar:
            condition = batch["condition"].to(self.device)
            target = batch["target"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target)
                loss = (outputs["loss_mse"]
                        + self.cfg.physics_loss_weight * outputs["loss_div"]
                        + self.cfg.vorticity_loss_weight * outputs["loss_curl"])

            total_loss += loss.item()
            num_batches += 1

        self.ema.restore(self.model)
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        path = os.path.join(self.ckpt_dir, filename)
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "optim_step": self.optim_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
        }
        torch.save(state, path)
        logger.info(f"Checkpoint 已保存: {path}")

    def _load_checkpoint(self, path: str):
        logger.info(f"加载 checkpoint: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(
            state["model_state_dict"], strict=False
        )
        if missing:
            logger.warning(f"⚠️ 模型中有 {len(missing)} 个 key 未从 checkpoint 加载 (使用随机初始化):")
            for k in missing:
                logger.warning(f"  MISSING: {k}")
        if unexpected:
            logger.warning(f"⚠️ checkpoint 中有 {len(unexpected)} 个 key 在当前模型中不存在 (被忽略):")
            for k in unexpected:
                logger.warning(f"  UNEXPECTED: {k}")
        if not missing and not unexpected:
            logger.info("✅ 模型权重完全匹配，全部加载成功")
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.ema.load_state_dict(state["ema_state_dict"])
        self.epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.optim_step = state["optim_step"]
        self.best_val_loss = state["best_val_loss"]
        self.patience_counter = state["patience_counter"]

        remaining_epochs = self.cfg.max_epochs - self.epoch
        steps_per_epoch = max(
            len(self.train_loader) // self.cfg.gradient_accumulation_steps, 1
        )
        remaining_optim_steps = steps_per_epoch * remaining_epochs
        warmup_steps = self.cfg.warmup_steps
        cosine_steps = max(remaining_optim_steps - warmup_steps, 1)

        for pg in self.optimizer.param_groups:
            pg["lr"] = self.cfg.learning_rate

        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.cfg.warmup_start_lr / self.cfg.learning_rate,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=self.cfg.min_lr,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[warmup_steps],
        )

        logger.info(
            f"恢复训练: epoch={self.epoch}, optim_step={self.optim_step}, "
            f"best_val_loss={self.best_val_loss:.6f}"
        )
        logger.info(
            f"学习率调度器已重建: warmup {warmup_steps} steps → "
            f"peak_lr={self.cfg.learning_rate:.2e} → "
            f"cosine decay {cosine_steps} steps → min_lr={self.cfg.min_lr:.2e}"
        )

def _find_ar_sequences(test_dataset, ar_steps: int, num_samples: int):
    if len(test_dataset) == 0:
        return []

    sequences = []
    current_tid = None
    current_start = -1
    current_len = 0

    for idx in range(len(test_dataset)):
        sample_info = test_dataset.samples[idx]
        tid = sample_info[0]

        if tid == current_tid:
            current_len += 1
        else:
            if current_tid is not None and current_len >= ar_steps:
                sequences.append((current_start, current_len, current_tid))
            current_tid = tid
            current_start = idx
            current_len = 1

    if current_tid is not None and current_len >= ar_steps:
        sequences.append((current_start, current_len, current_tid))

    valid_starts = []
    for seg_start, seg_len, tid in sequences:
        max_start = seg_start + seg_len - ar_steps
        for idx in range(seg_start, max_start + 1):
            valid_starts.append(idx)
            if len(valid_starts) >= num_samples:
                return valid_starts

    return valid_starts

@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    test_dataset,
    data_cfg,
    infer_cfg,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    num_samples: int = 5,
):
    from inference import ERA5Predictor, evaluate_predictions

    predictor = ERA5Predictor(model, data_cfg, infer_cfg, norm_mean, norm_std, device)

    ar_steps = infer_cfg.autoregressive_steps
    noise_sigma = infer_cfg.autoregressive_noise_sigma
    C = data_cfg.num_channels

    effective_num = num_samples if num_samples > 0 else len(test_dataset)
    valid_starts = _find_ar_sequences(test_dataset, ar_steps, effective_num)
    n_samples = len(valid_starts)

    if n_samples == 0:
        logger.warning("测试集中无足够长的连续序列，跳过自回归评估")
        return

    logger.info(f"找到 {n_samples} 个有效评估起始点")

    all_preds = [[] for _ in range(ar_steps)]
    all_gts = [[] for _ in range(ar_steps)]

    for sample_idx, start_idx in enumerate(
        tqdm(valid_starts, desc="测试集自回归推理", unit="样本")
    ):
        sample = test_dataset[start_idx]
        cond = sample["condition"].unsqueeze(0).to(device)

        preds = predictor.predict_autoregressive(
            cond, num_steps=ar_steps, noise_sigma=noise_sigma,
        )

        for t in range(ar_steps):
            all_preds[t].append(preds[t].cpu())
            gt_sample = test_dataset[start_idx + t]
            gt_step = gt_sample["target"][:C]
            all_gts[t].append(gt_step.unsqueeze(0))

    valid_preds = [torch.cat(all_preds[t], dim=0) for t in range(ar_steps)]
    valid_gts = [torch.cat(all_gts[t], dim=0) for t in range(ar_steps)]

    var_names = []
    for var in data_cfg.pressure_level_vars:
        for lev in data_cfg.pressure_levels:
            var_names.append(f"{var}_{lev}")
    for var in data_cfg.surface_vars:
        var_names.append(var)

    results = evaluate_predictions(
        valid_preds, valid_gts, norm_mean, norm_std, var_names,
    )
    rmse = results["rmse"]

    n_leads = rmse.shape[0]
    n_vars = rmse.shape[1]

    header = f"{'时效':>10}"
    for v in var_names:
        header += f"  {v:>8}"
    logger.info(header)
    logger.info("=" * (10 + 10 * n_vars))

    for t in range(n_leads):
        lead_h = (t + 1) * data_cfg.time_interval_hours
        row = f"+ {lead_h:>3}h    "
        for c in range(n_vars):
            row += f"  {rmse[t, c]:>8.2f}"
        logger.info(row)

    mean_row = f"{'平均':>10}"
    for c in range(n_vars):
        mean_row += f"  {rmse[:, c].mean():>8.2f}"
    logger.info(mean_row)

def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="ERA5-Diffusion Training")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                        help="ERA5 数据根目录；也可通过 TYPHOON_DATA_ROOT 设置")
    parser.add_argument("--work_dir", type=str, default=DIFFUSION_DIR)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess_dir", type=str, default=DEFAULT_PREPROCESS_DIR,
                        help="预处理 NPY 目录；也可通过 TYPHOON_PREPROCESS_DIR 设置")
    parser.add_argument("--test_samples", type=int, default=0, help="测试评估样本数, 0=全部")
    parser.add_argument("--split_by_year", action="store_true", default=False,
                        help="按年份划分数据集 (train=1950-2016, val=2017-2018, test=2019-2021)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_cfg, model_cfg, train_cfg, infer_cfg = get_config(data_root=args.data_root)

    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.epochs:
        train_cfg.max_epochs = args.epochs
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.resume:
        train_cfg.resume_from = args.resume
    train_cfg.seed = args.seed

    if args.preprocess_dir:
        data_cfg.preprocessed_dir = args.preprocess_dir
        logger.info(f"检查预处理数据: {args.preprocess_dir}")
        preprocess_to_npy(
            data_root=args.data_root,
            output_dir=args.preprocess_dir,
            pl_vars=data_cfg.pressure_level_vars,
            sfc_vars=data_cfg.surface_vars,
            pressure_levels=data_cfg.pressure_levels,
        )

    data_cfg.norm_stats_path = os.path.join(args.work_dir, "norm_stats.pt")

    logger.info("构建数据加载器...")
    train_loader, val_loader, test_loader, norm_mean, norm_std = build_dataloaders(
        data_cfg, train_cfg, split_by_year=args.split_by_year
    )
    logger.info(f"训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} batches")
    logger.info(f"验证集: {len(val_loader.dataset)} 样本")

    logger.info("构建模型...")
    model = ERA5DiffusionModel(model_cfg, data_cfg, train_cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")

    trainer = Trainer(model, train_loader, val_loader, train_cfg, data_cfg, args.work_dir)
    trainer.train()

    logger.info("=" * 60)
    logger.info("在测试集上评估 (自回归 24步 → 72h)...")

    best_ckpt_path = os.path.join(args.work_dir, train_cfg.checkpoint_dir, "best.pt")
    if not os.path.exists(best_ckpt_path):
        logger.warning(f"找不到 best checkpoint: {best_ckpt_path}, 跳过测试集评估")
    else:
        device = trainer.device
        eval_model = ERA5DiffusionModel(model_cfg, data_cfg,train_cfg).to(device)
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)

        if "ema_state_dict" in ckpt:
            eval_model.load_state_dict(ckpt["model_state_dict"])
            ema = EMA(eval_model, decay=train_cfg.ema_decay)
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.apply_shadow(eval_model)
            logger.info("已加载 best checkpoint 的 EMA 参数")
        else:
            eval_model.load_state_dict(ckpt["model_state_dict"])
            logger.info("已加载 best checkpoint (无 EMA)")

        eval_model.eval()

        test_dataset = test_loader.dataset
        evaluate_on_test(
            eval_model, test_dataset, data_cfg, infer_cfg,
            norm_mean, norm_std, device, num_samples=args.test_samples,
        )
        logger.info("测试集评估完成!")

if __name__ == "__main__":
    main()
