"""
src/training/trainer.py — fixed version.

Key changes from original
──────────────────────────
1. Phase 1 (epochs 1–80): diffusion loss ONLY.  No perceptual, no adversarial.
   The denoiser must converge on noise prediction before any image-level loss
   is applied.  Original phase 1 was only 50 epochs and included perceptual +
   histogram losses that caused noise prediction collapse (eps_pred std → 0.1).

2. Phase 2 (epochs 81–100): diffusion + light perceptual (weight 0.05).
   No adversarial — it destabilises training at this dataset scale (890 images).
   No discriminator update step.

3. Removed the loss_weights override in TrainerConfig.  Weights are now
   controlled exclusively by CompositeLoss.set_phase(), eliminating the
   inconsistency between trainer.py and composite.py weight definitions.

4. grad_clip reduced to 0.5 (from 1.0) for more stable phase-2 training.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

from src.data.physics_dataset import PhysicsUIEBDataModule, PhysicsDataModuleConfig
from src.losses.composite import CompositeLoss, LossWeights
from src.models.p_uwdm import PUWDM, PUWDMConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    # ── paths ──────────────────────────────────────────────────────────────
    data_root: str = "dataset/UIEB"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs/p_uwdm"

    # ── training schedule ──────────────────────────────────────────────────
    total_epochs: int = 100
    phase1_epochs: int = 80  # FIXED: was 50 — need longer pure-diffusion warmup

    # ── optimiser ──────────────────────────────────────────────────────────
    lr_generator: float = 2e-4
    lr_discriminator: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 0.5  # FIXED: reduced from 1.0

    # ── data ───────────────────────────────────────────────────────────────
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    image_size: int = 256

    # ── diffusion ──────────────────────────────────────────────────────────
    num_train_timesteps: int = 1000

    # ── precision ──────────────────────────────────────────────────────────
    use_amp: bool = True
    compile_model: bool = True

    # ── EMA ────────────────────────────────────────────────────────────────
    ema_decay: float = 0.9999
    ema_update_every: int = 10

    # ── checkpointing ──────────────────────────────────────────────────────
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3

    # ── model config ───────────────────────────────────────────────────────
    model: PUWDMConfig = field(default_factory=PUWDMConfig)

    # NOTE: loss_weights removed — weights are now controlled by
    # CompositeLoss.set_phase() to avoid config inconsistency.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: PUWDM,
    opt_g: AdamW,
    opt_d: AdamW,
    sched_g,
    sched_d,
    scaler: torch.amp.GradScaler,
    best_val_loss: float,
) -> None:
    ema_model = getattr(model, "_ema", None)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema_model.shadow if ema_model is not None else None,
            "opt_g_state": opt_g.state_dict(),
            "opt_d_state": opt_d.state_dict(),
            "sched_g_state": sched_g.state_dict(),
            "sched_d_state": sched_d.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )
    log.info("Saved checkpoint → %s", path)


def _prune_old_checkpoints(ckpt_dir: Path, keep: int) -> None:
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=os.path.getmtime)
    for old in ckpts[:-keep]:
        old.unlink()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PUWDMTrainer:
    """
    Two-phase trainer for P-UWDM (fixed).

    Phase 1 (epochs 1–80):   Diffusion loss only.  No image-level losses.
    Phase 2 (epochs 81–100): Diffusion + light perceptual (low timesteps).
    """

    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.use_amp else torch.float32

        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
        )
        self.writer = SummaryWriter(log_dir=cfg.log_dir)

        self._build_data()
        self._build_model()
        self._build_loss()
        self._build_optimisers()

        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
        self.global_step = 0
        self.best_val_loss = math.inf

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_data(self) -> None:
        cfg = self.cfg
        dm_cfg = PhysicsDataModuleConfig(
            raw_dir=str(Path(cfg.data_root) / "raw"),
            ref_dir=str(Path(cfg.data_root) / "reference"),
            split_manifest=str(Path(cfg.data_root) / "split_manifest.json"),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            load_size=(cfg.image_size, cfg.image_size),
        )
        self.dm = PhysicsUIEBDataModule(dm_cfg)
        self.dm.setup()
        self.train_loader = self.dm.train_dataloader()
        self.val_loader = self.dm.val_dataloader()
        log.info(
            "Data: %d train / %d val batches (batch_size=%d)",
            len(self.train_loader),
            len(self.val_loader),
            cfg.batch_size,
        )

    def _build_model(self) -> None:
        cfg = self.cfg
        self.model = PUWDM(cfg.model).to(self.device)
        if cfg.compile_model:
            log.info("torch.compile() — this takes ~60 s on first run …")
            self.model = torch.compile(self.model)
        log.info("Model params: %s M", f"{_count_params(self.model) / 1e6:.1f}")

    def _build_loss(self) -> None:
        # Weights are set by set_phase() — no override from config.
        self.criterion = CompositeLoss().to(self.device)

    def _build_optimisers(self) -> None:
        cfg = self.cfg

        self.opt_g = AdamW(
            self.model.parameters(),
            lr=cfg.lr_generator,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        # Discriminator optimiser kept for checkpoint compat, but not stepped.
        self.opt_d = AdamW(
            self.criterion.discriminator.parameters(),
            lr=cfg.lr_discriminator,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        warmup_steps = 5
        self.sched_g = SequentialLR(
            self.opt_g,
            schedulers=[
                LinearLR(
                    self.opt_g,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                CosineAnnealingLR(
                    self.opt_g, T_max=cfg.total_epochs - warmup_steps, eta_min=1e-6
                ),
            ],
            milestones=[warmup_steps],
        )
        self.sched_d = CosineAnnealingLR(
            self.opt_d, T_max=cfg.phase1_epochs, eta_min=1e-6
        )

    # ------------------------------------------------------------------
    # Phase control
    # ------------------------------------------------------------------

    def _enter_phase1(self) -> None:
        log.info("═" * 60)
        log.info(
            "PHASE 1  (epochs 1–%d): DIFFUSION LOSS ONLY",
            self.cfg.phase1_epochs,
        )
        log.info("  eps_pred std should rise from ~0.1 → ~1.0 by epoch 40")
        log.info("═" * 60)
        self.criterion.set_phase(1)
        _freeze(self.criterion.discriminator)

    def _enter_phase2(self) -> None:
        log.info("═" * 60)
        log.info(
            "PHASE 2  (epochs %d–%d): diffusion + perceptual (t<200 only)",
            self.cfg.phase1_epochs + 1,
            self.cfg.total_epochs,
        )
        log.info("═" * 60)
        self.criterion.set_phase(2)
        # Discriminator stays frozen — no adversarial training

    # ------------------------------------------------------------------
    # Core train / val steps
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, phase: int) -> dict[str, float]:
        self.model.train()
        device, dtype = self.device, self.dtype

        raw = batch["raw"].to(device, non_blocking=True)
        ref = batch["reference"].to(device, non_blocking=True)
        ambient = batch["ambient"].to(device, non_blocking=True)
        transmission = batch["transmission"].to(device, non_blocking=True)
        degradation = batch["degradation"].to(device, non_blocking=True)
        severity = batch["severity"].to(device, non_blocking=True)

        B = raw.size(0)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=self.cfg.use_amp):
            step_out = self.model.training_step(
                {
                    "raw": raw,
                    "reference": ref,
                    "ambient": ambient,
                    "transmission": transmission,
                    "degradation": degradation,
                    "severity": severity,
                }
            )
            loss_dict = self.criterion(
                noise_pred=step_out["noise_pred"],
                noise_target=step_out["noise_target"],
                timesteps=step_out["timesteps"],
                alphas_cumprod=step_out["alphas_cumprod"],
                enhanced=step_out["enhanced"],
                reference=ref,
                raw=raw,
            )
            g_loss = loss_dict["total"]

        # Generator update
        self.opt_g.zero_grad(set_to_none=True)
        self.scaler.scale(g_loss).backward()
        self.scaler.unscale_(self.opt_g)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.opt_g)
        self.scaler.update()
        self.global_step += 1

        # EMA update
        if self.global_step % self.cfg.ema_update_every == 0:
            self.model.update_ema()

        return {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }

    @torch.no_grad()
    def _val_step(self, batch: dict) -> float:
        self.model.eval()
        device, dtype = self.device, self.dtype
        raw = batch["raw"].to(device, non_blocking=True)
        ref = batch["reference"].to(device, non_blocking=True)
        ambient = batch["ambient"].to(device, non_blocking=True)
        transmission = batch["transmission"].to(device, non_blocking=True)
        degradation = batch["degradation"].to(device, non_blocking=True)
        severity = batch["severity"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=self.cfg.use_amp):
            step_out = self.model.training_step(
                {
                    "raw": raw,
                    "reference": ref,
                    "ambient": ambient,
                    "transmission": transmission,
                    "degradation": degradation,
                    "severity": severity,
                }
            )
            # Validation always uses phase-1 weights (diffusion only) for a
            # clean, comparable signal across both phases.
            val_loss = self.criterion.diffusion_loss(
                step_out["noise_pred"],
                step_out["noise_target"],
                step_out["timesteps"],
                step_out["alphas_cumprod"],
            )
        return val_loss.item()

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, phase: int) -> dict[str, float]:
        t0 = time.perf_counter()
        running: dict[str, float] = {}
        n_batches = len(self.train_loader)

        for i, batch in enumerate(self.train_loader):
            losses = self._train_step(batch, phase=phase)
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v

            if (i + 1) % 20 == 0:
                step_losses = {k: v / (i + 1) for k, v in running.items()}
                log.info(
                    "  step %4d/%d  total=%.4f  diff=%.4f  perc=%.4f",
                    i + 1,
                    n_batches,
                    step_losses.get("total", 0),
                    step_losses.get("diffusion", 0),
                    step_losses.get("perceptual", 0),
                )

        avg = {k: v / n_batches for k, v in running.items()}
        elapsed = time.perf_counter() - t0
        log.info(
            "Epoch %3d/%d  [train]  total=%.4f  diff=%.4f  time=%.0fs",
            epoch,
            self.cfg.total_epochs,
            avg.get("total", 0),
            avg.get("diffusion", 0),
            elapsed,
        )
        return avg

    def _run_val(self, epoch: int) -> float:
        total = 0.0
        for batch in self.val_loader:
            total += self._val_step(batch)
        avg = total / len(self.val_loader)
        log.info(
            "Epoch %3d/%d  [val]    diff_loss=%.4f",
            epoch,
            self.cfg.total_epochs,
            avg,
        )
        return avg

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def _try_resume(self, ckpt_path: Optional[str]) -> int:
        if ckpt_path is None:
            return 1

        log.info("Resuming from %s", ckpt_path)
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        self.opt_g.load_state_dict(ck["opt_g_state"])
        self.opt_d.load_state_dict(ck["opt_d_state"])
        self.sched_g.load_state_dict(ck["sched_g_state"])
        self.sched_d.load_state_dict(ck["sched_d_state"])
        self.scaler.load_state_dict(ck["scaler_state"])
        self.best_val_loss = ck.get("best_val_loss", math.inf)
        ema_state = ck.get("ema_state")
        if ema_state is not None and getattr(self.model, "_ema", None) is not None:
            self.model._ema.shadow = ema_state
        return ck["epoch"] + 1

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(self, resume_from: Optional[str] = None) -> None:
        cfg = self.cfg
        start_epoch = self._try_resume(resume_from)
        log.info("Starting from epoch %d", start_epoch)

        phase = 2 if start_epoch > cfg.phase1_epochs else 1
        if phase == 1:
            self._enter_phase1()
        else:
            self._enter_phase2()

        for epoch in range(start_epoch, cfg.total_epochs + 1):

            if epoch == cfg.phase1_epochs + 1 and phase == 1:
                self._enter_phase2()
                phase = 2

            train_losses = self._run_epoch(epoch, phase=phase)
            val_loss = self._run_val(epoch)

            self.sched_g.step()

            for k, v in train_losses.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            self.writer.add_scalar("val/diffusion_loss", val_loss, epoch)
            self.writer.add_scalar(
                "lr/generator", self.opt_g.param_groups[0]["lr"], epoch
            )

            # Log eps_pred std every 10 epochs as a health check
            if epoch % 10 == 0:
                self._log_eps_std(epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                _save_checkpoint(
                    self.ckpt_dir / "best.pt",
                    epoch,
                    self.model,
                    self.opt_g,
                    self.opt_d,
                    self.sched_g,
                    self.sched_d,
                    self.scaler,
                    self.best_val_loss,
                )

            if epoch % cfg.save_every_n_epochs == 0:
                _save_checkpoint(
                    self.ckpt_dir / f"epoch_{epoch:04d}.pt",
                    epoch,
                    self.model,
                    self.opt_g,
                    self.opt_d,
                    self.sched_g,
                    self.sched_d,
                    self.scaler,
                    self.best_val_loss,
                )
                _prune_old_checkpoints(self.ckpt_dir, cfg.keep_last_n_checkpoints)

        self.writer.close()
        log.info("Training complete. Best val diff_loss: %.4f", self.best_val_loss)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_eps_std(self, epoch: int) -> None:
        """
        Log eps_pred std on a single batch — should approach ~1.0 as the
        denoiser learns to correctly predict unit-variance noise.
        Values well below 1.0 (< 0.5) indicate noise prediction collapse.
        """
        self.model.eval()
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        device, dtype = self.device, self.dtype
        raw = batch["raw"].to(device)
        ref = batch["reference"].to(device)
        ambient = batch["ambient"].to(device)
        transmission = batch["transmission"].to(device)
        degradation = batch["degradation"].to(device)
        severity = batch["severity"].to(device)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=self.cfg.use_amp):
            step_out = self.model.training_step(
                {
                    "raw": raw,
                    "reference": ref,
                    "ambient": ambient,
                    "transmission": transmission,
                    "degradation": degradation,
                    "severity": severity,
                }
            )
        eps_std = step_out["noise_pred"].float().std().item()
        log.info(
            "Epoch %3d  eps_pred std=%.4f  (target ~1.0;  <0.5 = collapse)",
            epoch,
            eps_std,
        )
        self.writer.add_scalar("diag/eps_pred_std", eps_std, epoch)
