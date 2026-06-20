"""
src/training/trainer.py

Two-phase training loop for P-UWDM.

Phase 1 (epochs 1–50):   Train denoiser + conditioning networks only.
                          Discriminator is frozen. No adversarial loss.
Phase 2 (epochs 51–100): Unfreeze discriminator.  All five loss terms active.
                          Lower LR via cosine annealing.

Hardware target: RTX 4090 (24 GB), bfloat16 AMP, torch.compile.
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
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

from src.data.physics_dataset import PhysicsUIEBDataModule
from src.losses.composite import CompositeLoss
from src.models.p_uwdm import PUWDM, PUWDMConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    # ── paths ──────────────────────────────────────────────────────────────
    data_root: str = "data/UIEB"  # root that contains raw/ & reference/
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs/p_uwdm"

    # ── training schedule ──────────────────────────────────────────────────
    total_epochs: int = 100
    phase1_epochs: int = 50  # Phase-1: no discriminator

    # ── optimiser ──────────────────────────────────────────────────────────
    lr_generator: float = 2e-4
    lr_discriminator: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0

    # ── data ───────────────────────────────────────────────────────────────
    batch_size: int = 16  # fits 4090 @ bfloat16
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    image_size: int = 256

    # ── diffusion ──────────────────────────────────────────────────────────
    num_train_timesteps: int = 1000

    # ── precision ──────────────────────────────────────────────────────────
    use_amp: bool = True  # bfloat16 on Ampere+
    compile_model: bool = True  # torch.compile (PyTorch ≥ 2.0)

    # ── EMA ────────────────────────────────────────────────────────────────
    ema_decay: float = 0.9999
    ema_update_every: int = 10  # steps between EMA updates

    # ── checkpointing ──────────────────────────────────────────────────────
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3

    # ── loss weights (passed straight to CompositeLoss) ───────────────────
    loss_weights: dict = field(
        default_factory=lambda: {
            "diffusion": 1.0,
            "adversarial": 0.1,
            "perceptual": 0.05,
            "histogram": 0.1,
            "contrastive": 0.05,
        }
    )

    # ── model config (forwarded to PUWDMConfig) ───────────────────────────
    model: PUWDMConfig = field(default_factory=PUWDMConfig)


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
    scaler: GradScaler,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": model.ema.shadow,
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
    Two-phase trainer for P-UWDM.

    Usage
    -----
    >>> cfg = TrainerConfig()
    >>> trainer = PUWDMTrainer(cfg)
    >>> trainer.fit()
    """

    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.use_amp else torch.float32

        # ── directories ────────────────────────────────────────────────────
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── logging ────────────────────────────────────────────────────────
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
        )
        self.writer = SummaryWriter(log_dir=cfg.log_dir)

        # ── build components ───────────────────────────────────────────────
        self._build_data()
        self._build_model()
        self._build_loss()
        self._build_optimisers()

        self.scaler = GradScaler(enabled=cfg.use_amp)
        self.global_step = 0
        self.best_val_loss = math.inf

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_data(self) -> None:
        cfg = self.cfg
        self.dm = PhysicsUIEBDataModule(
            data_root=cfg.data_root,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            image_size=cfg.image_size,
        )
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
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        log.info("Model params: %s M", f"{_count_params(self.model) / 1e6:.1f}")

    def _build_loss(self) -> None:
        self.criterion = CompositeLoss(
            weights=self.cfg.loss_weights,
            device=self.device,
        ).to(self.device)

    def _build_optimisers(self) -> None:
        cfg = self.cfg

        # Generator: denoiser + conditioning networks
        gen_params = [
            p
            for name, p in self.model.named_parameters()
            if "discriminator" not in name and p.requires_grad
        ]
        self.opt_g = AdamW(
            gen_params,
            lr=cfg.lr_generator,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        # Discriminator
        disc_params = [
            p for name, p in self.model.named_parameters() if "discriminator" in name
        ]
        self.opt_d = AdamW(
            disc_params,
            lr=cfg.lr_discriminator,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        # Schedulers
        #   Phase 1: linear warm-up over first 5 epochs
        #   Phase 2: cosine annealing for remaining epochs
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
            "PHASE 1  (epochs 1–%d): generator-only training", self.cfg.phase1_epochs
        )
        log.info("═" * 60)
        _freeze(self.model.discriminator)

    def _enter_phase2(self) -> None:
        log.info("═" * 60)
        log.info(
            "PHASE 2  (epochs %d–%d): full adversarial training",
            self.cfg.phase1_epochs + 1,
            self.cfg.total_epochs,
        )
        log.info("═" * 60)
        _unfreeze(self.model.discriminator)

    # ------------------------------------------------------------------
    # Core train / val steps
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, phase: int) -> dict[str, float]:
        """
        Returns a dict of scalar losses for logging.
        `batch` keys expected from PhysicsUIEBDataset:
            raw        – degraded image  [B,3,H,W]
            reference  – clean target    [B,3,H,W]
            (physics estimators are run inside PUWDM.training_step)
        """
        self.model.train()
        device, dtype = self.device, self.dtype

        raw = batch["raw"].to(device, non_blocking=True)
        ref = batch["reference"].to(device, non_blocking=True)

        # ── sample random timestep ─────────────────────────────────────
        B = raw.size(0)
        t = torch.randint(0, self.cfg.num_train_timesteps, (B,), device=device)

        # ── forward pass (bfloat16) ────────────────────────────────────
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=self.cfg.use_amp):
            step_out = self.model.training_step({"raw": raw, "reference": ref, "t": t})
            # step_out keys: x_noisy, noise_pred, x0_pred, x_ref, x_raw, t

            loss_dict = self.criterion(step_out, phase=phase)
            g_loss = loss_dict["total"]

        # ── generator update ───────────────────────────────────────────
        self.opt_g.zero_grad(set_to_none=True)
        self.scaler.scale(g_loss).backward(retain_graph=(phase == 2))
        self.scaler.unscale_(self.opt_g)
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if "discriminator" not in str(p)],
            self.cfg.grad_clip,
        )
        self.scaler.step(self.opt_g)

        # ── discriminator update (phase 2 only) ───────────────────────
        d_loss_val = 0.0
        if phase == 2:
            with torch.autocast(
                device_type="cuda", dtype=dtype, enabled=self.cfg.use_amp
            ):
                d_loss = self.criterion.discriminator_loss(step_out)
            self.opt_d.zero_grad(set_to_none=True)
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.opt_d)
            nn.utils.clip_grad_norm_(
                self.model.discriminator.parameters(), self.cfg.grad_clip
            )
            self.scaler.step(self.opt_d)
            d_loss_val = d_loss.item()

        self.scaler.update()

        # ── EMA update ─────────────────────────────────────────────────
        if self.global_step % self.cfg.ema_update_every == 0:
            self.model.ema.update(self.model)

        return {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        } | {"d_loss": d_loss_val}

    @torch.no_grad()
    def _val_step(self, batch: dict) -> float:
        self.model.eval()
        device, dtype = self.device, self.dtype
        raw = batch["raw"].to(device, non_blocking=True)
        ref = batch["reference"].to(device, non_blocking=True)
        B = raw.size(0)
        t = torch.randint(0, self.cfg.num_train_timesteps, (B,), device=device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=self.cfg.use_amp):
            step_out = self.model.training_step({"raw": raw, "reference": ref, "t": t})
            loss_dict = self.criterion(step_out, phase=1)  # no adversarial in val
        return loss_dict["total"].item()

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

            self.global_step += 1

            # log every 20 steps
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
            "Epoch %3d/%d  [train]  total=%.4f  time=%.0fs",
            epoch,
            self.cfg.total_epochs,
            avg.get("total", 0),
            elapsed,
        )
        return avg

    def _run_val(self, epoch: int) -> float:
        total = 0.0
        for batch in self.val_loader:
            total += self._val_step(batch)
        avg = total / len(self.val_loader)
        log.info("Epoch %3d/%d  [val]    total=%.4f", epoch, self.cfg.total_epochs, avg)
        return avg

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def _try_resume(self, ckpt_path: Optional[str]) -> int:
        """Returns start_epoch (1-based)."""
        if ckpt_path is None:
            # auto-find latest
            ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"), key=os.path.getmtime)
            if not ckpts:
                return 1
            ckpt_path = str(ckpts[-1])

        log.info("Resuming from %s", ckpt_path)
        ck = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
        self.model.ema.shadow = ck["ema_state"]
        self.opt_g.load_state_dict(ck["opt_g_state"])
        self.opt_d.load_state_dict(ck["opt_d_state"])
        self.sched_g.load_state_dict(ck["sched_g_state"])
        self.sched_d.load_state_dict(ck["sched_d_state"])
        self.scaler.load_state_dict(ck["scaler_state"])
        self.best_val_loss = ck.get("best_val_loss", math.inf)
        return ck["epoch"] + 1

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(self, resume_from: Optional[str] = None) -> None:
        cfg = self.cfg
        start_epoch = self._try_resume(resume_from)
        log.info("Starting from epoch %d", start_epoch)

        self._enter_phase1()
        phase = 1

        for epoch in range(start_epoch, cfg.total_epochs + 1):

            # phase transition
            if epoch == cfg.phase1_epochs + 1 and phase == 1:
                self._enter_phase2()
                phase = 2

            # ── train ─────────────────────────────────────────────────
            train_losses = self._run_epoch(epoch, phase=phase)

            # ── val ───────────────────────────────────────────────────
            val_loss = self._run_val(epoch)

            # ── schedulers ────────────────────────────────────────────
            self.sched_g.step()
            if phase == 2:
                self.sched_d.step()

            # ── tensorboard ───────────────────────────────────────────
            for k, v in train_losses.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            self.writer.add_scalar("val/total_loss", val_loss, epoch)
            self.writer.add_scalar(
                "lr/generator", self.opt_g.param_groups[0]["lr"], epoch
            )

            # ── checkpoint ────────────────────────────────────────────
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
        log.info("Training complete. Best val loss: %.4f", self.best_val_loss)
