"""
src/training/trainer.py — fixed version + LSUI combined-training support.

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

5. NEW — LSUI combined-training support (cfg.use_lsui / lsui_raw_dir /
   lsui_ref_dir). When enabled, LSUI pairs are appended to the TRAIN split
   only; UIEB val/test (the fixed 134-image thesis benchmark) are untouched.

6. NEW — augmentation is now actually wired in. Previously ``_build_data``
   passed no transform to PhysicsUIEBDataModule at all, so every run to
   date (including all phase1/2/2-hist/3-hist checkpoints) trained on
   RAW, UNAUGMENTED images despite configs/data_config.yaml defining a
   full augmentation pipeline. cfg.augment=True (default) now builds real
   flip/rotation/color-jitter transforms via src.data.transforms, loaded
   from configs/data_config.yaml — but with the `normalize` block forced
   to identity (mean=0, std=1) regardless of what the yaml says, because
   the diffusion model (DDIM, clip_denoised=True) expects [0,1] data, not
   ImageNet-normalised data. Set cfg.augment=False to reproduce the old
   (unaugmented) behaviour exactly.

7. NEW — dataset_cfg is now built EXPLICITLY with imagenet_normalised=False
   rather than relying on PhysicsDatasetConfig's dataclass default. This
   was a real bug: the old default (imagenet_normalised=True) caused
   _denorm_for_physics() to be wrongly applied to already-[0,1] pixels,
   corrupting the ambient/transmission/degradation conditioning priors in
   every run to date. See src/data/physics_dataset.py module docstring
   (BUG-3) for full detail.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

from src.data.physics_dataset import (
    PhysicsUIEBDataModule,
    PhysicsDataModuleConfig,
    PhysicsDatasetConfig,
)
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

    # ── phase-2 LR schedule (NEW — fixes perceptual/histogram stall) ────────
    # Previously sched_g was a SINGLE cosine schedule spanning the whole run
    # (total_epochs), so by the time phase 2 turned on new loss terms
    # (perceptual/histogram), LR had already decayed ~80%+ toward eta_min —
    # leaving almost no room to actually learn the new loss composition.
    # Phase 2 now gets its OWN warmup + cosine schedule over just the
    # remaining epochs, starting from lr_phase2 (not wherever the phase-1
    # cosine happened to leave off). Adam's moment estimates (built up over
    # 240 epochs of pure diffusion loss) are also reset by default, since
    # they're stale for gradients from loss terms that didn't exist before.
    lr_phase2: float = 5e-5
    phase2_warmup_epochs: int = 5
    reset_optimizer_on_phase2: bool = True

    # ── data ───────────────────────────────────────────────────────────────
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    image_size: int = 256

    # ── augmentation (NEW) ──────────────────────────────────────────────────
    # If True (default), real flip/rotation/color-jitter augmentation is
    # built from data_config_path — with normalize forced to identity so
    # data stays in [0,1] for the diffusion model. If False, reproduces the
    # old (unaugmented) behaviour of every prior training phase exactly.
    augment: bool = True
    data_config_path: str = "configs/data_config.yaml"

    # ── LSUI combined-training (NEW) ────────────────────────────────────────
    # When use_lsui=True, LSUI pairs are appended to the TRAIN split only.
    # UIEB val/test (fixed 134-image thesis benchmark) are never touched.
    use_lsui: bool = False
    lsui_raw_dir: Optional[str] = None  # e.g. "dataset/LSUI/input"
    lsui_ref_dir: Optional[str] = None  # e.g. "dataset/LSUI/GT"

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

        # ── Augmentation transforms (NEW) ───────────────────────────────
        transform_train = None
        transform_val = None
        if cfg.augment:
            from src.utils.config import load_data_config
            from src.data.transforms import get_train_transforms, get_val_transforms

            data_cfg = load_data_config(
                cfg.data_config_path,
                overrides={
                    "preprocessing": {
                        "image_size": [cfg.image_size, cfg.image_size],
                        # Identity normalize — the diffusion model expects
                        # raw pixels in [0,1] (DDIM, clip_denoised=True),
                        # NOT ImageNet-normalised data. Only the
                        # augmentation ops themselves (flip/rotation/
                        # color-jitter) are taken from the yaml.
                        "normalize": {
                            "mean": [0.0, 0.0, 0.0],
                            "std": [1.0, 1.0, 1.0],
                        },
                    }
                },
            )
            transform_train = get_train_transforms(data_cfg)
            transform_val = get_val_transforms(data_cfg)
            log.info(
                "Augmentation ENABLED (flip/rotation/color-jitter from %s); "
                "normalize forced to identity to preserve [0,1] range.",
                cfg.data_config_path,
            )
        else:
            log.info(
                "Augmentation DISABLED — matching all prior training "
                "phases exactly (raw, unaugmented images)."
            )

        # ── Physics-dataset config: EXPLICIT fix for the imagenet_normalised
        # default bug (see physics_dataset.py module docstring, BUG-3). We
        # never apply real ImageNet normalisation in this pipeline (identity
        # normalize above, or no transform at all), so this must be False.
        ds_cfg = PhysicsDatasetConfig(
            load_size=(cfg.image_size, cfg.image_size),
            physics_on_augmented=True,
            imagenet_normalised=False,
        )

        dm_cfg = PhysicsDataModuleConfig(
            raw_dir=str(Path(cfg.data_root) / "raw"),
            ref_dir=str(Path(cfg.data_root) / "reference"),
            split_manifest=str(Path(cfg.data_root) / "split_manifest.json"),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            load_size=(cfg.image_size, cfg.image_size),
            dataset_cfg=ds_cfg,
            use_lsui=cfg.use_lsui,
            lsui_raw_dir=cfg.lsui_raw_dir,
            lsui_ref_dir=cfg.lsui_ref_dir,
        )
        self.dm = PhysicsUIEBDataModule(
            dm_cfg, transform_train=transform_train, transform_val=transform_val
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

        # NOTE: this schedule covers PHASE 1 ONLY (T_max=phase1_epochs), not
        # the whole run. Phase 2 gets its own fresh warmup+cosine schedule
        # built in _enter_phase2() — see TrainerConfig.lr_phase2 docstring
        # for why: a single run-long cosine left phase 2 with almost no LR
        # budget to learn its newly-introduced loss terms.
        warmup_steps = 5
        phase1_t_max = max(cfg.phase1_epochs - warmup_steps, 1)
        self.sched_g = SequentialLR(
            self.opt_g,
            schedulers=[
                LinearLR(
                    self.opt_g,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                CosineAnnealingLR(self.opt_g, T_max=phase1_t_max, eta_min=1e-6),
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
        cfg = self.cfg
        log.info("═" * 60)
        log.info(
            "PHASE 2  (epochs %d–%d): diffusion + perceptual + histogram (t<200 only)",
            cfg.phase1_epochs + 1,
            cfg.total_epochs,
        )
        log.info("═" * 60)
        self.criterion.set_phase(2)
        # Discriminator stays frozen — no adversarial training

        # ── Fresh LR schedule + optimizer state for phase 2 ─────────────
        # If this is a plain resume INTO an already-running phase 2, the
        # subsequent _load_checkpoint_state() call in fit() will immediately
        # overwrite everything set here with the exact saved state — so it's
        # safe to always rebuild unconditionally. This block only actually
        # changes behavior at the genuine phase-1 → phase-2 transition.
        if cfg.reset_optimizer_on_phase2:
            self.opt_g.state = defaultdict(dict)
            log.info("  Reset opt_g Adam moment estimates for phase 2.")
        for group in self.opt_g.param_groups:
            group["lr"] = cfg.lr_phase2

        remaining = max(cfg.total_epochs - cfg.phase1_epochs, 1)
        warmup = min(cfg.phase2_warmup_epochs, max(remaining - 1, 0))
        if warmup > 0:
            self.sched_g = SequentialLR(
                self.opt_g,
                schedulers=[
                    LinearLR(
                        self.opt_g,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=warmup,
                    ),
                    CosineAnnealingLR(
                        self.opt_g, T_max=max(remaining - warmup, 1), eta_min=1e-6
                    ),
                ],
                milestones=[warmup],
            )
        else:
            self.sched_g = CosineAnnealingLR(self.opt_g, T_max=remaining, eta_min=1e-6)
        log.info(
            "  New phase-2 schedule: lr=%.2e → eta_min=1e-6 over %d epochs "
            "(warmup=%d)",
            cfg.lr_phase2,
            remaining,
            warmup,
        )

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

        # perceptual/histogram are only computed on the subset of samples
        # with t < LOW_T_THRESHOLD (~20% at default settings). Most batches
        # contribute a loss of exactly 0 for these terms. Averaging that
        # naively over n_batches mixes real signal with pure "did this batch
        # happen to draw a low-t sample" noise. Instead, accumulate a
        # sample-count-weighted average so the reported/logged value
        # reflects the true mean over samples that actually contributed —
        # this is what should be watched to judge whether these losses are
        # actually decreasing.
        low_t_weighted = {"perceptual": 0.0, "histogram": 0.0}
        low_t_total = 0

        for i, batch in enumerate(self.train_loader):
            losses = self._train_step(batch, phase=phase)
            count = int(losses.pop("low_t_count", 0))
            for k, v in losses.items():
                if k in ("perceptual", "histogram"):
                    continue
                running[k] = running.get(k, 0.0) + v
            if count > 0:
                low_t_total += count
                low_t_weighted["perceptual"] += losses["perceptual"] * count
                low_t_weighted["histogram"] += losses["histogram"] * count

            if (i + 1) % 20 == 0:
                step_losses = {k: v / (i + 1) for k, v in running.items()}
                running_perc = (
                    low_t_weighted["perceptual"] / low_t_total if low_t_total else 0.0
                )
                running_hist = (
                    low_t_weighted["histogram"] / low_t_total if low_t_total else 0.0
                )
                log.info(
                    "  step %4d/%d  total=%.4f  diff=%.4f  perc=%.4f  hist=%.4f  "
                    "(low_t_samples=%d)",
                    i + 1,
                    n_batches,
                    step_losses.get("total", 0),
                    step_losses.get("diffusion", 0),
                    running_perc,
                    running_hist,
                    low_t_total,
                )

        avg = {k: v / n_batches for k, v in running.items()}
        if low_t_total > 0:
            avg["perceptual"] = low_t_weighted["perceptual"] / low_t_total
            avg["histogram"] = low_t_weighted["histogram"] / low_t_total
        else:
            avg["perceptual"] = 0.0
            avg["histogram"] = 0.0
        avg["low_t_samples_per_epoch"] = float(low_t_total)

        elapsed = time.perf_counter() - t0
        log.info(
            "Epoch %3d/%d  [train]  total=%.4f  diff=%.4f  perc=%.4f  hist=%.4f  "
            "low_t_n=%d  time=%.0fs",
            epoch,
            self.cfg.total_epochs,
            avg.get("total", 0),
            avg.get("diffusion", 0),
            avg.get("perceptual", 0),
            avg.get("histogram", 0),
            low_t_total,
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

    def _load_checkpoint_state(self, ck: dict) -> None:
        """
        Load model/optimizer/scheduler/scaler state from an already-read
        checkpoint dict. Caller (fit()) must first call _enter_phase1() or
        _enter_phase2() — based on the checkpoint's epoch — so self.opt_g /
        self.sched_g already have the correct shape/type for this phase
        before their state is overwritten here. This ordering matters now
        that phase 2 uses a different scheduler than phase 1: loading state
        into the wrong-shaped scheduler would silently corrupt it (plain
        __dict__.update under the hood) or error outright.
        """
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
        log.info("Restored model/optimizer/scheduler/scaler state from checkpoint.")

    def load_weights_from(self, ckpt_path: str) -> None:
        """
        Load ONLY model + EMA weights from a checkpoint — no optimizer,
        scheduler, epoch count, or best_val_loss.

        Use this (instead of fit(resume_from=...)) when starting a new
        fine-tuning phase with a different loss composition or dataset
        (e.g. the combined UIEB+LSUI phase1b re-exposure). A literal resume
        would also restore the old, already-decayed cosine LR schedule,
        effectively training the new epochs at ~eta_min. This gives the
        extension run its own fresh warmup + cosine decay instead.
        """
        log.info(
            "Loading weights ONLY (fresh optimizer/scheduler/epoch count) from %s",
            ckpt_path,
        )
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        ema_state = ck.get("ema_state")
        if ema_state is not None and getattr(self.model, "_ema", None) is not None:
            self.model._ema.shadow = ema_state
        log.info(
            "Loaded weights from source checkpoint epoch %d; this run starts at epoch 1 "
            "with a fresh LR schedule.",
            ck.get("epoch", -1),
        )

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(self, resume_from: Optional[str] = None) -> None:
        cfg = self.cfg

        ck = None
        if resume_from is not None:
            log.info("Resuming from %s", resume_from)
            ck = torch.load(resume_from, map_location=self.device, weights_only=False)
            start_epoch = ck["epoch"] + 1
        else:
            start_epoch = 1
        log.info("Starting from epoch %d", start_epoch)

        # Build the phase-appropriate optimizer LR / scheduler shape FIRST,
        # then load checkpoint state into it (see _load_checkpoint_state
        # docstring for why this order matters).
        phase = 2 if start_epoch > cfg.phase1_epochs else 1
        if phase == 1:
            self._enter_phase1()
        else:
            self._enter_phase2()

        if ck is not None:
            self._load_checkpoint_state(ck)

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
