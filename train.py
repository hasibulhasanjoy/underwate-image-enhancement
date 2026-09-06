#!/usr/bin/env python3
"""
train.py — P-UWDM training entry point (fixed + LSUI combined-training support).

Three modes, all through this one script:

1. Fresh run (random init):
       python train.py --total_epochs 100 --phase1_epochs 80

2. Full resume (restores model + optimizer + scheduler + epoch count —
   use this to continue an interrupted run with no change in config):
       python train.py --resume
       python train.py --checkpoint checkpoints/epoch_0050.pt --resume

3. Weights-only init (loads ONLY model + EMA weights from a checkpoint,
   then starts a FRESH optimizer/scheduler/epoch count — use this when
   starting a new fine-tuning phase with a different loss composition or
   dataset, e.g. adding the RCC module or the combined UIEB+LSUI
   re-exposure, so the LR schedule gets a proper warmup + decay instead
   of inheriting an already-decayed one):
       python train.py --init_weights_from checkpoints_phase3_hist/epoch_0150.pt \\
           --checkpoint_dir checkpoints_phase1b_lsui \\
           --log_dir runs/p_uwdm_phase1b_lsui \\
           --total_epochs 120 \\
           --use_lsui --lsui_raw_dir dataset/LSUI/input --lsui_ref_dir dataset/LSUI/GT

   IMPORTANT: for this mode, trainer.py now controls which phase/LR the
   run starts in via --finetune_phase (default 2 = safe fine-tune:
   diffusion+perceptual+histogram at --lr_phase2). --phase1_epochs is
   NOT consulted for mode 3 anymore — it only matters for modes 1/2.
   Do not set --finetune_phase 1 unless you deliberately want to redo
   Phase 1's from-scratch diffusion-only training at --lr_generator on
   top of the loaded weights (this WILL catastrophically forget an
   already-converged checkpoint if used by mistake — see trainer.py's
   load_weights_from() docstring, item 8 in the module header).

Examples for the shared-GPU / small-VRAM case:
    python train.py --batch_size 8             # smaller batch (shared GPU)

NEW — LSUI combined training:
    --use_lsui appends all LSUI (input, GT) pairs to the TRAIN split ONLY.
    The UIEB val/test splits (fixed 134-image thesis benchmark) are never
    touched — evaluate.py still reports against the exact same benchmark
    as every prior phase.

NEW — augmentation:
    Real flip/rotation/color-jitter augmentation is now wired in by default
    (cfg.augment=True), built from --data_config (default
    configs/data_config.yaml). Pass --no_augment to reproduce the old
    (unaugmented) behaviour of every prior training phase exactly.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import PUWDMTrainer, TrainerConfig
from src.models.p_uwdm import PUWDMConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train P-UWDM (fixed)")

    p.add_argument("--data_root", default="dataset/UIEB")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir", default="runs/p_uwdm")

    p.add_argument("--total_epochs", type=int, default=100)
    p.add_argument("--phase1_epochs", type=int, default=80)

    p.add_argument("--lr_generator", type=float, default=2e-4)
    p.add_argument("--lr_discriminator", type=float, default=1e-4)

    # Phase-2 LR schedule (NEW — fixes perceptual/histogram loss stalling
    # flat instead of decreasing; see trainer.py TrainerConfig docstring)
    p.add_argument(
        "--lr_phase2",
        type=float,
        default=5e-5,
        help="Fresh starting LR for phase 2 (diffusion+perceptual+histogram), "
        "instead of inheriting the near-exhausted tail of phase 1's cosine "
        "schedule. Decays via its own cosine schedule to the same eta_min "
        "over the remaining epochs.",
    )
    p.add_argument("--phase2_warmup_epochs", type=int, default=5)
    p.add_argument(
        "--no_reset_optimizer_phase2",
        action="store_true",
        help="Keep Adam's moment estimates from phase 1 when entering phase "
        "2, instead of resetting them (default: reset, since they were "
        "built up under a loss composition that didn't include perceptual/"
        "histogram terms).",
    )

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=256)

    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_update_every", type=int, default=10)
    p.add_argument("--save_every_n_epochs", type=int, default=5)
    p.add_argument("--keep_last_n_checkpoints", type=int, default=3)

    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_compile", action="store_true")

    # Augmentation (NEW)
    p.add_argument(
        "--data_config",
        default="configs/data_config.yaml",
        help="Path to data_config.yaml, used only for augmentation settings "
        "(flip/rotation/color-jitter). normalize is always forced to "
        "identity regardless of what the yaml specifies, to keep data in "
        "[0,1] for the diffusion model.",
    )
    p.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable augmentation entirely, reproducing the exact "
        "(unaugmented) behaviour of every prior training phase.",
    )

    # LSUI combined training (NEW)
    p.add_argument(
        "--use_lsui",
        action="store_true",
        help="Append all LSUI (input, GT) pairs to the TRAIN split only. "
        "UIEB val/test splits (fixed 134-image thesis benchmark) are "
        "never touched.",
    )
    p.add_argument("--lsui_raw_dir", default="dataset/LSUI/input")
    p.add_argument("--lsui_ref_dir", default="dataset/LSUI/GT")

    # Red Channel Compensation (NEW)
    p.add_argument(
        "--no_red_channel_compensation",
        action="store_true",
        help="Disable the physics-guided Red Channel Compensation module "
        "(the 'Red Channel Compensation' block in the architecture "
        "diagram). Useful for an ablation run, or to exactly reproduce a "
        "pre-RCC training pipeline.",
    )

    # Mode: full resume (restores optimizer/scheduler/epoch count)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", default=None)

    # Mode: weights-only init (fresh optimizer/scheduler/epoch count) —
    # use when starting a new fine-tuning phase (e.g. new loss terms, or a
    # new dataset like combined UIEB+LSUI) from an existing checkpoint
    # rather than a literal continuation.
    p.add_argument(
        "--init_weights_from",
        default=None,
        help=(
            "Path to a checkpoint to load ONLY model+EMA weights from. "
            "Starts a fresh optimizer/scheduler/epoch count (epoch 1). "
            "Mutually exclusive with --resume/--checkpoint."
        ),
    )
    p.add_argument(
        "--finetune_phase",
        type=int,
        choices=[1, 2],
        default=2,
        help=(
            "Only used with --init_weights_from. Which phase/schedule the "
            "new run starts in: 2 (default, SAFE) = diffusion+perceptual+"
            "histogram at --lr_phase2, appropriate for adding a module "
            "(e.g. RCC) or a dataset to an already phase-2/3-converged "
            "checkpoint. 1 (DANGEROUS) = from-scratch diffusion-only loss "
            "at --lr_generator, resetting the optimizer to a from-scratch "
            "LR — only for cases like a changed denoiser architecture "
            "that genuinely needs noise prediction relearned from "
            "scratch. Using 1 on an already-converged checkpoint by "
            "mistake causes catastrophic forgetting (confirmed: PSNR "
            "18.26 -> 12.57 dB in 20 epochs on this project)."
        ),
    )
    p.add_argument(
        "--finetune_lr",
        type=float,
        default=None,
        help=(
            "Only used with --init_weights_from and --finetune_phase 2. "
            "Overrides --lr_phase2 for this run specifically. Leave unset "
            "to just use --lr_phase2."
        ),
    )

    args = p.parse_args()

    if args.init_weights_from and (args.resume or args.checkpoint):
        p.error(
            "--init_weights_from cannot be combined with --resume/--checkpoint. "
            "Use --init_weights_from for a fresh fine-tuning phase, or "
            "--resume/--checkpoint to literally continue an interrupted run."
        )

    if (
        args.finetune_phase != 2 or args.finetune_lr is not None
    ) and not args.init_weights_from:
        p.error("--finetune_phase/--finetune_lr only apply with --init_weights_from.")

    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    args = parse_args()

    cfg = TrainerConfig(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        total_epochs=args.total_epochs,
        phase1_epochs=args.phase1_epochs,
        lr_generator=args.lr_generator,
        lr_discriminator=args.lr_discriminator,
        lr_phase2=args.lr_phase2,
        phase2_warmup_epochs=args.phase2_warmup_epochs,
        reset_optimizer_on_phase2=not args.no_reset_optimizer_phase2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_amp=not args.no_amp,
        compile_model=not args.no_compile,
        ema_decay=args.ema_decay,
        ema_update_every=args.ema_update_every,
        save_every_n_epochs=args.save_every_n_epochs,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        augment=not args.no_augment,
        data_config_path=args.data_config,
        use_lsui=args.use_lsui,
        lsui_raw_dir=args.lsui_raw_dir,
        lsui_ref_dir=args.lsui_ref_dir,
        model=PUWDMConfig(
            use_red_channel_compensation=not args.no_red_channel_compensation
        ),
    )

    trainer = PUWDMTrainer(cfg)

    if args.init_weights_from:
        log.info(
            "Mode: WEIGHTS-ONLY INIT from %s — fresh optimizer/scheduler/epoch "
            "count, finetune_phase=%d%s",
            args.init_weights_from,
            args.finetune_phase,
            f", finetune_lr={args.finetune_lr:.2e}" if args.finetune_lr else "",
        )
        if args.finetune_phase == 1:
            log.warning(
                "--finetune_phase 1 requested: this WILL restart from-scratch "
                "diffusion-only training at lr_generator=%.1e on top of the "
                "loaded weights. Double check this is really what you want.",
                args.lr_generator,
            )
        trainer.load_weights_from(
            args.init_weights_from,
            finetune_phase=args.finetune_phase,
            finetune_lr=args.finetune_lr,
        )
        trainer.fit(resume_from=None)
        return

    resume_ckpt = args.checkpoint
    if args.resume and resume_ckpt is None:
        # auto-detect latest epoch checkpoint
        ckpts = sorted(Path(args.checkpoint_dir).glob("epoch_*.pt"))
        resume_ckpt = str(ckpts[-1]) if ckpts else None

    if resume_ckpt:
        log.info("Mode: FULL RESUME from %s", resume_ckpt)
    else:
        log.info("Mode: FRESH RUN (random init)")

    trainer.fit(resume_from=resume_ckpt)


if __name__ == "__main__":
    main()
