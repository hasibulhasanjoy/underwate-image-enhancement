#!/usr/bin/env python3
"""
train.py — P-UWDM training entry point (fixed).

Usage:
    python train.py                            # fresh 100-epoch run
    python train.py --batch_size 4             # smaller batch (shared GPU)
    python train.py --resume                   # resume from latest checkpoint
    python train.py --checkpoint checkpoints/epoch_0050.pt --resume
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

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=256)

    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_compile", action="store_true")

    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", default=None)

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    cfg = TrainerConfig(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        total_epochs=args.total_epochs,
        phase1_epochs=args.phase1_epochs,
        lr_generator=args.lr_generator,
        lr_discriminator=args.lr_discriminator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_amp=not args.no_amp,
        compile_model=not args.no_compile,
        model=PUWDMConfig(),
    )

    trainer = PUWDMTrainer(cfg)

    resume_ckpt = args.checkpoint
    if args.resume and resume_ckpt is None:
        # auto-detect latest epoch checkpoint
        ckpts = sorted(Path(args.checkpoint_dir).glob("epoch_*.pt"))
        resume_ckpt = str(ckpts[-1]) if ckpts else None

    trainer.fit(resume_from=resume_ckpt)


if __name__ == "__main__":
    main()
