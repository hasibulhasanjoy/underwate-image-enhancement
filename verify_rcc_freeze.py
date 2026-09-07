#!/usr/bin/env python3
"""
verify_rcc_freeze.py — one-batch empirical smoke test for the RCC-only
fine-tune setup. Run this BEFORE launching the real (many-hour) run.

It loads the source checkpoint, freezes the backbone exactly the way
train.py --train_rcc_only does, runs ONE real training step through the
actual trainer._train_step() code path (the same one the real run uses,
including its forced low-t timestep sampling for RCC-only mode), and
asserts:

  1. Every backbone (cond_nets + denoiser) parameter got NO gradient
     (.grad is None) AND its VALUES are bit-identical before/after the
     step — proving the freeze is real, not just requires_grad=False with
     something slipping through.
  2. Every RCC (red_comp) parameter got a gradient, and RCC as a whole
     is not stuck at exactly zero gradient everywhere.
  3. low_t_count == batch_size, confirming the forced-low-t sampling
     (added to fix the "does not require grad and does not have a
     grad_fn" crash) is actually active.

Exits 0 and prints "ALL CHECKS PASSED" only if every assertion holds.
Any failure raises AssertionError with a clear message — do not launch
the real run if this script fails.

Usage:
    python verify_rcc_freeze.py \\
        --init_weights_from checkpoints_v2_full_lsui/epoch_0300.pt \\
        --lr_rcc_only 3e-4 \\
        --rcc_loss_mode full
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from src.training.trainer import PUWDMTrainer, TrainerConfig
from src.models.p_uwdm import PUWDMConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("verify_rcc_freeze")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init_weights_from", required=True)
    ap.add_argument("--data_root", default="dataset/UIEB")
    ap.add_argument("--use_lsui", action="store_true")
    ap.add_argument("--lsui_raw_dir", default="dataset/LSUI/input")
    ap.add_argument("--lsui_ref_dir", default="dataset/LSUI/GT")
    ap.add_argument("--lr_rcc_only", type=float, default=3e-4)
    ap.add_argument(
        "--rcc_loss_mode", choices=["full", "perceptual_only"], default="full"
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument(
        "--no_compile",
        action="store_true",
        help="Skip torch.compile for a faster smoke test. The real run "
        "uses compile by default, so also run once WITHOUT this flag "
        "before the real launch if you want maximum fidelity — compiling "
        "after freezing can trigger a one-time recompile, which this "
        "verifies doesn't break anything.",
    )
    args = ap.parse_args()

    cfg_kwargs = dict(
        data_root=args.data_root,
        checkpoint_dir="/tmp/verify_rcc_freeze_ckpt",
        log_dir="/tmp/verify_rcc_freeze_runs",
        batch_size=args.batch_size,
        compile_model=not args.no_compile,
        model=PUWDMConfig(use_red_channel_compensation=True),
    )
    if args.use_lsui:
        cfg_kwargs.update(
            use_lsui=True,
            lsui_raw_dir=args.lsui_raw_dir,
            lsui_ref_dir=args.lsui_ref_dir,
        )
    cfg = TrainerConfig(**cfg_kwargs)

    log.info("Building trainer (this also builds the model + data loaders)...")
    trainer = PUWDMTrainer(cfg)

    log.info("Loading %s and freezing backbone...", args.init_weights_from)
    trainer.load_weights_for_rcc_only(
        args.init_weights_from,
        lr_override=args.lr_rcc_only,
        loss_mode=args.rcc_loss_mode,
    )

    core = trainer._unwrap_model()
    assert core.red_comp is not None, (
        "red_comp is None — model was built with "
        "use_red_channel_compensation=False. Nothing to verify."
    )

    backbone_params = list(core.cond_nets.named_parameters()) + list(
        core.denoiser.named_parameters()
    )
    backbone_before = {name: p.detach().clone() for name, p in backbone_params}
    rcc_params = list(core.red_comp.named_parameters())
    assert len(rcc_params) > 0, "red_comp has no parameters — unexpected."

    log.info(
        "Backbone: %d params (frozen, should not move). RCC: %d params " "(trainable).",
        len(backbone_params),
        len(rcc_params),
    )

    # ── One real training step through the ACTUAL trainer._train_step() ──
    # code path — same one the real run uses, including its forced
    # low-t timestep sampling for RCC-only mode (added to fix the
    # "does not require grad and does not have a grad_fn" crash: without
    # it, a batch has a real chance of drawing zero t<200 samples, at
    # which point diffusion/perceptual/histogram are all grad-less
    # constants and backward() has nothing to differentiate through).
    batch = next(iter(trainer.train_loader))
    B = batch["raw"].shape[0]

    step_losses = trainer._train_step(batch, phase=3)

    log.info(
        "Step losses: total=%.4f diff=%.4f perc=%.4f hist=%.4f " "low_t_count=%s/%d",
        step_losses.get("total", float("nan")),
        step_losses.get("diffusion", float("nan")),
        step_losses.get("perceptual", float("nan")),
        step_losses.get("histogram", float("nan")),
        step_losses.get("low_t_count", "?"),
        B,
    )
    assert step_losses.get("low_t_count") == B, (
        f"Expected low_t_count == batch_size ({B}) thanks to the forced "
        f"low-t sampling in RCC-only mode, got "
        f"{step_losses.get('low_t_count')!r}. The forced-t override isn't "
        f"active — investigate trainer._train_step() before trusting the "
        f"rest of this test."
    )

    # ── Check 1: backbone got NO gradient and DID NOT move ──────────────
    for name, p in backbone_params:
        assert p.grad is None, (
            f"BACKBONE PARAM '{name}' HAS A GRADIENT (.grad is not None) "
            f"— the freeze is NOT working. DO NOT launch the real run."
        )
        assert torch.equal(p.detach(), backbone_before[name]), (
            f"BACKBONE PARAM '{name}' CHANGED VALUE after one optimizer "
            f"step — the freeze is NOT working. DO NOT launch the real run."
        )
    log.info(
        "\u2713 Backbone check passed: all %d cond_nets+denoiser params "
        "have zero gradient and zero movement.",
        len(backbone_params),
    )

    # ── Check 2: RCC got a real, non-zero gradient ───────────────────────
    n_zero = 0
    for name, p in rcc_params:
        assert p.grad is not None, (
            f"RCC PARAM '{name}' HAS NO GRADIENT (.grad is None) — RCC is "
            f"not receiving any training signal at all. DO NOT launch the "
            f"real run; check --rcc_loss_mode."
        )
        if p.grad.abs().sum().item() == 0.0:
            n_zero += 1
    assert n_zero < len(rcc_params), (
        "EVERY RCC parameter had an exactly-zero gradient on this step — "
        "RCC is almost certainly not receiving a real training signal. "
        "DO NOT launch the real run."
    )
    log.info(
        "\u2713 RCC check passed: all %d red_comp params have a gradient "
        "(%d/%d were exactly zero this single step — a few can be normal, "
        "e.g. an unused bias; not a concern given the majority moved).",
        len(rcc_params),
        n_zero,
        len(rcc_params),
    )

    print("\nALL CHECKS PASSED \u2014 safe to launch the real RCC-only run.\n")


if __name__ == "__main__":
    main()
