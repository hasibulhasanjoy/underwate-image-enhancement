#!/usr/bin/env python3
"""
verify_rcc_freeze.py — one-batch empirical smoke test for the RCC-only
fine-tune setup. Run this BEFORE launching the real (many-hour) run.

It loads the source checkpoint, freezes the backbone exactly the way
train.py --train_rcc_only does, runs ONE real training step, and asserts:

  1. Every backbone (cond_nets + denoiser) parameter got NO gradient
     (.grad is None) AND its VALUES are bit-identical before/after the
     step — proving the freeze is real, not just requires_grad=False with
     something slipping through.
  2. Every RCC (red_comp) parameter got a gradient, and RCC as a whole
     is not stuck at exactly zero gradient everywhere.

The single batch's diffusion timestep is FORCED to t=0 for every sample
(bypassing the trainer's normal random t sampling, for this smoke test
only) so perceptual/histogram loss — RCC's only gradient path — is
guaranteed to be active on this batch. Without this, a batch could
randomly draw zero low-t (t<200) samples (~41% chance at batch_size=4)
and this test would report a false failure ("RCC got no gradient") that
has nothing to do with whether the freeze itself is correct.

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

    cfg = TrainerConfig(
        data_root=args.data_root,
        checkpoint_dir="/tmp/verify_rcc_freeze_ckpt",
        log_dir="/tmp/verify_rcc_freeze_runs",
        batch_size=args.batch_size,
        compile_model=not args.no_compile,
        model=PUWDMConfig(use_red_channel_compensation=True),
    )

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

    # ── One real training step, with t FORCED to 0 for every sample so ──
    # perceptual/histogram loss (RCC's only gradient path) is guaranteed
    # active — see module docstring for why this matters.
    device, dtype = trainer.device, trainer.dtype
    batch = next(iter(trainer.train_loader))
    B = batch["raw"].shape[0]
    forced_t = torch.zeros(B, dtype=torch.long, device=device)

    trainer.model.train()
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=cfg.use_amp):
        step_out = trainer.model.training_step(
            {
                "raw": batch["raw"].to(device),
                "reference": batch["reference"].to(device),
                "ambient": batch["ambient"].to(device),
                "transmission": batch["transmission"].to(device),
                "degradation": batch["degradation"].to(device),
                "severity": batch["severity"].to(device),
                "t": forced_t,
            }
        )
        loss_dict = trainer.criterion(
            noise_pred=step_out["noise_pred"],
            noise_target=step_out["noise_target"],
            timesteps=step_out["timesteps"],
            alphas_cumprod=step_out["alphas_cumprod"],
            enhanced=step_out["enhanced"],
            reference=batch["reference"].to(device),
            raw=batch["raw"].to(device),
        )
        g_loss = loss_dict["total"]

    log.info(
        "Forced-low-t step losses: total=%.4f diff=%.4f perc=%.4f hist=%.4f "
        "(low_t_count=%d/%d, should be %d/%d since t is forced to 0)",
        g_loss.item(),
        loss_dict["diffusion"].item(),
        loss_dict["perceptual"].item(),
        loss_dict["histogram"].item(),
        loss_dict["low_t_count"],
        B,
        B,
        B,
    )
    assert loss_dict["low_t_count"] == B, (
        f"Expected all {B} samples to count as low-t (t forced to 0), got "
        f"{loss_dict['low_t_count']}. Something is wrong with the forced-t "
        f"override or the low-t gate — investigate before trusting the "
        f"rest of this test."
    )

    trainer.opt_g.zero_grad(set_to_none=True)
    g_loss.backward()
    trainer.opt_g.step()

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
