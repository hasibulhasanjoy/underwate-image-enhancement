"""
evaluate.py
────────────────────────────────────────────────────────────────────────────
P-UWDM Evaluation Pipeline
────────────────────────────────────────────────────────────────────────────

Loads the best checkpoint (best.pt), runs DDIM 50-step inference on the
UIEB test split, and computes:
  • PSNR      (dB)         — structural fidelity
  • SSIM      [0-1]        — perceptual similarity
  • LPIPS     [0-1]        — deep perceptual distance (lower = better)
  • UCIQE     [0-1]        — underwater image quality (no reference)
  • UIQM      (scalar)     — underwater image quality measure

Side-by-side comparison grids (input | enhanced | GT) are saved to:
  results/visuals/

A CSV summary and per-image table are saved to:
  results/metrics.csv
  results/summary.txt

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best.pt --num_steps 50
    python evaluate.py --checkpoint checkpoints/epoch_0100.pt --save_visuals
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.utils import make_grid, save_image

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports (so missing optional deps give a clear error)
# ──────────────────────────────────────────────────────────────────────────────


def _import_lpips():
    try:
        import lpips

        return lpips
    except ImportError:
        log.error("lpips not installed. Run: pip install lpips")
        sys.exit(1)


def _import_skimage():
    try:
        from skimage.metrics import structural_similarity, peak_signal_noise_ratio

        return structural_similarity, peak_signal_noise_ratio
    except ImportError:
        log.error("scikit-image not installed. Run: pip install scikit-image")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet denorm (mirrors physics_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(t: Tensor) -> Tensor:
    """ImageNet-normalised tensor → [0, 1] float32."""
    mean = _IMAGENET_MEAN.to(t.device)
    std = _IMAGENET_STD.to(t.device)
    return (t * std + mean).clamp_(0.0, 1.0)


def to_uint8(t: Tensor) -> np.ndarray:
    """(C, H, W) float [0,1] tensor → (H, W, C) uint8 numpy array."""
    return (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# UCIQE  (Yang et al. 2015)
# ──────────────────────────────────────────────────────────────────────────────


def compute_uciqe(img_uint8: np.ndarray) -> float:
    """
    UCIQE from a uint8 RGB image (H, W, 3).
    Coefficients: c1=0.4680, c2=0.2745, c3=0.2576 (original paper).
    """
    from skimage.color import rgb2lab

    lab = rgb2lab(img_uint8.astype(np.float32) / 255.0)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Chroma
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = chroma.std()

    # Saturation
    # Avoid division by zero where L≈0
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(L > 1e-6, chroma / L, 0.0)
    mean_s = sat.mean()

    # Luminance contrast (top 1% - bottom 1% of L)
    L_flat = L.flatten()
    con_l = np.percentile(L_flat, 99) - np.percentile(L_flat, 1)

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe = c1 * sigma_c + c2 * con_l + c3 * mean_s
    return float(uciqe)


# ──────────────────────────────────────────────────────────────────────────────
# UIQM  (Panetta et al. 2016)
# ──────────────────────────────────────────────────────────────────────────────


def _uicm(img_rgb: np.ndarray) -> float:
    """Underwater Image Colorfulness Measure."""
    R = img_rgb[:, :, 0].astype(np.float64)
    G = img_rgb[:, :, 1].astype(np.float64)
    B = img_rgb[:, :, 2].astype(np.float64)
    RG = R - G
    YB = (R + G) / 2.0 - B
    mu_rg, sigma_rg = RG.mean(), RG.std()
    mu_yb, sigma_yb = YB.mean(), YB.std()
    # Asymmetric alpha-trim not strictly needed for thesis; using simpler form
    l = math.sqrt(mu_rg**2 + mu_yb**2)
    r = math.sqrt(sigma_rg**2 + sigma_yb**2)
    return -0.0268 * l + 0.1586 * r


def _uism(img_rgb: np.ndarray) -> float:
    """Underwater Image Sharpness Measure (via Sobel on each channel)."""
    from skimage.filters import sobel

    val = 0.0
    weights = [0.299, 0.587, 0.114]  # R, G, B luminance weights
    for c, w in enumerate(weights):
        ch = img_rgb[:, :, c].astype(np.float64) / 255.0
        edge = sobel(ch)
        # EME on the edge map (block-based)
        val += w * _eme(edge)
    return val


def _eme(img: np.ndarray, block: int = 8) -> float:
    """Enhancement Measure Estimation."""
    H, W = img.shape
    bH = H // block
    bW = W // block
    if bH == 0 or bW == 0:
        return 0.0
    total = 0.0
    count = 0
    for i in range(bH):
        for j in range(bW):
            patch = img[i * block : (i + 1) * block, j * block : (j + 1) * block]
            mn, mx = patch.min(), patch.max()
            if mx > 1e-6 and mn > 1e-6:
                total += math.log(mx / mn)
            count += 1
    return (2.0 / count) * total if count else 0.0


def _uiconm(img_rgb: np.ndarray, block: int = 8) -> float:
    """Underwater Image Contrast Measure."""
    gray = (
        0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    ).astype(np.float64) / 255.0
    return _eme(gray, block)


def compute_uiqm(img_uint8: np.ndarray) -> float:
    """
    UIQM from uint8 RGB (H, W, 3).
    UIQM = c1*UICM + c2*UISM + c3*UIConM
    c1=0.0282, c2=0.2953, c3=3.5753  (Panetta et al. 2016)
    """
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm = _uicm(img_uint8)
    uism = _uism(img_uint8)
    uiconm = _uiconm(img_uint8)
    return c1 * uicm + c2 * uism + c3 * uiconm


# ──────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ──────────────────────────────────────────────────────────────────────────────


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """
    torch.compile() wraps the model and prepends '_orig_mod.' to every
    parameter name in the state dict.  Strip it so the weights load cleanly
    into a plain (uncompiled) PUWDM instance at eval time.
    """
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict  # already clean
    stripped = {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
    log.info("Stripped '_orig_mod.' prefix from %d state-dict keys.", len(stripped))
    return stripped


def load_model(checkpoint_path: str, device: torch.device):
    """Load PUWDM from a training checkpoint and return it in eval mode."""
    from src.models.p_uwdm import PUWDM, PUWDMConfig

    log.info("Loading checkpoint: %s", checkpoint_path)
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = PUWDM(PUWDMConfig())
    model_state = _strip_compiled_prefix(ck["model_state"])
    model.load_state_dict(model_state)

    # Apply EMA weights for inference if available
    ema_state = ck.get("ema_state")
    if ema_state is not None and model._ema is not None:
        # EMA shadow keys may carry _orig_mod. prefix from torch.compile
        # and may be scoped as 'denoiser.X' or '_orig_mod.denoiser.X'
        def _fix_ema_keys(sd):
            out = {}
            for k, v in sd.items():
                k = k.replace("_orig_mod.", "")
                if k.startswith("denoiser."):
                    k = k[len("denoiser.") :]
                out[k] = v
            return out

        model._ema.shadow = _fix_ema_keys(ema_state)
        model._ema.copy_to(model.denoiser)
        log.info("EMA weights applied to denoiser.")
    else:
        log.info("No EMA state found — using raw model weights.")

    epoch = ck.get("epoch", "?")
    best = ck.get("best_val_loss", float("nan"))
    log.info("Checkpoint epoch=%s  best_val_loss=%.4f", epoch, best)

    model.eval()
    model.to(device)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Build test DataLoader from existing PhysicsUIEBDataModule
# ──────────────────────────────────────────────────────────────────────────────


def build_test_loader(
    data_root: str, image_size: int, batch_size: int, num_workers: int
):
    """Return a DataLoader over the UIEB test split."""
    from src.data.physics_dataset import PhysicsUIEBDataModule, PhysicsDataModuleConfig

    cfg = PhysicsDataModuleConfig(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    dm = PhysicsUIEBDataModule(cfg)
    dm.setup()
    return dm.test_dataloader()


# ──────────────────────────────────────────────────────────────────────────────
# Per-batch evaluation
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_batch(
    model,
    batch: Dict[str, Tensor],
    device: torch.device,
    num_steps: int,
    lpips_fn,
    ssim_fn,
    psnr_fn,
) -> List[Dict]:
    """
    Run inference + metric computation on a single batch.
    Returns a list of per-image metric dicts.
    """
    raw = batch["raw"].to(device)
    reference = batch["reference"].to(device)
    ambient = batch["ambient"].to(device)
    transmission = batch["transmission"].to(device)
    degradation = batch["degradation"].to(device)
    severity = batch["severity"].to(device)

    with torch.no_grad():
        enhanced_norm = model.sample(
            raw=raw,
            physics_A=ambient,
            physics_t=transmission,
            degradation=degradation,
            severity=severity,
            num_steps=num_steps,
            eta=0.0,
            use_ema=True,
            progress=False,
        )  # (B, 3, H, W) in ImageNet-normalised space

    # Denormalise all tensors to [0, 1] for metric computation
    enhanced_01 = denorm(enhanced_norm)  # (B, 3, H, W)
    raw_01 = denorm(raw)
    ref_01 = denorm(reference)

    B = raw_01.shape[0]
    results = []

    for i in range(B):
        enh_np = to_uint8(enhanced_01[i])  # (H, W, 3) uint8
        ref_np = to_uint8(ref_01[i])
        raw_np = to_uint8(raw_01[i])

        # PSNR
        psnr_val = psnr_fn(ref_np, enh_np, data_range=255)

        # SSIM
        ssim_val = ssim_fn(
            ref_np,
            enh_np,
            data_range=255,
            channel_axis=2,
            win_size=7,
        )

        # LPIPS — expects (1, 3, H, W) in [-1, 1]
        enh_lpips = enhanced_01[i : i + 1] * 2 - 1
        ref_lpips = ref_01[i : i + 1] * 2 - 1
        lpips_val = lpips_fn(enh_lpips.to(device), ref_lpips.to(device)).item()

        # No-reference metrics on enhanced image
        uciqe_val = compute_uciqe(enh_np)
        uiqm_val = compute_uiqm(enh_np)

        results.append(
            {
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "uciqe": uciqe_val,
                "uiqm": uiqm_val,
                # tensors for visual saving
                "_raw_01": raw_01[i].cpu(),
                "_enh_01": enhanced_01[i].cpu(),
                "_ref_01": ref_01[i].cpu(),
            }
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visual grid saving
# ──────────────────────────────────────────────────────────────────────────────


def save_visual_grid(
    raw_01: Tensor,
    enh_01: Tensor,
    ref_01: Tensor,
    out_path: Path,
    idx: int,
):
    """Save a side-by-side [Input | Enhanced | GT] grid."""
    grid = make_grid(
        torch.stack([raw_01, enh_01, ref_01], dim=0),
        nrow=3,
        padding=4,
        pad_value=1.0,
    )
    save_image(grid, out_path / f"sample_{idx:04d}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="P-UWDM evaluation pipeline")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data_root", default="dataset/UIEB")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--num_steps", type=int, default=50, help="DDIM sampling steps (default 50)"
    )
    p.add_argument(
        "--save_visuals",
        action="store_true",
        default=True,
        help="Save side-by-side comparison images",
    )
    p.add_argument(
        "--max_visuals",
        type=int,
        default=30,
        help="Max visual grids to save (set 0 for all)",
    )
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Output directories ────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    visual_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visuals:
        visual_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── LPIPS (lazy) ──────────────────────────────────────────────────────
    lpips_mod = _import_lpips()
    lpips_fn = lpips_mod.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    # ── skimage metrics ───────────────────────────────────────────────────
    ssim_fn, psnr_fn = _import_skimage()

    # ── DataLoader ────────────────────────────────────────────────────────
    log.info("Building test DataLoader (data_root=%s)", args.data_root)
    test_loader = build_test_loader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    n_test = len(test_loader.dataset)
    log.info("Test set size: %d images", n_test)

    # ── Evaluation loop ───────────────────────────────────────────────────
    all_results = []
    visual_count = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(test_loader):
        log.info(
            "Batch %d/%d  (%d images processed)",
            batch_idx + 1,
            len(test_loader),
            batch_idx * args.batch_size,
        )

        per_img = evaluate_batch(
            model,
            batch,
            device,
            num_steps=args.num_steps,
            lpips_fn=lpips_fn,
            ssim_fn=ssim_fn,
            psnr_fn=psnr_fn,
        )

        for img_result in per_img:
            global_idx = len(all_results)

            # Save visual grid
            if args.save_visuals and (
                args.max_visuals == 0 or visual_count < args.max_visuals
            ):
                save_visual_grid(
                    img_result["_raw_01"],
                    img_result["_enh_01"],
                    img_result["_ref_01"],
                    visual_dir,
                    global_idx,
                )
                visual_count += 1

            # Strip tensor fields before storing
            metrics = {k: v for k, v in img_result.items() if not k.startswith("_")}
            metrics["idx"] = global_idx
            all_results.append(metrics)

    elapsed = time.time() - t0
    log.info(
        "Evaluation complete in %.1fs (%.2fs/image)",
        elapsed,
        elapsed / max(len(all_results), 1),
    )

    # ── Aggregate ─────────────────────────────────────────────────────────
    keys = ["psnr", "ssim", "lpips", "uciqe", "uiqm"]
    agg = {k: np.mean([r[k] for r in all_results]) for k in keys}
    agg_std = {k: np.std([r[k] for r in all_results]) for k in keys}

    # ── Print summary ─────────────────────────────────────────────────────
    summary_lines = [
        "",
        "═" * 55,
        "  P-UWDM EVALUATION RESULTS — UIEB Test Split",
        "═" * 55,
        f"  Checkpoint  : {args.checkpoint}",
        f"  DDIM steps  : {args.num_steps}",
        f"  Test images : {len(all_results)}",
        "─" * 55,
        f"  PSNR   (↑)  : {agg['psnr']:.4f} dB   ± {agg_std['psnr']:.4f}   [target >22]",
        f"  SSIM   (↑)  : {agg['ssim']:.4f}      ± {agg_std['ssim']:.4f}   [target >0.85]",
        f"  LPIPS  (↓)  : {agg['lpips']:.4f}      ± {agg_std['lpips']:.4f}",
        f"  UCIQE  (↑)  : {agg['uciqe']:.4f}      ± {agg_std['uciqe']:.4f}   [target >0.6]",
        f"  UIQM   (↑)  : {agg['uiqm']:.4f}      ± {agg_std['uiqm']:.4f}",
        "═" * 55,
        "",
    ]
    print("\n".join(summary_lines))

    # ── Save summary text ─────────────────────────────────────────────────
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    log.info("Summary saved → %s", summary_path)

    # ── Save per-image CSV ────────────────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx"] + keys)
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Per-image CSV → %s", csv_path)

    # ── Save aggregate JSON (easy to load in thesis notebook) ─────────────
    json_path = out_dir / "aggregate.json"
    json_path.write_text(
        json.dumps(
            {k: {"mean": float(agg[k]), "std": float(agg_std[k])} for k in keys},
            indent=2,
        )
    )
    log.info("Aggregate JSON → %s", json_path)

    if args.save_visuals:
        log.info("Visual grids (%d) → %s", visual_count, visual_dir)

    return agg


if __name__ == "__main__":
    main()
