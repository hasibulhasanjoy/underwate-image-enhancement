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
  • UCIQE     (scalar)     — underwater image quality (no reference; this
                             implementation's Lab-space formula is NOT
                             normalized to [0-1] — typical values are ~10-40)
  • UIQM      (scalar)     — underwater image quality measure

Side-by-side comparison grids (input | enhanced | GT) are saved to:
  results/visuals/

A CSV summary and per-image table are saved to:
  results/metrics.csv
  results/summary.txt

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best.pt --num_steps 50
    python evaluate.py --checkpoint checkpoints/epoch_0100.pt --no_save_visuals
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
from torchvision.utils import make_grid

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
# Tensor utilities
# ──────────────────────────────────────────────────────────────────────────────


def to_uint8(t: Tensor) -> np.ndarray:
    """(C, H, W) float [0,1] tensor → (H, W, C) uint8 numpy array."""
    return (
        (t.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
    )


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

    chroma = np.sqrt(a**2 + b**2)
    sigma_c = chroma.std()

    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(L > 1e-6, chroma / L, 0.0)
    mean_s = sat.mean()

    L_flat = L.flatten()
    con_l = np.percentile(L_flat, 99) - np.percentile(L_flat, 1)

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe = c1 * sigma_c + c2 * con_l + c3 * mean_s
    return float(uciqe)


# ──────────────────────────────────────────────────────────────────────────────
# UIQM  (Panetta et al. 2016)
# ──────────────────────────────────────────────────────────────────────────────


def _uicm(img_rgb: np.ndarray) -> float:
    R = img_rgb[:, :, 0].astype(np.float64)
    G = img_rgb[:, :, 1].astype(np.float64)
    B = img_rgb[:, :, 2].astype(np.float64)
    RG = R - G
    YB = (R + G) / 2.0 - B
    mu_rg, sigma_rg = RG.mean(), RG.std()
    mu_yb, sigma_yb = YB.mean(), YB.std()
    l = math.sqrt(mu_rg**2 + mu_yb**2)
    r = math.sqrt(sigma_rg**2 + sigma_yb**2)
    return -0.0268 * l + 0.1586 * r


def _uism(img_rgb: np.ndarray) -> float:
    from skimage.filters import sobel

    val = 0.0
    weights = [0.299, 0.587, 0.114]
    for c, w in enumerate(weights):
        ch = img_rgb[:, :, c].astype(np.float64) / 255.0
        edge = sobel(ch)
        val += w * _eme(edge)
    return val


def _eme(img: np.ndarray, block: int = 8) -> float:
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
    gray = (
        0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    ).astype(np.float64) / 255.0
    return _eme(gray, block)


def compute_uiqm(img_uint8: np.ndarray) -> float:
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

    # Instantiate a plain (uncompiled) model for inference
    model = PUWDM(PUWDMConfig())
    model_state = _strip_compiled_prefix(ck["model_state"])
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        log.warning("Missing keys in model_state (%d): %s", len(missing), missing[:5])
    if unexpected:
        log.warning(
            "Unexpected keys in model_state (%d): %s", len(unexpected), unexpected[:5]
        )

    epoch = ck.get("epoch", "?")
    best = ck.get("best_val_loss", float("nan"))
    log.info("Checkpoint epoch=%s  best_val_loss=%.4f", epoch, best)

    # Skip EMA entirely — use raw model weights.
    # The EMA shadow was initialised when the model was still in early training
    # (collapsed eps_pred std ~0.1). With decay=0.9999 over 100 epochs it never
    # caught up to the raw weights (eps_pred std ~1.0+). Applying EMA reverts
    # the model to a worse state. Disable it so model.sample() never swaps it in.
    model._ema = None
    log.info("Using raw model weights (EMA disabled — raw weights are better trained).")

    model.eval()
    model.to(device)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Build test DataLoader
# ──────────────────────────────────────────────────────────────────────────────


def build_test_loader(
    data_root: str, image_size: int, batch_size: int, num_workers: int
):
    """Return a DataLoader over the UIEB test split.

    NOTE: this file has a documented history of the ImageNet-denorm bug
    silently reappearing on fresh uploads (see project notes). To make this
    file self-defending against that regression, `imagenet_normalised` is
    pinned EXPLICITLY to False here rather than relying on
    PhysicsDatasetConfig's dataclass default — the same explicit pattern
    used in src/training/trainer.py._build_data(). This pipeline never
    applies real ImageNet normalisation (identity normalize during training,
    no transform at all here during eval — both stay in [0,1]), so this
    must always be False. Do not remove this explicit override even if the
    dataclass default looks correct at the time — that's exactly the
    assumption that broke before.
    """
    from src.data.physics_dataset import (
        PhysicsUIEBDataModule,
        PhysicsDataModuleConfig,
        PhysicsDatasetConfig,
    )

    ds_cfg = PhysicsDatasetConfig(
        load_size=(image_size, image_size),
        physics_on_augmented=True,
        imagenet_normalised=False,  # EXPLICIT — see docstring above
    )

    cfg = PhysicsDataModuleConfig(
        raw_dir=str(Path(data_root) / "raw"),
        ref_dir=str(Path(data_root) / "reference"),
        split_manifest=str(Path(data_root) / "split_manifest.json"),
        load_size=(image_size, image_size),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        dataset_cfg=ds_cfg,
        use_lsui=False,  # EXPLICIT — test split must stay pure UIEB, always
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

    Images are in [0, 1] throughout — no ImageNet normalisation is used.
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
            use_ema=False,  # EMA disabled — raw weights already loaded
            progress=False,
        )  # (B, 3, H, W) — already in [0, 1], same space as the dataset

    # NOTE: no denormalization here. `imagenet_normalised=True` in the dataset
    # config does NOT actually apply normalization (verified: raw tensor
    # min=0, max=1, no negatives) — all data is already [0, 1]. Applying an
    # ImageNet mean/std denorm on top of already-[0,1] data was the exact
    # bug that produced invalid inflated metrics (PSNR=30.72, SSIM=0.964)
    # and washed-out visuals in an earlier evaluation pass. Clamp only.
    enhanced_01 = enhanced_norm.clamp(0.0, 1.0)
    raw_01 = raw.clamp(0.0, 1.0)
    ref_01 = reference.clamp(0.0, 1.0)

    # Log min/max of first image in batch for debugging
    log.debug(
        "enhanced range: [%.4f, %.4f]  raw: [%.4f, %.4f]",
        enhanced_01[0].min().item(),
        enhanced_01[0].max().item(),
        raw_01[0].min().item(),
        raw_01[0].max().item(),
    )

    lpips_mod = sys.modules.get("lpips")

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

        # No-reference metrics
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


def save_annotated_grid(
    raw_01: Tensor,
    enh_01: Tensor,
    ref_01: Tensor,
    metrics: Dict[str, float],
    out_path: Path,
    idx: int,
    psnr_target: float = 22.0,
    ssim_target: float = 0.85,
) -> Path:
    """
    Save a [Input | Enhanced | Reference] grid with a metrics panel burned
    in underneath — PSNR/SSIM/LPIPS/UCIQE/UIQM plus a MEETS/BELOW TARGET
    tag (vs. the thesis targets PSNR>22 dB, SSIM>0.85) — so image quality
    can be judged directly from the file, without cross-referencing the CSV.

    Returns the path the file was saved to (filename encodes idx + SSIM so
    files sort meaningfully by name as well as via ranking.txt).
    """
    from PIL import ImageDraw, ImageFont

    grid = make_grid(
        torch.stack([raw_01, enh_01, ref_01], dim=0),
        nrow=3,
        padding=4,
        pad_value=1.0,
    )
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    grid_img = Image.fromarray(grid_np)

    panel_h = 92
    canvas = Image.new("RGB", (grid_img.width, grid_img.height + panel_h), "white")
    canvas.paste(grid_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default(size=16)
        font_small = ImageFont.load_default(size=13)
    except TypeError:
        # Older Pillow without the `size` kwarg on load_default()
        font = font_small = ImageFont.load_default()

    meets_target = metrics["psnr"] >= psnr_target and metrics["ssim"] >= ssim_target
    tag = "MEETS TARGET" if meets_target else "BELOW TARGET"
    tag_color = (0, 130, 0) if meets_target else (180, 0, 0)

    y = grid_img.height + 6
    draw.text((10, y), f"idx={idx:04d}   {tag}", fill=tag_color, font=font)
    draw.text(
        (10, y + 22),
        f"PSNR={metrics['psnr']:.2f} dB   SSIM={metrics['ssim']:.4f}   "
        f"LPIPS={metrics['lpips']:.4f}",
        fill="black",
        font=font_small,
    )
    draw.text(
        (10, y + 42),
        f"UCIQE={metrics['uciqe']:.2f}   UIQM={metrics['uiqm']:.2f}",
        fill="black",
        font=font_small,
    )
    draw.text(
        (10, y + 64),
        "Input | Enhanced | Reference",
        fill=(90, 90, 90),
        font=font_small,
    )

    fname = f"idx{idx:04d}_ssim{metrics['ssim']:.3f}_psnr{metrics['psnr']:.1f}.png"
    save_path = out_path / fname
    canvas.save(save_path)
    return save_path


def write_ranking_file(all_results: List[Dict], out_path: Path) -> None:
    """
    Write ranking.txt sorted worst-to-best by SSIM, so poor performers can
    be found immediately without opening metrics.csv or scanning all images.
    """
    ranked = sorted(all_results, key=lambda r: r["ssim"])
    lines = [
        "Ranked WORST -> BEST by SSIM",
        "=" * 70,
        f"{'idx':>5}  {'ssim':>7}  {'psnr':>7}  {'lpips':>7}  {'uciqe':>7}  {'uiqm':>7}",
        "-" * 70,
    ]
    for r in ranked:
        lines.append(
            f"{r['idx']:>5}  {r['ssim']:>7.4f}  {r['psnr']:>7.2f}  "
            f"{r['lpips']:>7.4f}  {r['uciqe']:>7.2f}  {r['uiqm']:>7.2f}"
        )
    (out_path / "ranking.txt").write_text("\n".join(lines))
    log.info("Ranking (worst->best by SSIM) → %s", out_path / "ranking.txt")


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
        "--no_save_visuals",
        dest="save_visuals",
        action="store_false",
        default=True,
        help="Skip saving side-by-side comparison images (saved by default).",
    )
    p.add_argument(
        "--max_visuals",
        type=int,
        default=0,
        help="Max annotated visual grids to save, 0 = save all test images "
        "(default: all — was previously capped at 30). Use a small number "
        "for a fast preview run.",
    )
    p.add_argument(
        "--psnr_target",
        type=float,
        default=22.0,
        help="PSNR threshold for the MEETS/BELOW TARGET tag on each image.",
    )
    p.add_argument(
        "--ssim_target",
        type=float,
        default=0.85,
        help="SSIM threshold for the MEETS/BELOW TARGET tag on each image.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for results. If not set, defaults to "
        "'results_<checkpoint_dir>_<checkpoint_stem>' so evaluating multiple "
        "checkpoints (e.g. best.pt vs epoch_0300.pt) never silently "
        "overwrites a previous run's metrics/visuals.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Output directories ────────────────────────────────────────────────
    if args.out_dir is None:
        ckpt_path = Path(args.checkpoint)
        # e.g. checkpoints_v2_full_lsui/epoch_0300.pt -> results_checkpoints_v2_full_lsui_epoch_0300
        auto_name = f"results_{ckpt_path.parent.name}_{ckpt_path.stem}"
        out_dir = Path(auto_name)
        log.info(
            "--out_dir not set; auto-naming from checkpoint -> %s "
            "(pass --out_dir explicitly to override)",
            out_dir,
        )
    else:
        out_dir = Path(args.out_dir)
    visual_dir = out_dir / "visuals_annotated"
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
            metrics = {k: v for k, v in img_result.items() if not k.startswith("_")}
            metrics["idx"] = global_idx

            if args.save_visuals and (
                args.max_visuals == 0 or visual_count < args.max_visuals
            ):
                save_annotated_grid(
                    img_result["_raw_01"],
                    img_result["_enh_01"],
                    img_result["_ref_01"],
                    metrics=metrics,
                    out_path=visual_dir,
                    idx=global_idx,
                    psnr_target=args.psnr_target,
                    ssim_target=args.ssim_target,
                )
                visual_count += 1

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
        f"  UCIQE  (↑)  : {agg['uciqe']:.4f}      ± {agg_std['uciqe']:.4f}   (no fixed target — compare vs. baseline run)",
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

    # ── Save aggregate JSON ───────────────────────────────────────────────
    json_path = out_dir / "aggregate.json"
    json_path.write_text(
        json.dumps(
            {k: {"mean": float(agg[k]), "std": float(agg_std[k])} for k in keys},
            indent=2,
        )
    )
    log.info("Aggregate JSON → %s", json_path)

    if args.save_visuals:
        write_ranking_file(all_results, visual_dir)
        log.info(
            "Annotated visual grids (%d/%d) → %s",
            visual_count,
            len(all_results),
            visual_dir,
        )

    return agg


if __name__ == "__main__":
    main()
