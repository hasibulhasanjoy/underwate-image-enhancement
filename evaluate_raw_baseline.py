"""
evaluate_raw_baseline.py
────────────────────────────────────────────────────────────────────────────
P-UWDM — RAW (unenhanced) baseline evaluation

Computes PSNR / SSIM / LPIPS (vs. reference) and UCIQE / UIQM (no-reference)
directly on the *raw* underwater test images — no model, no checkpoint, no
DDIM sampling. This gives the "before enhancement" numbers for the thesis
comparison table (raw vs. P-UWDM output vs. reference).

Reuses the exact same metric functions and DataLoader construction as
evaluate.py (compute_uciqe, compute_uiqm, to_uint8, build_test_loader) so the
raw-baseline numbers are computed identically to the model-output numbers —
same [0,1] clamp-only pipeline, same UIEB test split, same LPIPS net.

Usage:
    python evaluate_raw_baseline.py
    python evaluate_raw_baseline.py --data_root dataset/UIEB --image_size 256
    python evaluate_raw_baseline.py --no_save_visuals
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.utils import make_grid

# ──────────────────────────────────────────────────────────────────────────────
# Reuse everything possible from evaluate.py so metric math never drifts
# between the raw-baseline run and the model-output run.
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # noqa: E402
    to_uint8,
    compute_uciqe,
    compute_uiqm,
    build_test_loader,
    _import_lpips,
    _import_skimage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Per-batch metric computation — raw vs. reference, no model inference
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_batch_raw(
    batch: Dict[str, Tensor],
    lpips_fn,
    ssim_fn,
    psnr_fn,
    device: torch.device,
) -> List[Dict]:
    """Same metric computation as evaluate.evaluate_batch, but the 'enhanced'
    image is just the raw input clamped to [0,1] — no model call at all."""
    raw_01 = batch["raw"].to(device).clamp(0.0, 1.0)
    ref_01 = batch["reference"].to(device).clamp(0.0, 1.0)

    B = raw_01.shape[0]
    results = []

    for i in range(B):
        raw_np = to_uint8(raw_01[i])
        ref_np = to_uint8(ref_01[i])

        psnr_val = psnr_fn(ref_np, raw_np, data_range=255)
        ssim_val = ssim_fn(ref_np, raw_np, data_range=255, channel_axis=2, win_size=7)

        raw_lpips = raw_01[i : i + 1] * 2 - 1
        ref_lpips = ref_01[i : i + 1] * 2 - 1
        lpips_val = lpips_fn(raw_lpips.to(device), ref_lpips.to(device)).item()

        uciqe_val = compute_uciqe(raw_np)
        uiqm_val = compute_uiqm(raw_np)

        results.append(
            {
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "uciqe": uciqe_val,
                "uiqm": uiqm_val,
                "_raw_01": raw_01[i].cpu(),
                "_ref_01": ref_01[i].cpu(),
            }
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visual grid: Raw | Reference (2 panels — no enhanced image exists here)
# ──────────────────────────────────────────────────────────────────────────────


def save_raw_grid(
    raw_01: Tensor,
    ref_01: Tensor,
    metrics: Dict[str, float],
    out_path: Path,
    idx: int,
) -> Path:
    from PIL import ImageDraw, ImageFont

    grid = make_grid(
        torch.stack([raw_01, ref_01], dim=0), nrow=2, padding=4, pad_value=1.0
    )
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    grid_img = Image.fromarray(grid_np)

    panel_h = 70
    canvas = Image.new("RGB", (grid_img.width, grid_img.height + panel_h), "white")
    canvas.paste(grid_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default(size=16)
        font_small = ImageFont.load_default(size=13)
    except TypeError:
        font = font_small = ImageFont.load_default()

    y = grid_img.height + 6
    draw.text(
        (10, y), f"idx={idx:04d}   RAW (no enhancement)", fill=(120, 60, 0), font=font
    )
    draw.text(
        (10, y + 22),
        f"PSNR={metrics['psnr']:.2f} dB   SSIM={metrics['ssim']:.4f}   "
        f"LPIPS={metrics['lpips']:.4f}",
        fill="black",
        font=font_small,
    )
    draw.text(
        (10, y + 42),
        f"UCIQE={metrics['uciqe']:.2f}   UIQM={metrics['uiqm']:.2f}   |   Raw | Reference",
        fill="black",
        font=font_small,
    )

    fname = f"idx{idx:04d}_ssim{metrics['ssim']:.3f}_psnr{metrics['psnr']:.1f}.png"
    save_path = out_path / fname
    canvas.save(save_path)
    return save_path


def write_ranking_file(all_results: List[Dict], out_path: Path) -> None:
    ranked = sorted(all_results, key=lambda r: r["ssim"])
    lines = [
        "RAW BASELINE — Ranked WORST -> BEST by SSIM",
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
    log.info("Ranking (worst->best by SSIM) -> %s", out_path / "ranking.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Raw (unenhanced) baseline metrics on the UIEB test split"
    )
    p.add_argument("--data_root", default="dataset/UIEB")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--no_save_visuals",
        dest="save_visuals",
        action="store_false",
        default=True,
        help="Skip saving raw|reference comparison images (saved by default).",
    )
    p.add_argument(
        "--max_visuals",
        type=int,
        default=0,
        help="Max visual grids to save, 0 = save all test images.",
    )
    p.add_argument(
        "--out_dir",
        default="results_raw_baseline",
        help="Output directory for results (default: results_raw_baseline).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    out_dir = Path(args.out_dir)
    visual_dir = out_dir / "visuals_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visuals:
        visual_dir.mkdir(parents=True, exist_ok=True)

    # ── LPIPS (lazy) ──────────────────────────────────────────────────────
    lpips_mod = _import_lpips()
    lpips_fn = lpips_mod.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    # ── skimage metrics ───────────────────────────────────────────────────
    ssim_fn, psnr_fn = _import_skimage()

    # ── DataLoader (same UIEB test split, no model needed) ────────────────
    log.info("Building test DataLoader (data_root=%s)", args.data_root)
    test_loader = build_test_loader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    n_test = len(test_loader.dataset)
    log.info("Test set size: %d images", n_test)

    # ── Evaluation loop (no model, no torch.no_grad needed but kept safe) ──
    all_results = []
    visual_count = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            log.info(
                "Batch %d/%d  (%d images processed)",
                batch_idx + 1,
                len(test_loader),
                batch_idx * args.batch_size,
            )

            per_img = evaluate_batch_raw(batch, lpips_fn, ssim_fn, psnr_fn, device)

            for img_result in per_img:
                global_idx = len(all_results)
                metrics = {k: v for k, v in img_result.items() if not k.startswith("_")}
                metrics["idx"] = global_idx

                if args.save_visuals and (
                    args.max_visuals == 0 or visual_count < args.max_visuals
                ):
                    save_raw_grid(
                        img_result["_raw_01"],
                        img_result["_ref_01"],
                        metrics=metrics,
                        out_path=visual_dir,
                        idx=global_idx,
                    )
                    visual_count += 1

                all_results.append(metrics)

    elapsed = time.time() - t0
    log.info(
        "Raw baseline evaluation complete in %.1fs (%.2fs/image)",
        elapsed,
        elapsed / max(len(all_results), 1),
    )

    # ── Aggregate ─────────────────────────────────────────────────────────
    keys = ["psnr", "ssim", "lpips", "uciqe", "uiqm"]
    agg = {k: np.mean([r[k] for r in all_results]) for k in keys}
    agg_std = {k: np.std([r[k] for r in all_results]) for k in keys}

    # ── Print + save summary ─────────────────────────────────────────────
    summary_lines = [
        "",
        "═" * 55,
        "  RAW (UNENHANCED) BASELINE — UIEB Test Split",
        "═" * 55,
        f"  Test images : {len(all_results)}",
        "─" * 55,
        f"  PSNR   : {agg['psnr']:.4f} dB   ± {agg_std['psnr']:.4f}",
        f"  SSIM   : {agg['ssim']:.4f}      ± {agg_std['ssim']:.4f}",
        f"  LPIPS  : {agg['lpips']:.4f}      ± {agg_std['lpips']:.4f}",
        f"  UCIQE  : {agg['uciqe']:.4f}      ± {agg_std['uciqe']:.4f}",
        f"  UIQM   : {agg['uiqm']:.4f}      ± {agg_std['uiqm']:.4f}",
        "═" * 55,
        "",
    ]
    print("\n".join(summary_lines))

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    log.info("Summary saved -> %s", summary_path)

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx"] + keys)
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Per-image CSV -> %s", csv_path)

    json_path = out_dir / "aggregate.json"
    json_path.write_text(
        json.dumps(
            {k: {"mean": float(agg[k]), "std": float(agg_std[k])} for k in keys},
            indent=2,
        )
    )
    log.info("Aggregate JSON -> %s", json_path)

    if args.save_visuals:
        write_ranking_file(all_results, visual_dir)
        log.info(
            "Raw|Reference visual grids (%d/%d) -> %s",
            visual_count,
            len(all_results),
            visual_dir,
        )

    return agg


if __name__ == "__main__":
    main()
