#!/usr/bin/env python3
"""
thesis_eval.py
────────────────────────────────────────────────────────────────────────────
P-UWDM thesis-reporting evaluation script — companion to evaluate.py.

Reuses evaluate.py's tested model-loading / dataloader / metric-computation
code directly (imported, not reimplemented) — evaluate.py already has a
documented history of the imagenet_normalised bug reappearing on fresh
uploads, and its docstring explicitly says not to re-fix it again. Building
on top of the already-correct pipeline avoids re-introducing that bug here.

Two modes:

1. FULL (default) — for the thesis results section:
     - Full metrics (PSNR/SSIM/LPIPS/UCIQE/UIQM): mean, std, median
     - thesis_summary.txt   — human-readable summary
     - thesis_table.tex     — ready-to-paste LaTeX results table
     - metrics.csv          — per-image metrics (all 134 test images)
     - thesis_figures/      — top ~30-35 best-looking [Input|Enhanced|
                               Reference] comparison images (clean, no
                               burned-in text) + enhanced-only crops, both
                               at full resolution for the paper
     - thesis_figures/figure_index.csv — idx + all metrics + suggested
                               LaTeX caption text for each selected figure
     - thesis_figures/full_ranking.csv — every test image ranked by
                               composite visual-quality score, in case you
                               want to swap a pick by hand

2. QUICK (--quick) — for checking progress mid-training on whatever epoch
   checkpoint has been saved so far:
     - PSNR + SSIM only (skips LPIPS/UCIQE/UIQM and all visual saving —
       much faster, meant to be run repeatedly while training is ongoing)
     - Appends a row to a persistent progress log CSV
     - Regenerates a PSNR/SSIM-vs-epoch trend plot (if matplotlib available)

Usage:
    # Full thesis run (do this once training/fine-tuning is done)
    python thesis_eval.py --checkpoint checkpoints_v3_phase2fix/best.pt

    # Quick progress check while training is still running
    python thesis_eval.py --checkpoint checkpoints_v3_phase2fix/epoch_0060.pt --quick
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

# Reuse evaluate.py's tested pipeline directly.
from evaluate import (  # noqa: E402
    load_model,
    build_test_loader,
    evaluate_batch,
    to_uint8,
    _import_lpips,
    _import_skimage,
)

METRIC_KEYS = ["psnr", "ssim", "lpips", "uciqe", "uiqm"]
METRIC_DIRECTION = {  # for the LaTeX table arrow
    "psnr": r"$\uparrow$",
    "ssim": r"$\uparrow$",
    "lpips": r"$\downarrow$",
    "uciqe": r"$\uparrow$",
    "uiqm": r"$\uparrow$",
}


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint epoch label (cheap: filename first, full load only as fallback)
# ──────────────────────────────────────────────────────────────────────────


def get_epoch_label(checkpoint_path: str) -> Optional[int]:
    m = re.match(r"epoch_(\d+)", Path(checkpoint_path).stem)
    if m:
        return int(m.group(1))
    # e.g. best.pt — no epoch in the filename, fall back to reading the file.
    try:
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return ck.get("epoch")
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# Quick (PSNR/SSIM-only) evaluation — mirrors evaluate_batch's tensor
# handling exactly (no denorm, clamp(0,1) only) to avoid the historical
# denorm bug reappearing in a "fast path" that skips the main pipeline.
# ──────────────────────────────────────────────────────────────────────────


def quick_evaluate_batch(
    model, batch, device, num_steps, ssim_fn, psnr_fn
) -> List[Dict]:
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
            use_ema=False,
            progress=False,
        )
    enhanced_01 = enhanced_norm.clamp(0.0, 1.0)
    ref_01 = reference.clamp(0.0, 1.0)

    results = []
    for i in range(raw.shape[0]):
        enh_np = to_uint8(enhanced_01[i])
        ref_np = to_uint8(ref_01[i])
        results.append(
            {
                "psnr": psnr_fn(ref_np, enh_np, data_range=255),
                "ssim": ssim_fn(
                    ref_np, enh_np, data_range=255, channel_axis=2, win_size=7
                ),
            }
        )
    return results


# ──────────────────────────────────────────────────────────────────────────
# Thesis figure selection + saving
# ──────────────────────────────────────────────────────────────────────────


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-8 else np.zeros_like(x)


def rank_for_thesis(all_results: List[Dict]) -> List[Dict]:
    """
    Composite visual-quality ranking: weighted z-score combination of the
    three reference-based fidelity metrics (SSIM weighted highest since
    it's a named thesis target). This is a proxy for "looks close to the
    clean reference," not a substitute for eyeballing the picks — the full
    ranking CSV is saved alongside so any pick can be swapped by hand.
    """
    ssim = np.array([r["ssim"] for r in all_results])
    psnr = np.array([r["psnr"] for r in all_results])
    lpips = np.array([r["lpips"] for r in all_results])
    composite = 0.5 * _zscore(ssim) + 0.3 * _zscore(psnr) - 0.2 * _zscore(lpips)

    for r, c in zip(all_results, composite):
        r["composite_score"] = float(c)
    return sorted(all_results, key=lambda r: r["composite_score"], reverse=True)


def save_clean_figure(
    raw_01: torch.Tensor,
    enh_01: torch.Tensor,
    ref_01: torch.Tensor,
    out_dir: Path,
    rank: int,
    idx: int,
) -> None:
    """Publication-ready figures: no burned-in text, two variants per pick."""
    grid = make_grid(
        torch.stack([raw_01, enh_01, ref_01], dim=0), nrow=3, padding=6, pad_value=1.0
    )
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(grid_np).save(out_dir / f"fig{rank:02d}_idx{idx:04d}_compare.png")

    enh_np = to_uint8(enh_01)
    Image.fromarray(enh_np).save(out_dir / f"fig{rank:02d}_idx{idx:04d}_enhanced.png")


# ──────────────────────────────────────────────────────────────────────────
# Progress log + trend plot (for --quick mode, but harmless in full mode)
# ──────────────────────────────────────────────────────────────────────────


def append_progress_log(
    log_path: Path, checkpoint: str, epoch, mode: str, agg: Dict
) -> None:
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "checkpoint",
                    "epoch",
                    "mode",
                    "psnr",
                    "ssim",
                    "lpips",
                    "uciqe",
                    "uiqm",
                ]
            )
        writer.writerow(
            [
                datetime.datetime.now().isoformat(timespec="seconds"),
                checkpoint,
                epoch if epoch is not None else "",
                mode,
                f"{agg.get('psnr', float('nan')):.4f}",
                f"{agg.get('ssim', float('nan')):.4f}",
                f"{agg['lpips']:.4f}" if "lpips" in agg else "",
                f"{agg['uciqe']:.4f}" if "uciqe" in agg else "",
                f"{agg['uiqm']:.4f}" if "uiqm" in agg else "",
            ]
        )
    log.info("Appended progress row → %s", log_path)


def plot_progress(
    log_path: Path, out_png: Path, psnr_target: float, ssim_target: float
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning(
            "matplotlib not installed — skipping trend plot. `pip install matplotlib` to enable."
        )
        return

    rows = []
    with open(log_path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["epoch"]), float(r["psnr"]), float(r["ssim"])))
            except (ValueError, KeyError):
                continue  # skip rows with missing/non-numeric epoch (e.g. best.pt runs)

    if len(rows) < 2:
        log.info(
            "Fewer than 2 epoch-labeled rows in progress log — skipping trend plot for now."
        )
        return

    rows.sort(key=lambda r: r[0])
    epochs, psnrs, ssims = zip(*rows)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(epochs, psnrs, marker="o", color="#1f77b4")
    ax1.axhline(
        psnr_target, color="gray", linestyle="--", label=f"target {psnr_target}"
    )
    ax1.set_ylabel("PSNR (dB)")
    ax1.legend()
    ax2.plot(epochs, ssims, marker="o", color="#d62728")
    ax2.axhline(
        ssim_target, color="gray", linestyle="--", label=f"target {ssim_target}"
    )
    ax2.set_ylabel("SSIM")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    fig.suptitle("Evaluation metrics vs. training epoch")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("Progress trend plot → %s", out_png)


# ──────────────────────────────────────────────────────────────────────────
# LaTeX table
# ──────────────────────────────────────────────────────────────────────────


def write_latex_table(
    agg: Dict, agg_std: Dict, out_path: Path, psnr_target: float, ssim_target: float
) -> None:
    targets = {"psnr": f"> {psnr_target:g}", "ssim": f"> {ssim_target:g}"}
    rows = []
    for k in METRIC_KEYS:
        label = k.upper()
        rows.append(
            f"{label} {METRIC_DIRECTION[k]} & {agg[k]:.2f} & {agg_std[k]:.2f} & {targets.get(k, '--')} \\\\"
        )
    tex = (
        "% Auto-generated by thesis_eval.py — paste into your results section.\n"
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Quantitative evaluation results on the UIEB test split (134 images).}\n"
        "\\label{tab:results}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Metric & Mean & Std. Dev. & Target \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    out_path.write_text(tex)
    log.info("LaTeX table → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="P-UWDM thesis-reporting evaluation")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data_root", default="dataset/UIEB")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--out_dir", default=None)

    p.add_argument(
        "--quick",
        action="store_true",
        help="Fast progress-check mode: PSNR+SSIM only, no visuals, just "
        "appends to the progress log + refreshes the trend plot. Use this "
        "repeatedly during training on whatever checkpoint has been saved "
        "so far.",
    )

    p.add_argument(
        "--top_n",
        type=int,
        default=32,
        help="Number of thesis figures to select (30-35 range).",
    )
    p.add_argument("--psnr_target", type=float, default=22.0)
    p.add_argument("--ssim_target", type=float, default=0.85)

    p.add_argument("--progress_log", default="progress_log.csv")
    p.add_argument("--no_progress_log", action="store_true")
    p.add_argument("--no_plot", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "Device: %s  |  Mode: %s", device, "QUICK" if args.quick else "FULL (thesis)"
    )

    if args.out_dir is None:
        ckpt_path = Path(args.checkpoint)
        prefix = "quick_" if args.quick else "thesis_"
        out_dir = Path(f"{prefix}results_{ckpt_path.parent.name}_{ckpt_path.stem}")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    epoch_label = get_epoch_label(args.checkpoint)

    ssim_fn, psnr_fn = _import_skimage()
    test_loader = build_test_loader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    n_test = len(test_loader.dataset)
    log.info("Test set size: %d images (checkpoint epoch=%s)", n_test, epoch_label)

    all_results: List[Dict] = []

    if args.quick:
        # ── QUICK: PSNR + SSIM only ──────────────────────────────────────
        for batch_idx, batch in enumerate(test_loader):
            log.info("Batch %d/%d", batch_idx + 1, len(test_loader))
            for r in quick_evaluate_batch(
                model, batch, device, args.num_steps, ssim_fn, psnr_fn
            ):
                r["idx"] = len(all_results)
                all_results.append(r)

        agg = {k: float(np.mean([r[k] for r in all_results])) for k in ("psnr", "ssim")}
        agg_std = {
            k: float(np.std([r[k] for r in all_results])) for k in ("psnr", "ssim")
        }

        print(
            f"\nQUICK CHECK — {args.checkpoint} (epoch {epoch_label})\n"
            f"  PSNR: {agg['psnr']:.4f} ± {agg_std['psnr']:.4f} dB   [target > {args.psnr_target}]\n"
            f"  SSIM: {agg['ssim']:.4f} ± {agg_std['ssim']:.4f}      [target > {args.ssim_target}]\n"
        )

        csv_path = out_dir / "metrics_quick.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(all_results)
        log.info("Quick per-image CSV → %s", csv_path)

        agg_for_log = agg

    else:
        # ── FULL: reuse evaluate.py's evaluate_batch (all 5 metrics) ─────
        lpips_mod = _import_lpips()
        lpips_fn = lpips_mod.LPIPS(net="vgg").to(device)
        lpips_fn.eval()

        for batch_idx, batch in enumerate(test_loader):
            log.info("Batch %d/%d", batch_idx + 1, len(test_loader))
            per_img = evaluate_batch(
                model,
                batch,
                device,
                num_steps=args.num_steps,
                lpips_fn=lpips_fn,
                ssim_fn=ssim_fn,
                psnr_fn=psnr_fn,
            )
            for r in per_img:
                r["idx"] = len(all_results)
                all_results.append(r)

        agg = {k: float(np.mean([r[k] for r in all_results])) for k in METRIC_KEYS}
        agg_std = {k: float(np.std([r[k] for r in all_results])) for k in METRIC_KEYS}
        agg_median = {
            k: float(np.median([r[k] for r in all_results])) for k in METRIC_KEYS
        }

        summary_lines = [
            "",
            "═" * 60,
            "  P-UWDM THESIS EVALUATION — UIEB Test Split",
            "═" * 60,
            f"  Checkpoint  : {args.checkpoint}  (epoch {epoch_label})",
            f"  DDIM steps  : {args.num_steps}",
            f"  Test images : {len(all_results)}",
            "─" * 60,
        ]
        for k in METRIC_KEYS:
            target_str = ""
            if k == "psnr":
                target_str = f"  [target > {args.psnr_target}]"
            elif k == "ssim":
                target_str = f"  [target > {args.ssim_target}]"
            summary_lines.append(
                f"  {k.upper():6s}: mean={agg[k]:.4f}  std={agg_std[k]:.4f}  "
                f"median={agg_median[k]:.4f}{target_str}"
            )
        summary_lines += ["═" * 60, ""]
        print("\n".join(summary_lines))
        (out_dir / "thesis_summary.txt").write_text("\n".join(summary_lines))

        csv_path = out_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["idx"] + METRIC_KEYS)
            writer.writeheader()
            writer.writerows(
                [{k: r[k] for k in ["idx"] + METRIC_KEYS} for r in all_results]
            )
        log.info("Per-image CSV → %s", csv_path)

        write_latex_table(
            agg,
            agg_std,
            out_dir / "thesis_table.tex",
            args.psnr_target,
            args.ssim_target,
        )

        # ── Thesis figures: top-N best-looking, clean (no burned-in text) ─
        fig_dir = out_dir / "thesis_figures"
        fig_dir.mkdir(exist_ok=True)
        ranked = rank_for_thesis(all_results)

        with open(fig_dir / "full_ranking.csv", "w", newline="") as f:
            fieldnames = ["rank", "idx"] + METRIC_KEYS + ["composite_score"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, r in enumerate(ranked, start=1):
                writer.writerow(
                    {"rank": rank, **{k: r[k] for k in fieldnames if k != "rank"}}
                )

        index_rows = []
        for rank, r in enumerate(ranked[: args.top_n], start=1):
            save_clean_figure(
                r["_raw_01"], r["_enh_01"], r["_ref_01"], fig_dir, rank, r["idx"]
            )
            caption = (
                f"Enhancement result on UIEB test image {r['idx']} "
                f"(PSNR={r['psnr']:.2f}\\,dB, SSIM={r['ssim']:.3f})."
            )
            index_rows.append(
                {
                    "rank": rank,
                    "idx": r["idx"],
                    **{k: r[k] for k in METRIC_KEYS},
                    "composite_score": r["composite_score"],
                    "suggested_caption": caption,
                }
            )
        with open(fig_dir / "figure_index.csv", "w", newline="") as f:
            fieldnames = (
                ["rank", "idx"] + METRIC_KEYS + ["composite_score", "suggested_caption"]
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(index_rows)

        log.info(
            "Saved %d thesis figures (ranked by composite SSIM/PSNR/LPIPS score) → %s\n"
            "  Full ranking of all %d images is in full_ranking.csv if you want to swap a pick by hand.",
            len(index_rows),
            fig_dir,
            len(all_results),
        )

        agg_for_log = agg

    # ── Progress log + trend plot (both modes) ────────────────────────────
    if not args.no_progress_log:
        log_path = Path(args.progress_log)
        append_progress_log(
            log_path,
            args.checkpoint,
            epoch_label,
            "quick" if args.quick else "full",
            agg_for_log,
        )
        if not args.no_plot:
            plot_progress(
                log_path,
                log_path.with_suffix(".png"),
                args.psnr_target,
                args.ssim_target,
            )


if __name__ == "__main__":
    main()
