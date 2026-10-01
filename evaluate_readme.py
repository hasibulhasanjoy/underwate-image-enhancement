"""
evaluate_readme.py
────────────────────────────────────────────────────────────────────────────
P-UWDM  ->  GitHub README showcase generator
────────────────────────────────────────────────────────────────────────────

Runs the SAME inference + metric code as evaluate.py (it imports from it, so
any fix there automatically applies here) and produces README-ready assets:

  readme_assets/
  ├── results.md                 <- metrics table + copy-paste README snippet
  ├── metrics_summary.json       <- aggregate numbers + run info
  ├── metrics_per_image.csv      <- every test image, enhanced AND input metrics
  ├── showcase_best.png          <- Input | Ours | Reference, top-N images
  ├── showcase_random.png        <- same layout, RANDOM images (unselected)
  ├── metric_distributions.png   <- PSNR / SSIM / LPIPS histograms
  └── best/
      ├── best01_idx0012_comparison.png   <- clean side-by-side strip (no text)
      └── best01_idx0012/{input,enhanced,reference}.png

"Best" images are chosen by a composite score (z-scored PSNR + SSIM - LPIPS),
restricted to images where the model actually improved over the raw input.

Usage:
    python evaluate_readme.py
    python evaluate_readme.py --checkpoint checkpoints/best.pt --num_best 8
    python evaluate_readme.py --out_dir docs/assets --model_name "P-UWDM (ours)"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# Reuse everything from the existing pipeline (model loading, dataloader,
# batched DDIM inference, PSNR/SSIM/LPIPS/UCIQE/UIQM).
import evaluate as ev

log = ev.log

METRICS = ["psnr", "ssim", "lpips", "uciqe", "uiqm"]
LABELS = {
    "psnr": "PSNR (dB) ↑",
    "ssim": "SSIM ↑",
    "lpips": "LPIPS ↓",
    "uciqe": "UCIQE ↑",
    "uiqm": "UIQM ↑",
}
DECIMALS = {"psnr": 2, "ssim": 4, "lpips": 4, "uciqe": 3, "uiqm": 3}

Triplet = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (input, enhanced, reference)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics for an arbitrary prediction (used to score the raw INPUT as baseline)
# ──────────────────────────────────────────────────────────────────────────────


def metrics_for(pred_01, ref_01, lpips_fn, ssim_fn, psnr_fn, device) -> Dict[str, float]:
    """Same formulas/settings as evaluate.evaluate_batch, for one (C,H,W) pair."""
    pred_np = ev.to_uint8(pred_01)
    ref_np = ev.to_uint8(ref_01)
    with torch.no_grad():
        lp = lpips_fn(
            (pred_01[None] * 2 - 1).to(device), (ref_01[None] * 2 - 1).to(device)
        ).item()
    return {
        "psnr": float(psnr_fn(ref_np, pred_np, data_range=255)),
        "ssim": float(
            ssim_fn(ref_np, pred_np, data_range=255, channel_axis=2, win_size=7)
        ),
        "lpips": float(lp),
        "uciqe": float(ev.compute_uciqe(pred_np)),
        "uiqm": float(ev.compute_uiqm(pred_np)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Image selection
# ──────────────────────────────────────────────────────────────────────────────


def _z(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-8)


def score_records(records: List[Dict]) -> None:
    """Add a composite 'score' (higher = better) to every record, in place."""
    s = (
        _z([r["psnr"] for r in records])
        + _z([r["ssim"] for r in records])
        - _z([r["lpips"] for r in records])
    )
    for r, v in zip(records, s):
        r["score"] = float(v)


def select_best(records: List[Dict], k: int, require_improvement: bool = True) -> List[int]:
    pool = records
    if require_improvement:
        improved = [
            r
            for r in records
            if r["psnr"] > r["input_psnr"] and r["ssim"] > r["input_ssim"]
        ]
        if len(improved) >= k:
            pool = improved
        else:
            log.warning(
                "Only %d images improved on both PSNR and SSIM (< %d requested); "
                "falling back to all images.",
                len(improved),
                k,
            )
    return [r["idx"] for r in sorted(pool, key=lambda r: -r["score"])[:k]]


def select_random(records: List[Dict], k: int, exclude: Sequence[int], seed: int) -> List[int]:
    rng = random.Random(seed)
    candidates = [r["idx"] for r in records if r["idx"] not in set(exclude)]
    return sorted(rng.sample(candidates, min(k, len(candidates))))


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        log.error("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)


def make_showcase_figure(
    indices: Sequence[int],
    records: List[Dict],
    images: List[Triplet],
    path: Path,
    model_name: str,
    dpi: int,
) -> None:
    """Rows of  Input | Ours | Reference  with before/after metrics underneath."""
    plt = _plt()
    n = len(indices)
    col_titles = ["Input (degraded)", model_name, "Reference (GT)"]

    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.7 * n), squeeze=False)
    for row, idx in enumerate(indices):
        r = records[idx]
        for col, img in enumerate(images[idx]):
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=13, fontweight="bold")
        axes[row][0].set_xlabel(
            f"PSNR {r['input_psnr']:.2f} dB | SSIM {r['input_ssim']:.3f} | "
            f"LPIPS {r['input_lpips']:.3f}",
            fontsize=9,
            color="#555555",
        )
        axes[row][1].set_xlabel(
            f"PSNR {r['psnr']:.2f} dB | SSIM {r['ssim']:.3f} | LPIPS {r['lpips']:.3f}",
            fontsize=9,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_distribution_figure(
    records: List[Dict], path: Path, psnr_target: float, ssim_target: float, dpi: int
) -> None:
    plt = _plt()
    specs = [
        ("psnr", "PSNR (dB) ↑", psnr_target),
        ("ssim", "SSIM ↑", ssim_target),
        ("lpips", "LPIPS ↓", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (k, title, target) in zip(axes, specs):
        vals = [r[k] for r in records]
        ax.hist(vals, bins=20, color="#3b82c4", alpha=0.85, edgecolor="white")
        ax.axvline(np.mean(vals), color="#d62728", lw=2, label=f"mean {np.mean(vals):.3f}")
        ax.axvline(
            np.mean([r[f"input_{k}"] for r in records]),
            color="#777777",
            lw=2,
            ls="--",
            label=f"input mean {np.mean([r[f'input_{k}'] for r in records]):.3f}",
        )
        if target is not None:
            ax.axvline(target, color="#2ca02c", lw=2, ls=":", label=f"target {target}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("# test images")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_strip(triplet: Triplet, path: Path, gap: int = 6) -> None:
    """Clean text-free [Input | Enhanced | Reference] strip."""
    h, w, _ = triplet[0].shape
    canvas = Image.new("RGB", (3 * w + 2 * gap, h), "white")
    for i, arr in enumerate(triplet):
        canvas.paste(Image.fromarray(arr), (i * (w + gap), 0))
    canvas.save(path)


def save_individual(triplet: Triplet, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, arr in zip(("input", "enhanced", "reference"), triplet):
        Image.fromarray(arr).save(folder / f"{name}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────


def build_results_md(
    agg: Dict[str, Dict[str, float]],
    base: Dict[str, float],
    info: Dict,
    model_name: str,
    out_dir: Path,
    best_files: List[Tuple[int, Path]],
) -> str:
    rel = out_dir.as_posix()
    lines = [
        f"# {model_name} — Evaluation Results",
        "",
        "## Quantitative results",
        "",
        f"Evaluated on the **UIEB test split** ({info['num_images']} images, "
        f"{info['image_size']}x{info['image_size']}), DDIM {info['num_steps']} steps.",
        "",
        f"| Metric | Input (degraded) | {model_name} | Δ |",
        "|---|:---:|:---:|:---:|",
    ]
    for k in METRICS:
        d = DECIMALS[k]
        delta = agg[k]["mean"] - base[k]
        lines.append(
            f"| {LABELS[k]} | {base[k]:.{d}f} | "
            f"**{agg[k]['mean']:.{d}f}** ± {agg[k]['std']:.{d}f} | {delta:+.{d}f} |"
        )
    lines += [
        "",
        "> PSNR / SSIM / LPIPS are measured against the UIEB reference images. "
        "UCIQE / UIQM are no-reference metrics (the UCIQE here is not normalised "
        "to [0, 1]). Values are mean ± std over the test set.",
        "",
        "## Run details",
        "",
        "| | |",
        "|---|---|",
        f"| Checkpoint | `{info['checkpoint']}` |",
        f"| Parameters | {info['params_millions']:.2f} M |",
        f"| DDIM steps | {info['num_steps']} |",
        f"| Red-channel compensation | {'enabled' if info['rcc'] else 'disabled'} |",
        f"| Avg. time / image | {info['sec_per_image']:.2f} s (inference + metrics, {info['device']}) |",
        "",
        "## Copy-paste README snippet",
        "",
        "````markdown",
        "## Results",
        "",
        f"![Best results]({rel}/showcase_best.png)",
        "",
        f"| Metric | Input | {model_name} |",
        "|---|:---:|:---:|",
    ]
    for k in METRICS:
        d = DECIMALS[k]
        lines.append(f"| {LABELS[k]} | {base[k]:.{d}f} | **{agg[k]['mean']:.{d}f}** |")
    lines += [
        "",
        "### Unselected samples",
        f"![Random samples]({rel}/showcase_random.png)",
        "",
        "### Metric distributions",
        f"![Distributions]({rel}/metric_distributions.png)",
        "````",
        "",
        "## Individual best images",
        "",
    ]
    for rank, (idx, p) in enumerate(best_files, 1):
        lines.append(f"- #{rank} (idx {idx}): `{p.as_posix()}`")
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P-UWDM README asset generator")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data_root", default="dataset/UIEB")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--out_dir", default="readme_assets")
    p.add_argument("--model_name", default="P-UWDM (ours)")
    p.add_argument("--num_best", type=int, default=6, help="Images in showcase_best.png")
    p.add_argument("--num_random", type=int, default=4, help="Images in showcase_random.png")
    p.add_argument("--seed", type=int, default=42, help="Seed for the random showcase")
    p.add_argument(
        "--allow_regressions",
        action="store_true",
        help="Don't restrict 'best' to images that beat the raw input on PSNR and SSIM.",
    )
    p.add_argument("--psnr_target", type=float, default=22.0)
    p.add_argument("--ssim_target", type=float, default=0.85)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--red_channel_compensation", choices=["auto", "on", "off"], default="auto"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    out_dir = Path(args.out_dir)
    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    # ── Model / metrics / data (all reused from evaluate.py) ──────────────
    rcc = {"auto": None, "on": True, "off": False}[args.red_channel_compensation]
    model = ev.load_model(args.checkpoint, device, use_red_channel_compensation=rcc)
    lpips_fn = ev._import_lpips().LPIPS(net="vgg").to(device).eval()
    ssim_fn, psnr_fn = ev._import_skimage()
    loader = ev.build_test_loader(
        args.data_root, args.image_size, args.batch_size, args.num_workers
    )
    log.info("Test set size: %d images", len(loader.dataset))

    # ── Inference + metrics ───────────────────────────────────────────────
    records: List[Dict] = []
    images: List[Triplet] = []  # uint8 copies, kept small
    t0 = time.time()
    for b, batch in enumerate(loader):
        log.info("Batch %d/%d", b + 1, len(loader))
        per_img = ev.evaluate_batch(
            model,
            batch,
            device,
            num_steps=args.num_steps,
            lpips_fn=lpips_fn,
            ssim_fn=ssim_fn,
            psnr_fn=psnr_fn,
        )
        for r in per_img:
            rec = {"idx": len(records)}
            rec.update({k: float(r[k]) for k in METRICS})
            base = metrics_for(r["_raw_01"], r["_ref_01"], lpips_fn, ssim_fn, psnr_fn, device)
            rec.update({f"input_{k}": v for k, v in base.items()})
            records.append(rec)
            images.append(
                (
                    ev.to_uint8(r["_raw_01"]),
                    ev.to_uint8(r["_enh_01"]),
                    ev.to_uint8(r["_ref_01"]),
                )
            )
    elapsed = time.time() - t0
    n = len(records)
    log.info("Done: %d images in %.1fs (%.2fs/image)", n, elapsed, elapsed / max(n, 1))

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg = {
        k: {
            "mean": float(np.mean([r[k] for r in records])),
            "std": float(np.std([r[k] for r in records])),
        }
        for k in METRICS
    }
    base_mean = {k: float(np.mean([r[f"input_{k}"] for r in records])) for k in METRICS}

    info = {
        "checkpoint": args.checkpoint,
        "num_images": n,
        "image_size": args.image_size,
        "num_steps": args.num_steps,
        "rcc": model.red_comp is not None,
        "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        "sec_per_image": elapsed / max(n, 1),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    }

    # ── Select images ─────────────────────────────────────────────────────
    score_records(records)
    best_idx = select_best(records, args.num_best, not args.allow_regressions)
    rand_idx = select_random(records, args.num_random, best_idx, args.seed)
    log.info("Best images: %s", best_idx)
    log.info("Random images (seed=%d): %s", args.seed, rand_idx)

    # ── Figures ───────────────────────────────────────────────────────────
    make_showcase_figure(
        best_idx, records, images, out_dir / "showcase_best.png", args.model_name, args.dpi
    )
    if rand_idx:
        make_showcase_figure(
            rand_idx, records, images, out_dir / "showcase_random.png", args.model_name, args.dpi
        )
    make_distribution_figure(
        records, out_dir / "metric_distributions.png", args.psnr_target, args.ssim_target, args.dpi
    )

    best_files: List[Tuple[int, Path]] = []
    for rank, idx in enumerate(best_idx, 1):
        stem = f"best{rank:02d}_idx{idx:04d}"
        save_strip(images[idx], best_dir / f"{stem}_comparison.png")
        save_individual(images[idx], best_dir / stem)
        best_files.append((idx, best_dir / f"{stem}_comparison.png"))

    # ── Tables / JSON / CSV ───────────────────────────────────────────────
    (out_dir / "results.md").write_text(
        build_results_md(agg, base_mean, info, args.model_name, out_dir, best_files),
        encoding="utf-8",
    )
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                **info,
                "ours": agg,
                "input_baseline_mean": base_mean,
                "best_idx": best_idx,
                "random_idx": rand_idx,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fields = ["idx"] + METRICS + [f"input_{k}" for k in METRICS] + ["score"]
    with open(out_dir / "metrics_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    # ── Console summary (ASCII only, safe for Windows consoles) ───────────
    print()
    print("=" * 66)
    print(f"  {args.model_name} - UIEB test split ({n} images, {args.num_steps} DDIM steps)")
    print("=" * 66)
    print(f"  {'Metric':<8}{'Input':>12}{'Ours':>14}{'Std':>10}{'Delta':>10}")
    print("-" * 66)
    for k in METRICS:
        d = DECIMALS[k]
        print(
            f"  {k.upper():<8}{base_mean[k]:>12.{d}f}{agg[k]['mean']:>14.{d}f}"
            f"{agg[k]['std']:>10.{d}f}{agg[k]['mean'] - base_mean[k]:>+10.{d}f}"
        )
    print("=" * 66)
    print(f"  Assets written to: {out_dir.resolve()}")
    print(f"  Start with: {out_dir / 'results.md'}")
    print()


if __name__ == "__main__":
    main()
