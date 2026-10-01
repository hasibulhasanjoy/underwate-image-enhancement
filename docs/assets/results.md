# P-UWDM (ours) — Evaluation Results

## Quantitative results

Evaluated on the **UIEB test split** (134 images, 256x256), DDIM 50 steps.

| Metric | Input (degraded) | P-UWDM (ours) | Δ |
|---|:---:|:---:|:---:|
| PSNR (dB) ↑ | 16.99 | **18.30** ± 3.65 | +1.31 |
| SSIM ↑ | 0.7575 | **0.7952** ± 0.1237 | +0.0378 |
| LPIPS ↓ | 0.2168 | **0.2445** ± 0.1265 | +0.0276 |
| UCIQE ↑ | 21.473 | **23.591** ± 3.608 | +2.118 |
| UIQM ↑ | 5.351 | **7.979** ± 2.533 | +2.628 |

> PSNR / SSIM / LPIPS are measured against the UIEB reference images. UCIQE / UIQM are no-reference metrics (the UCIQE here is not normalised to [0, 1]). Values are mean ± std over the test set.

## Run details

| | |
|---|---|
| Checkpoint | `checkpoints_v7_rcc_only/best.pt` |
| Parameters | 50.70 M |
| DDIM steps | 50 |
| Red-channel compensation | enabled |
| Avg. time / image | 3.25 s (inference + metrics, CPU) |

## Copy-paste README snippet

````markdown
## Results

![Best results](docs/assets/showcase_best.png)

| Metric | Input | P-UWDM (ours) |
|---|:---:|:---:|
| PSNR (dB) ↑ | 16.99 | **18.30** |
| SSIM ↑ | 0.7575 | **0.7952** |
| LPIPS ↓ | 0.2168 | **0.2445** |
| UCIQE ↑ | 21.473 | **23.591** |
| UIQM ↑ | 5.351 | **7.979** |

### Unselected samples
![Random samples](docs/assets/showcase_random.png)

### Metric distributions
![Distributions](docs/assets/metric_distributions.png)
````

## Individual best images

- #1 (idx 74): `docs/assets/best/best01_idx0074_comparison.png`
- #2 (idx 95): `docs/assets/best/best02_idx0095_comparison.png`
- #3 (idx 93): `docs/assets/best/best03_idx0093_comparison.png`
- #4 (idx 114): `docs/assets/best/best04_idx0114_comparison.png`
- #5 (idx 30): `docs/assets/best/best05_idx0030_comparison.png`
- #6 (idx 82): `docs/assets/best/best06_idx0082_comparison.png`
- #7 (idx 9): `docs/assets/best/best07_idx0009_comparison.png`
- #8 (idx 71): `docs/assets/best/best08_idx0071_comparison.png`
