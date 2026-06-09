"""
src/utils/config.py
-------------------
Typed configuration management using dataclasses + PyYAML.

Responsibilities:
  - Load and validate YAML config files.
  - Expose strongly-typed dataclasses so the rest of the codebase
    gets IDE auto-complete and static-analysis benefits.
  - Support nested overrides (e.g. from CLI or sweep tools).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Leaf config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    name: str = "UIEB"
    root_dir: str = "dataset/UIEB"
    raw_subdir: str = "raw"
    reference_subdir: str = "reference"
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    )


@dataclass
class SplitsConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42
    stratified: bool = False

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


@dataclass
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class NormalizeConfig:
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class PreprocessingConfig:
    image_size: Tuple[int, int] = (256, 256)  # (H, W)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)


@dataclass
class ColorJitterConfig:
    enabled: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05


@dataclass
class RandomRotationConfig:
    enabled: bool = True
    degrees: float = 15.0


@dataclass
class RandomCropConfig:
    enabled: bool = False
    size: Tuple[int, int] = (224, 224)


@dataclass
class GaussianBlurConfig:
    enabled: bool = False
    kernel_size: int = 3
    sigma: Tuple[float, float] = (0.1, 2.0)


@dataclass
class AugmentationSplitConfig:
    random_horizontal_flip: bool = True
    random_vertical_flip: bool = False
    random_rotation: RandomRotationConfig = field(default_factory=RandomRotationConfig)
    color_jitter: ColorJitterConfig = field(default_factory=ColorJitterConfig)
    random_crop: RandomCropConfig = field(default_factory=RandomCropConfig)
    gaussian_blur: GaussianBlurConfig = field(default_factory=GaussianBlurConfig)


@dataclass
class AugmentationConfig:
    train: AugmentationSplitConfig = field(default_factory=AugmentationSplitConfig)
    # val and test are deterministic — no augmentation fields needed


@dataclass
class CacheConfig:
    enabled: bool = False
    max_size_gb: float = 4.0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_dataclass(cls, data: dict):
    """Recursively instantiate a dataclass from a plain dict."""
    import dataclasses
    import sys

    if not dataclasses.is_dataclass(cls):
        return data  # not a dataclass — return raw value

    # Build a name→class registry from every dataclass defined in this module
    # so we can resolve string annotations like "DatasetConfig" reliably.
    _module = sys.modules[cls.__module__]
    _registry = {
        name: obj
        for name, obj in vars(_module).items()
        if dataclasses.is_dataclass(obj)
    }

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]

        # f.type is always a string when `from __future__ import annotations` is active
        f_type = f.type
        if isinstance(f_type, str):
            f_type = _registry.get(f_type, None)

        if (
            f_type is not None
            and dataclasses.is_dataclass(f_type)
            and isinstance(value, dict)
        ):
            value = _dict_to_dataclass(f_type, value)

        kwargs[f.name] = value

    return cls(**kwargs)


def load_data_config(
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> DataConfig:
    """
    Load a DataConfig from *config_path* (YAML).

    Parameters
    ----------
    config_path:
        Path to the YAML file.  If None the function returns the
        default DataConfig built from dataclass defaults.
    overrides:
        Nested dict of values to overlay on top of the loaded YAML
        before constructing the dataclass.  Useful for programmatic
        overrides (e.g. from CLI argument parsing or hyperparameter
        sweeps).

    Returns
    -------
    DataConfig
    """
    raw: dict = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path.resolve()}")
        with path.open("r") as fh:
            raw = yaml.safe_load(fh) or {}

    if overrides:
        raw = _deep_update(raw, overrides)

    return _dict_to_dataclass(DataConfig, raw)
