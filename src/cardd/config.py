"""Experiment configuration schema.

Collapses what used to be four near-duplicate notebooks (differing only in hyperparameters and an
optional Albumentations transform list) into one canonical code path (`train.py::run_training`)
driven by a YAML file per experiment.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    variant: str = "yolo11n"
    # False loads f"{variant}.yaml" (random-init architecture, no network fetch) instead of the
    # pretrained f"{variant}.pt" checkpoint - used by the CI smoke config so tests never depend on
    # Ultralytics' asset servers being reachable.
    pretrained: bool = True


@dataclass
class TrainConfig:
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    seed: int = 0
    deterministic: bool = True
    patience: int = 20
    device: Any = 0
    cache: str = "ram"
    optimizer: str = "auto"
    lr0: float = 0.01
    weight_decay: float = 0.0005
    freeze: int | None = None
    cls: float = 0.5


@dataclass
class AlbumentationSpec:
    name: str
    params: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    data_yaml: Path = Path("configs/cardd_yolo.yaml")
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # Raw passthrough for augmentation hyperparameters Ultralytics accepts directly as train()
    # kwargs (hsv_h, hsv_v, degrees, perspective, ...) - not worth a dedicated dataclass field
    # per parameter since the set Ultralytics supports is large and rarely all used at once.
    augment: dict = field(default_factory=dict)
    albumentations: list[AlbumentationSpec] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # final escape hatch, highest precedence
    mlflow_experiment: str = "cardd-yolo11"
    mlflow_tags: dict = field(default_factory=dict)
    source_path: Path | None = None  # set by load_experiment_config, not user-specified


def _apply_override(data: dict, dotted_key: str, value: str) -> None:
    """Apply a `--set key.subkey=value` override in place, parsing `value` as YAML so ints/
    floats/bools/lists round-trip correctly instead of always landing as strings."""
    parsed = yaml.safe_load(value)
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = parsed


def load_experiment_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    for override in overrides or []:
        key, _, value = override.partition("=")
        if not _:
            raise ValueError(f"--set override must be key=value, got: {override!r}")
        _apply_override(raw, key.strip(), value.strip())

    model = ModelConfig(**raw.get("model", {}))
    train = TrainConfig(**raw.get("train", {}))
    albumentations = [AlbumentationSpec(**spec) for spec in raw.get("albumentations", [])]

    cfg = ExperimentConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        data_yaml=Path(raw.get("data_yaml", "configs/cardd_yolo.yaml")),
        model=model,
        train=train,
        augment=raw.get("augment", {}),
        albumentations=albumentations,
        extra=raw.get("extra", {}),
        mlflow_experiment=raw.get("mlflow_experiment", "cardd-yolo11"),
        mlflow_tags=raw.get("mlflow_tags", {}),
        source_path=Path(path),
    )
    return cfg


def build_train_kwargs(cfg: ExperimentConfig) -> dict:
    """Merge order (lowest -> highest precedence): TrainConfig fields, cfg.augment, cfg.extra,
    then attach `augmentations=[...]` only if cfg.albumentations is non-empty."""
    kwargs = dataclasses.asdict(cfg.train)
    kwargs.update(copy.deepcopy(cfg.augment))
    kwargs.update(copy.deepcopy(cfg.extra))
    if cfg.albumentations:
        kwargs["augmentations"] = build_albumentations(cfg.albumentations)
    return kwargs


def build_albumentations(specs: list[AlbumentationSpec]) -> list:
    import albumentations as A

    transforms = []
    for spec in specs:
        try:
            transform_cls = getattr(A, spec.name)
        except AttributeError as e:
            raise AttributeError(
                f"'{spec.name}' is not a valid albumentations transform name (typo?)"
            ) from e
        transforms.append(transform_cls(**spec.params))
    return transforms


def flatten_config_for_mlflow(cfg: ExperimentConfig, prefix: str = "") -> dict:
    """dataclasses.asdict + dot-flatten nested keys, for mlflow.log_params (which wants a flat
    str->str/number mapping, not nested dicts)."""
    flat: dict = {}

    def _flatten(obj, key_prefix):
        if dataclasses.is_dataclass(obj):
            obj = dataclasses.asdict(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{key_prefix}{k}.")
        elif isinstance(obj, (list, tuple)):
            flat[key_prefix.rstrip(".")] = str(obj)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            flat[key_prefix.rstrip(".")] = obj
        else:
            # e.g. Path - mlflow.log_params needs primitives
            flat[key_prefix.rstrip(".")] = str(obj)

    _flatten(dataclasses.asdict(cfg), prefix)
    flat.pop("source_path", None)
    return flat
