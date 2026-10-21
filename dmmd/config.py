from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "seed": 42,
    "data": {
        "root": "./data", "download": False, "num_workers": 4,
        "validation_size": 5000, "random_flip": True,
        "synthetic": False, "synthetic_size": 128,
    },
    "model": {
        "base_channels": 64, "channel_mult": [1, 2, 2, 2],
        "time_dim": 128, "head_dim": 32, "dropout": 0.2,
        "head_spatial_size": 1,
    },
    "noise": {
        "timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02,
        "t_min": 0.001, "t_max": 0.999,
    },
    "train": {
        "steps": 20000, "batch_size": 64, "noise_levels_per_step": 1,
        "mmd_scope": "local",
        "lr": 0.0001, "betas": [0.9, 0.999],
        "lambda_l2": 0.1, "lambda_gp": 1.0,
        "max_grad_norm": None, "log_every": 50,
        "save_every": 1000, "keep_last": 3, "eval_every": 2000,
    },
    "sampling": {
        "t_max": 0.9, "t_min": 0.1, "noise_levels": 10,
        "steps_per_level": 10, "step_size": 0.1,
        "particles": 200, "num_samples": 1024,
        "reference_size": 1024, "feature_batch_size": 128,
        "pixel_clip": None,
    },
    "preview": {
        "particles": 16, "reference_size": 128,
        "noise_levels": 10, "steps_per_level": 5,
        "validation_batch_size": 64,
    },
    "runtime": {"device": "auto", "cpu_threads": 2, "expected_world_size": None},
}


def _merge(base, updates, path=""):
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {path}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected a mapping at {path}{key}")
            _merge(base[key], value, f"{path}{key}.")
        else:
            base[key] = value
    return base


def _read(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    seen.add(path)
    with path.open() as f:
        values = yaml.safe_load(f) or {}
    parent = values.pop("base", None)
    if parent:
        result = _read(path.parent / parent, seen)
        # Merge against the full schema to allow partial child mappings.
        return _merge(result, values)
    return _merge(deepcopy(DEFAULTS), values)


def load_config(path, overrides=()):
    config = _read(path)
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Overrides must look like train.steps=1000")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown configuration key: {key}")
            node = node[part]
        if parts[-1] not in node or isinstance(node[parts[-1]], dict):
            raise ValueError(f"Unknown or non-scalar override: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    validate(config)
    return config


def validate(c):
    positive = [
        ("train", "steps"), ("train", "batch_size"),
        ("train", "noise_levels_per_step"), ("train", "lr"),
        ("train", "log_every"), ("train", "save_every"),
        ("train", "keep_last"), ("model", "base_channels"),
        ("model", "head_dim"), ("noise", "timesteps"),
        ("sampling", "particles"), ("sampling", "num_samples"),
        ("sampling", "noise_levels"), ("sampling", "steps_per_level"),
        ("sampling", "feature_batch_size"), ("sampling", "step_size"),
        ("preview", "particles"), ("preview", "reference_size"),
        ("preview", "noise_levels"), ("preview", "steps_per_level"),
        ("preview", "validation_batch_size"), ("runtime", "cpu_threads"),
    ]
    for section, key in positive:
        if c[section][key] <= 0:
            raise ValueError(f"{section}.{key} must be positive")
    if c["train"]["batch_size"] < 2:
        raise ValueError("The off-diagonal MMD estimator requires batch_size >= 2")
    if c["train"]["mmd_scope"] not in ("local", "global"):
        raise ValueError("train.mmd_scope must be local or global")
    expected = c["runtime"]["expected_world_size"]
    if expected is not None and (not isinstance(expected, int) or expected < 1):
        raise ValueError("runtime.expected_world_size must be a positive integer or null")
    if c["train"]["eval_every"] < 0 or c["data"]["num_workers"] < 0:
        raise ValueError("eval_every and num_workers cannot be negative")
    if c["data"]["validation_size"] < 0 or c["sampling"]["reference_size"] < 0:
        raise ValueError("validation_size and reference_size cannot be negative")
    if c["train"]["lambda_l2"] < 0 or c["train"]["lambda_gp"] < 0:
        raise ValueError("Regularization weights cannot be negative")
    for section in ("noise", "sampling"):
        if not 0 <= c[section]["t_min"] < c[section]["t_max"] <= 1:
            raise ValueError(f"Require 0 <= {section}.t_min < t_max <= 1")
    if not (0 < c["noise"]["beta_start"] <= c["noise"]["beta_end"] < 1):
        raise ValueError("Require 0 < beta_start <= beta_end < 1")
    if not 0 <= c["model"]["dropout"] < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if c["model"]["time_dim"] < 4 or c["model"]["time_dim"] % 2:
        raise ValueError("model.time_dim must be even and at least 4")
    mult = c["model"]["channel_mult"]
    if not isinstance(mult, list) or not 2 <= len(mult) <= 4:
        raise ValueError("Use two to four U-Net scales for 32x32 images")
    if any(not isinstance(m, int) or m <= 0 for m in mult):
        raise ValueError("channel_mult entries must be positive integers")
    # Old checkpoint configs predate this option and use the global head.
    grid = c["model"].get("head_spatial_size", 1)
    smallest_map = 32 // (2 ** (len(mult) - 1))
    if (isinstance(grid, bool) or not isinstance(grid, int)
            or not 1 <= grid <= smallest_map):
        raise ValueError(f"model.head_spatial_size must be an integer in [1, {smallest_map}]")
    if len(c["train"]["betas"]) != 2 or any(
        not 0 <= b < 1 for b in c["train"]["betas"]
    ):
        raise ValueError("train.betas must contain two values in [0,1)")
    for value in (c["sampling"]["pixel_clip"], c["train"]["max_grad_norm"]):
        if value is not None and value <= 0:
            raise ValueError("Optional clipping thresholds must be positive")


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path
