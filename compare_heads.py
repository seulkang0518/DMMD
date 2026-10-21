#!/usr/bin/env python3
"""Matched screening FID for the saved global and spatial-head checkpoints.

Run from this repository after pausing training. No training or checkpoint
updates are performed. Complete validated sample folders can be reused.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from dmmd.config import ROOT, project_path
from dmmd.runtime import load_checkpoint
from sample import validate_stats


def inspect_checkpoint(checkpoint, stats_path, grid, step):
    for path in (checkpoint, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; supply the corresponding path explicitly")
    payload, stats = load_checkpoint(checkpoint), load_checkpoint(stats_path)
    if payload.get("kind") != "dmmd_features":
        raise ValueError(f"Not a feature checkpoint: {checkpoint}")
    validate_stats(payload, stats, checkpoint)
    config = payload["config"]
    if payload["step"] != step:
        raise ValueError(f"Expected step {step}, found {payload['step']} in {checkpoint}")
    if config["model"].get("head_spatial_size", 1) != grid:
        raise ValueError(f"Expected head grid {grid} in {checkpoint}")
    if payload.get("world_size", 1) != 1:
        raise ValueError("This comparison expects the one-GPU training runs")
    if config["data"]["synthetic"] or stats["synthetic"]:
        raise ValueError("Synthetic checkpoints/statistics cannot be compared as CIFAR-10")
    if stats["reference_size"] != 50000:
        raise ValueError("Both feature caches must use all 50,000 training references")
    expected_times = torch.linspace(.9, .1, 10)
    if stats["times"].shape != expected_times.shape or not torch.allclose(
            stats["times"].cpu(), expected_times, rtol=0, atol=1e-7):
        raise ValueError("Both feature caches must use 10 times from 0.9 to 0.1")
    if stats["sampling_config"]["pixel_clip"] is not None:
        raise ValueError("This comparison expects no raw-pixel clipping during sampling")
    model_config = dict(config["model"])
    model_config.pop("head_spatial_size", None)
    recipe = {
        "model_except_head": model_config, "noise": config["noise"],
        "seed": config["seed"],
        "data": {k: config["data"][k] for k in ("validation_size", "random_flip")},
        "train": {k: config["train"][k] for k in (
            "batch_size", "noise_levels_per_step", "mmd_scope", "lr", "betas",
            "lambda_l2", "lambda_gp", "max_grad_norm")},
    }
    return {
        "checkpoint": str(checkpoint), "stats": str(stats_path), "grid": grid,
        "checkpoint_step": step, "checkpoint_sha256": stats["checkpoint_sha256"],
        "times": stats["times"].cpu().tolist(), "recipe": recipe,
    }


def validate_samples(directory, info, count, seed):
    """Reuse only complete samples with the requested checkpoint and protocol."""
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Incomplete output at {directory}; use a new --output-root")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "checkpoint_sha256": info["checkpoint_sha256"],
        "checkpoint_step": info["checkpoint_step"], "num_samples": count,
        "seed": seed, "mode": "interacting", "reference_size": 50000,
        "synthetic": False, "times": info["times"], "pixel_updates_per_sample": 100,
        "png_export": "clamp raw pixels to [-1,1], then map to [0,1]",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Sample protocol mismatch: {directory}: {key}")
    settings = metadata["sampling"]
    for key, value in {"particles": 200, "steps_per_level": 10,
                       "step_size": .1, "pixel_clip": None}.items():
        if settings.get(key) != value:
            raise ValueError(f"Sample protocol mismatch: {directory}: sampling.{key}")
    names = {path.name for path in (directory / "images").glob("*.png")}
    if names != {f"{i:06d}.png" for i in range(count)}:
        raise ValueError(f"Missing or unexpected PNGs in {directory / 'images'}")


def read_fid(directory, count):
    result = json.loads((directory / "fid.json").read_text())
    if result["generated_samples"] != count:
        raise ValueError("FID sample count does not match the requested comparison")
    if result["reference"] != "all 50000 unaugmented CIFAR-10 training images":
        raise ValueError("Unexpected FID reference")
    if result["protocol"] != "uint8 PNG -> torch-fidelity default Inception features":
        raise ValueError("Unexpected FID preprocessing")
    fid = float(result["metrics"]["frechet_inception_distance"])
    if not math.isfinite(fid):
        raise ValueError("FID is non-finite")
    return fid


def first_existing(paths):
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("Checkpoint not found. Checked: " + ", ".join(map(str, paths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=4000)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--spatial-checkpoint")
    parser.add_argument("--baseline-stats")
    parser.add_argument("--spatial-stats")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-root")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=100042)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--fid-batch-size", type=int, default=128)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.step < 1 or args.num_samples < 2 or args.fid_batch_size < 1:
        parser.error("Use a positive step/batch size and at least two samples")
    step = args.step
    baseline = project_path(args.baseline_checkpoint) if args.baseline_checkpoint else first_existing([
        ROOT / f"runs/eval_{step}/checkpoint.pt",
        ROOT / f"runs/cifar10_4090_full/checkpoints/step_{step:08d}.pt",
    ])
    spatial = project_path(args.spatial_checkpoint or
                           f"runs/cifar10_spatial4_10k/checkpoints/step_{step:08d}.pt")
    pair = {
        "global1": inspect_checkpoint(baseline, project_path(args.baseline_stats or
            f"runs/eval_{step}/stats.pt"), 1, step),
        "spatial4": inspect_checkpoint(spatial, project_path(args.spatial_stats or
            f"runs/eval_spatial4_{step}/stats.pt"), 4, step),
    }
    if pair["global1"]["recipe"] != pair["spatial4"]["recipe"]:
        raise ValueError("Training recipes differ beyond the feature heads and run controls")
    if pair["global1"]["checkpoint_sha256"] == pair["spatial4"]["checkpoint_sha256"]:
        raise ValueError("Both paths point to the same checkpoint")
    output_root = project_path(args.output_root or f"runs/compare_heads_{step}_{args.num_samples}")
    print(json.dumps({"stage": "preflight_passed", "checkpoints": pair,
                      "samples_per_model": args.num_samples, "sampling_seed": args.seed,
                      "output_root": str(output_root)}, indent=2), flush=True)
    if args.validate_only:
        return
    if importlib.util.find_spec("torch_fidelity") is None:
        raise SystemExit("Install metrics first: python -m pip install -r requirements-fid.txt")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")
    # Check both directories before starting any sampling. Never delete or
    # overwrite partial samples; a fresh output root is required for those.
    reuse = {}
    for label, info in pair.items():
        directory = output_root / label
        reuse[label] = directory.exists() and any(directory.iterdir())
        if reuse[label]:
            validate_samples(directory, info, args.num_samples, args.seed)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, info in pair.items():
        directory = output_root / label
        if not reuse[label]:
            command = [sys.executable, "-u", str(ROOT / "sample.py"),
                       "--checkpoint", info["checkpoint"], "--stats", info["stats"],
                       "--output", str(directory), "--device", args.device,
                       "--mode", "interacting", "--num-samples", str(args.num_samples),
                       "--particles", "200", "--steps-per-level", "10",
                       "--step-size", "0.1", "--seed", str(args.seed)]
            print(f"Generating {label}: {args.num_samples} samples", flush=True)
            subprocess.run(command, check=True, cwd=ROOT)
        else:
            print(f"Reusing complete, protocol-checked samples: {directory}", flush=True)
        validate_samples(directory, info, args.num_samples, args.seed)
        # Recompute metrics even when samples exist, so a stale fid.json cannot
        # silently be used for changed PNGs or an earlier evaluation run.
        command = [sys.executable, "-u", str(ROOT / "evaluate_fid.py"),
                   "--samples", str(directory), "--data-root", str(project_path(args.data_root)),
                   "--batch-size", str(args.fid_batch_size)]
        if args.device == "cpu":
            command.append("--cpu")
        subprocess.run(command, check=True, cwd=ROOT)
        rows.append({"model": label, "head_grid": info["grid"],
                     "checkpoint_step": step, "checkpoint_sha256": info["checkpoint_sha256"],
                     "samples": args.num_samples, "fid": read_fid(directory, args.num_samples),
                     "sample_directory": str(directory)})
    summary = {
        "purpose": "Matched small-sample screening; not final 50K FID",
        "reference": "all 50000 unaugmented CIFAR-10 training images",
        "seed": args.seed, "particles": 200, "pixel_updates_per_sample": 100,
        "step_size": .1, "device": args.device, "results": rows,
        "spatial_minus_global_fid": rows[1]["fid"] - rows[0]["fid"],
    }
    summary_path = output_root / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\nModel       Step   Samples       FID (screening)", flush=True)
    for row in rows:
        print(f"{row['model']:<10} {step:>6} {args.num_samples:>8} {row['fid']:>18.4f}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
