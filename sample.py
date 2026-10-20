#!/usr/bin/env python3
"""Generate pixel images using an interacting or approximate independent flow."""
import argparse
import json
import math
import time

import torch
from torchvision.utils import save_image

from dmmd.config import project_path
from dmmd.model import FeatureUNet
from dmmd.runtime import file_sha256, frozen, load_checkpoint, save_grid, seed_all, setup
from dmmd.sampling import sample_particles


def validate_stats(payload, stats, checkpoint):
    if stats.get("kind") != "dmmd_statistics":
        raise ValueError("Expected a file produced by cache_stats.py")
    if stats["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("Statistics belong to a different checkpoint; regenerate them")
    if stats["model_config"] != payload["config"]["model"]:
        raise ValueError("Statistics/model architecture mismatch")
    if stats["noise_config"] != payload["config"]["noise"]:
        raise ValueError("Statistics/noise schedule mismatch")
    dimension = 4 * payload["config"]["model"]["head_dim"]
    expected = (len(stats["times"]), dimension)
    if stats["mean_clean"].shape != expected or stats["mean_noisy"].shape != expected:
        raise ValueError("Invalid cached feature-mean shapes")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("interacting", "independent"), default="interacting")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--steps-per-level", type=int)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-tensors", action="store_true")
    args = parser.parse_args()
    checkpoint = project_path(args.checkpoint)
    payload = load_checkpoint(checkpoint)
    if payload.get("kind") != "dmmd_features":
        raise ValueError("Expected a DMMD feature checkpoint")
    stats = load_checkpoint(project_path(args.stats))
    validate_stats(payload, stats, checkpoint)
    config = payload["config"]
    config["runtime"]["device"] = args.device
    device, _, _ = setup(config)
    settings = dict(stats["sampling_config"])
    for name in ("num_samples", "particles", "steps_per_level", "step_size"):
        value = getattr(args, name)
        if value is not None:
            if value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
            settings[name] = value
    output = project_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory; {output} is not empty")
    image_dir = output / "images"
    image_dir.mkdir(parents=True)
    model = FeatureUNet(**config["model"]).to(device)
    model.load_state_dict(payload["model"])
    seed_all(args.seed)
    times = stats["times"].to(device)
    clean_means = stats["mean_clean"].to(device)
    noisy_means = stats["mean_noisy"].to(device) if args.mode == "independent" else None
    groups = math.ceil(settings["num_samples"] / settings["particles"])
    saved, examples, trace = 0, [], []
    started = time.perf_counter()
    with frozen(model):
        for group in range(groups):
            # Full particle groups avoid changing the final group's interaction law.
            initial = torch.randn(settings["particles"], 3, 32, 32, device=device)

            def record(level, inner, t, particles, gradient):
                if group == 0 and inner == settings["steps_per_level"] - 1:
                    trace.append({
                        "level": level, "t": t.item(),
                        "pixel_rms": particles.square().mean().sqrt().item(),
                        "gradient_l2": gradient.flatten(1).norm(dim=1).mean().item(),
                    })

            final = sample_particles(
                model, initial, times, clean_means, settings["steps_per_level"],
                settings["step_size"], noisy_means, settings["pixel_clip"], record,
            ).cpu()
            take = min(len(final), settings["num_samples"] - saved)
            final = final[:take]
            if len(examples) < 64:
                examples.extend(list(final[:64 - len(examples)]))
            for image in final:
                save_image(image.clamp(-1, 1).add(1).div(2),
                           image_dir / f"{saved:06d}.png")
                saved += 1
            if args.save_tensors:
                torch.save(final, output / f"particles_{group:05d}.pt")
            print(f"Saved {saved}/{settings['num_samples']} images", flush=True)
    save_grid(torch.stack(examples), output / "grid.png")
    metadata = {
        "checkpoint": str(checkpoint), "checkpoint_step": payload["step"],
        "checkpoint_sha256": stats["checkpoint_sha256"],
        "mode": args.mode, "num_samples": saved, "seed": args.seed,
        "reference_size": stats["reference_size"],
        "synthetic": stats["synthetic"],
        "sampling": settings, "times": times.cpu().tolist(),
        "pixel_updates_per_sample": len(times) * settings["steps_per_level"],
        "seconds": time.perf_counter() - started, "first_group_trace": trace,
        "png_export": "clamp raw pixels to [-1,1], then map to [0,1]",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Image folder: {image_dir}")


if __name__ == "__main__":
    run()
