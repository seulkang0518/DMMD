#!/usr/bin/env python3
"""Cache clean/noisy feature means for a particular frozen checkpoint and grid."""
import argparse
import copy
import json

import torch
from torch.utils.data import DataLoader, Subset

from dmmd.config import project_path, validate
from dmmd.data import datasets
from dmmd.model import FeatureUNet
from dmmd.noise import DDPMNoise
from dmmd.runtime import atomic_save, file_sha256, frozen, load_checkpoint, seed_all, setup
from dmmd.sampling import noise_grid


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference-size", type=int)
    parser.add_argument("--noise-levels", type=int)
    parser.add_argument("--t-min", type=float)
    parser.add_argument("--t-max", type=float)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    checkpoint = project_path(args.checkpoint)
    payload = load_checkpoint(checkpoint)
    if payload.get("kind") != "dmmd_features":
        raise ValueError("Expected a DMMD feature checkpoint")
    config = copy.deepcopy(payload["config"])
    config["runtime"]["device"] = args.device
    config["data"]["download"] = False
    if args.data_root:
        config["data"]["root"] = args.data_root
    for name in ("reference_size", "noise_levels", "t_min", "t_max"):
        value = getattr(args, name)
        if value is not None:
            config["sampling"][name] = value
    validate(config)
    device, _, _ = setup(config)
    seed_all(args.seed)
    model = FeatureUNet(**config["model"]).to(device)
    model.load_state_dict(payload["model"])
    noise = DDPMNoise(**config["noise"]).to(device)
    _, _, reference = datasets(config, augment=False)
    count = config["sampling"]["reference_size"]
    if count:
        reference = Subset(reference, range(min(count, len(reference))))
    loader = DataLoader(
        reference, batch_size=config["sampling"]["feature_batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    times = noise_grid(config["sampling"]).to(device)
    clean_means, noisy_means = [], []
    # Reset after model initialization for reproducible corruption statistics.
    seed_all(args.seed)
    with frozen(model), torch.no_grad():
        for t in times:
            sum_clean = torch.zeros(model.output_dim, device=device, dtype=torch.float64)
            sum_noisy = torch.zeros_like(sum_clean)
            seen = 0
            for clean, _ in loader:
                clean = clean.to(device, non_blocking=True)
                sum_clean += model(clean, t).double().sum(dim=0)
                sum_noisy += model(noise(clean, t), t).double().sum(dim=0)
                seen += len(clean)
            clean_means.append((sum_clean / seen).float().cpu())
            noisy_means.append((sum_noisy / seen).float().cpu())
            print(f"Cached t={t.item():.4f}: {seen} real images", flush=True)
    output = project_path(args.output)
    atomic_save({
        "format_version": 1, "kind": "dmmd_statistics",
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": payload["step"], "model_config": config["model"],
        "noise_config": config["noise"], "times": times.cpu(),
        "mean_clean": torch.stack(clean_means), "mean_noisy": torch.stack(noisy_means),
        "reference_size": len(reference), "seed": args.seed,
        "synthetic": config["data"]["synthetic"],
        "sampling_config": config["sampling"],
    }, output)
    print(json.dumps({"output": str(output), "reference_size": len(reference),
                      "noise_levels": len(times)}, indent=2))


if __name__ == "__main__":
    run()
