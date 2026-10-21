#!/usr/bin/env python3
"""Read-only probes of a DMMD checkpoint; run from the experiment repository.

This is a representation/field diagnostic, not a generation-quality metric.
No optimizer, checkpoint updates, or training-data augmentation are used.
"""
import argparse
import json
from pathlib import Path
import sys

import torch


def feature_summary(clean, noisy):
    """Same-size, off-diagonal linear MMD and its energy decomposition."""
    a, b = clean.double(), noisy.double()
    if a.ndim != 2 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Expected equally sized [N,D] feature batches, N >= 2")
    ma, mb = a.mean(0), b.mean(0)
    va = (a - ma).square().sum(1).mean()
    vb = (b - mb).square().sum(1).mean()
    energy = a.square().sum(1).mean() + b.square().sum(1).mean()
    gap = (ma - mb).square().sum()
    mmd = gap - (va + vb) / (len(a) - 1)
    denominator = energy.clamp_min(1e-30)
    return {
        "mmd2": mmd.item(),
        "l2_energy": energy.item(),
        "within_group_variance_fraction": ((va + vb) / denominator).item(),
        "mmd_over_twice_l2": (mmd / (2 * denominator)).item(),
        "clean_pairwise_rms": (2 * len(a) / (len(a) - 1) * va).sqrt().item(),
        "clean_noisy_mean_distance": gap.sqrt().item(),
    }


def shortcut_controls(clean, eps):
    """Controls without intact image layout; keep image/channel moments fixed."""
    axes = (2, 3)
    centered = eps - eps.mean(dim=axes, keepdim=True)
    matched = (centered / centered.square().mean(dim=axes, keepdim=True).sqrt().clamp_min(1e-12)
               * clean.var(dim=axes, correction=0, keepdim=True).sqrt()
               + clean.mean(dim=axes, keepdim=True))
    flat = clean.flatten(2)
    # A separate permutation per image, shared across RGB channels. This keeps
    # each RGB pixel intact and preserves the image's joint colour histogram.
    permutation = torch.rand(len(clean), flat.shape[-1], device=clean.device).argsort(dim=1)
    shuffled = flat.gather(2, permutation[:, None, :].expand_as(flat)).reshape_as(clean)
    return {"moment_matched_gaussian": matched, "shuffled_real_pixels": shuffled}


def control_summary(features, clean_features, noisy_features):
    """0=clean and 1=noisy along the learned witness direction, not a probability."""
    real, noisy, control = (value.double().mean(0)
                           for value in (clean_features, noisy_features, features))
    direction = noisy - real
    gap2 = direction.square().sum().clamp_min(1e-30)
    centered = features.double() - features.double().mean(0)
    return {
        "witness_position_clean0_noisy1": ((control-real).dot(direction) / gap2).item(),
        "mean_distance_to_clean_over_clean_noisy_gap": ((control-real).norm() / gap2.sqrt()).item(),
        "within_group_feature_rms": centered.square().sum(1).mean().sqrt().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--times", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--synthetic", action="store_true", help="Wiring check only")
    parser.add_argument("--shortcut-probes", action="store_true",
                        help="Probe moment-matched noise and pixel-shuffled real images")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    root = Path.cwd()
    if not (root / "dmmd" / "model.py").exists():
        parser.error("Run this command from the DMMD experiment repository")
    sys.path.insert(0, str(root))
    from dmmd.model import FeatureUNet
    from dmmd.noise import DDPMNoise
    from dmmd.runtime import file_sha256, frozen, load_checkpoint, seed_all

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    torch.set_num_threads(2)
    payload = load_checkpoint(args.checkpoint)
    if payload.get("kind") != "dmmd_features":
        parser.error("Expected a DMMD feature-network checkpoint")
    config = payload["config"]
    model = FeatureUNet(**config["model"]).to(device)
    model.load_state_dict(payload["model"])
    noise = DDPMNoise(**config["noise"]).to(device)
    seed_all(args.seed)
    if args.synthetic:
        clean = torch.rand(args.batch_size, 3, 32, 32) * 2 - 1
    else:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        dataset = CIFAR10(args.data_root, train=False, download=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,) * 3, (0.5,) * 3),
                         ]))
        indices = torch.randperm(len(dataset))[:args.batch_size]
        clean = torch.stack([dataset[int(i)][0] for i in indices])
    clean = clean.to(device)
    # Fix corruption across checkpoints, independently of model initialization.
    seed_all(args.seed + 1)
    eps = torch.randn_like(clean)
    controls = shortcut_controls(clean, eps) if args.shortcut_probes else {}
    head_width = config["model"]["head_dim"]
    print(json.dumps({"checkpoint": str(Path(args.checkpoint).resolve()),
                      "checkpoint_step": payload["step"],
                      "checkpoint_sha256": file_sha256(args.checkpoint),
                      "examples": len(clean), "dropout": "disabled (eval mode)",
                      "dataset": "synthetic wiring check" if args.synthetic
                      else "CIFAR-10 test split; no augmentation",
                      "seed": args.seed}), flush=True)
    with frozen(model):
        for time in args.times:
            t = clean.new_tensor(time)
            corrupted = noise(clean, t, eps)
            with torch.no_grad():
                hc, hn = model(clean, t), model(corrupted, t)
                difference = hn.mean(0) - hc.mean(0)
            query = corrupted.detach().requires_grad_(True)
            witness = (model(query, t) * difference).sum()
            gradient = torch.autograd.grad(witness, query)[0].detach()
            # Positive cosine: descent points toward radial contraction to zero.
            g, x = gradient.flatten(1), corrupted.flatten(1)
            cosine = (g * x).sum(1) / (g.norm(dim=1) * x.norm(dim=1)).clamp_min(1e-30)
            record = {"t": time, **feature_summary(hc, hn),
                      "head_variance_fractions": [
                          feature_summary(a, b)["within_group_variance_fraction"]
                          for a, b in zip(hc.split(head_width, 1), hn.split(head_width, 1))
                      ],
                      "witness_gradient_l2": g.norm(dim=1).mean().item(),
                      "descent_cosine_to_zero": cosine.mean().item()}
            if controls:
                with torch.no_grad():
                    record["controls"] = {
                        name: control_summary(model(images, t), hc, hn)
                        for name, images in controls.items()
                    }
            if not all(torch.isfinite(value).all() for value in (hc, hn, gradient)):
                raise FloatingPointError("Non-finite features or pixel gradients")
            print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
