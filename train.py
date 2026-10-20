#!/usr/bin/env python3
"""Train only the noise-conditioned feature network."""
import argparse
import copy
import fcntl
import json
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from dmmd.config import ROOT, load_config, project_path
from dmmd.data import datasets, first_images, infinite_batches, training_loader
from dmmd.model import FeatureUNet
from dmmd.noise import DDPMNoise
from dmmd.objective import feature_loss, linear_mmd2
from dmmd.runtime import (
    append_json, atomic_save, frozen, load_checkpoint, restore_rng, rng_state,
    save_grid, seed_all, setup, synchronize_gradients,
)
from dmmd.sampling import feature_mean, noise_grid, sample_particles


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.yaml"))
    parser.add_argument("--workdir", default="runs/cifar10_pilot")
    parser.add_argument("--data-root")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int, help="Total optimizer steps, including resumed steps")
    parser.add_argument("--resume", nargs="?", const="auto",
                        help="Checkpoint path, or omit value to use workdir/checkpoints/latest.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def checkpoint_path(workdir, resume):
    return workdir / "checkpoints/latest.pt" if resume == "auto" else project_path(resume)


def validate_resume(old, new):
    # Runtime, duration, cadence, dataset location/worker count and preview
    # settings may change. Changing the training recipe requires a new run.
    for key in ("model", "noise"):
        if old[key] != new[key]:
            raise ValueError(f"Resume would change {key}; start a new workdir")
    for key in ("lr", "betas", "batch_size", "noise_levels_per_step",
                "lambda_l2", "lambda_gp", "max_grad_norm"):
        if old["train"][key] != new["train"][key]:
            raise ValueError(f"Resume would change train.{key}; start a new workdir")
    if old["train"].get("mmd_scope", "local") != new["train"].get("mmd_scope", "local"):
        raise ValueError("Resume would change train.mmd_scope; start a new workdir")
    for key in ("validation_size", "random_flip", "synthetic", "synthetic_size"):
        if old["data"][key] != new["data"][key]:
            raise ValueError(f"Resume would change data.{key}; start a new workdir")
    if old["seed"] != new["seed"]:
        raise ValueError("Resume would change the experiment seed")


def save_checkpoint(model, optimizer, step, config, workdir, rank, world_size):
    state = rng_state()
    if world_size > 1:
        states = [None] * world_size
        dist.all_gather_object(states, state)
    else:
        states = [state]
    if rank == 0:
        payload = {
            "format_version": 1, "kind": "dmmd_features", "step": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": config, "rng_states": states, "world_size": world_size,
        }
        directory = workdir / "checkpoints"
        atomic_save(payload, directory / f"step_{step:08d}.pt")
        atomic_save(payload, directory / "latest.pt")
        saved = sorted(directory.glob("step_*.pt"))
        for old in saved[:-config["train"]["keep_last"]]:
            old.unlink()
        print(f"Saved feature checkpoint at step {step}", flush=True)


def preview(model, noise, reference, validation, config, step, workdir, device):
    # Preserve training randomness: preview frequency should not change
    # the sequence of training corruptions/dropout masks.
    state = rng_state()
    started = time.perf_counter()
    try:
        seed_all(config["seed"] + 100000)
        with frozen(model):
            if validation is not None:
                clean = validation.to(device)
                with torch.no_grad():
                    for t_value in (0.1, 0.5, 0.9):
                        t = clean.new_tensor(t_value)
                        h_clean = model(clean, t)
                        h_noisy = model(noise(clean, t), t)
                        record = {
                            "step": step, "kind": "heldout_features", "t": t_value,
                            "mmd2": linear_mmd2(h_noisy, h_clean).item(),
                            "clean_feature_rms": h_clean.square().mean().sqrt().item(),
                            "noisy_feature_rms": h_noisy.square().mean().sqrt().item(),
                        }
                        append_json(workdir / "validation.jsonl", record)
            sample_config = copy.deepcopy(config["sampling"])
            sample_config.update({
                "noise_levels": config["preview"]["noise_levels"],
                "steps_per_level": config["preview"]["steps_per_level"],
            })
            times = noise_grid(sample_config).to(device)
            real = reference.to(device)
            means = torch.stack([
                feature_mean(model, real, t, sample_config["feature_batch_size"])
                for t in times
            ])
            initial = torch.randn(config["preview"]["particles"], 3, 32, 32, device=device)
            trace = []

            def record(level, inner, t, particles, gradient):
                if inner == sample_config["steps_per_level"] - 1:
                    trace.append({
                        "level": level, "t": t.item(),
                        "pixel_rms": particles.square().mean().sqrt().item(),
                        "grad_l2": gradient.flatten(1).norm(dim=1).mean().item(),
                    })

            final = sample_particles(
                model, initial, times, means, sample_config["steps_per_level"],
                sample_config["step_size"], pixel_clip=sample_config["pixel_clip"],
                callback=record,
            )
            directory = workdir / "previews" / f"step_{step:08d}"
            save_grid(initial, directory / "initial_noise.png")
            save_grid(final, directory / "generated.png")
            save_grid(real[:64], directory / "reference.png")
            (directory / "trace.json").write_text(json.dumps(trace, indent=2))
            append_json(workdir / "validation.jsonl", {
                "step": step, "kind": "particle_preview",
                "seconds": time.perf_counter() - started, "trace": trace,
            })
    finally:
        restore_rng(state)


def run():
    args = arguments()
    overrides = list(args.set)
    if args.data_root:
        overrides.append("data.root=" + json.dumps(args.data_root))
    if args.download:
        overrides.append("data.download=true")
    if args.device:
        overrides.append("runtime.device=" + args.device)
    if args.steps is not None:
        overrides.append(f"train.steps={args.steps}")
    config = load_config(args.config, overrides)
    device, rank, world_size = setup(config, distributed=True)
    workdir = project_path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # Prevent two jobs from writing to the same run, including the first update.
    # The OS releases this lock on exit, even after SIGKILL.
    run_lock = None
    if rank == 0:
        run_lock = (workdir / ".run.lock").open("a")
        try:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another job is using {workdir}") from error
    seed_all(config["seed"])  # Initialize identical weights on all ranks.
    model = FeatureUNet(**config["model"]).to(device)
    noise = DDPMNoise(**config["noise"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["lr"], betas=config["train"]["betas"]
    )
    completed = 0
    resumed = None
    if args.resume:
        resumed = load_checkpoint(checkpoint_path(workdir, args.resume), "cpu")
        if resumed.get("kind") != "dmmd_features":
            raise ValueError("Expected a DMMD feature-network checkpoint")
        validate_resume(resumed["config"], config)
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        completed = int(resumed["step"])
        if config["train"]["steps"] <= completed:
            raise ValueError("--steps must exceed the checkpoint's completed step")
    elif (workdir / "checkpoints").exists() or (workdir / "train.jsonl").exists():
        raise FileExistsError(f"Existing run at {workdir}; use --resume or a new workdir")
    if rank == 0:
        filename = "resume_config.yaml" if args.resume else "config.yaml"
        (workdir / filename).write_text(yaml.safe_dump(config, sort_keys=False))
        print(json.dumps({
            "device": str(device), "world_size": world_size,
            "batch_per_rank": config["train"]["batch_size"],
            "global_images_per_step": config["train"]["batch_size"] * world_size,
            "mmd_scope": config["train"]["mmd_scope"],
            "noise_levels_per_step": config["train"]["noise_levels_per_step"],
            "features": model.output_dim,
            "parameters": sum(p.numel() for p in model.parameters()),
            "workdir": str(workdir), "completed_steps": completed,
            "synthetic": config["data"]["synthetic"],
        }, indent=2), flush=True)
    # Download once on rank zero, then all other ranks read the prepared files.
    if rank == 0:
        train_set, val_set, reference_set = datasets(config)
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        config["data"]["download"] = False
        train_set, val_set, reference_set = datasets(config)
    loader, sampler = training_loader(train_set, config, rank, world_size)
    batches = infinite_batches(loader, sampler, completed)
    reference = validation = None
    if rank == 0 and config["train"]["eval_every"]:
        reference = first_images(reference_set, config["preview"]["reference_size"])
        if len(val_set) >= 2:
            validation = first_images(val_set, config["preview"]["validation_batch_size"])
    seed_all(config["seed"] + rank)
    if resumed is not None:
        states = resumed.get("rng_states", [])
        if len(states) == world_size:
            restore_rng(states[rank])
        else:
            seed_all(config["seed"] + rank + completed)
            if rank == 0:
                print("GPU count changed; resuming with fresh per-rank RNG states.", flush=True)
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    for name in ("SIGTERM", "SIGINT", "SIGUSR1", "SIGUSR2"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    started = time.perf_counter()
    log_started, log_step = started, completed
    model.train()
    for step in range(completed + 1, config["train"]["steps"] + 1):
        clean = next(batches).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        levels = config["train"]["noise_levels_per_step"]
        for _ in range(levels):
            t = torch.empty((), device=device).uniform_(
                config["noise"]["t_min"], config["noise"]["t_max"]
            )
            # A common noise level across ranks; each rank has its own data/noise.
            if world_size > 1:
                dist.broadcast(t, src=0)
            noisy = noise(clean, t)
            loss, metrics = feature_loss(
                model, clean, noisy, t, config["train"]["lambda_l2"],
                config["train"]["lambda_gp"], create_graph=True,
                mmd_scope=config["train"]["mmd_scope"],
            )
            (loss / levels).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0) + value / levels
        synchronize_gradients(model, world_size)
        threshold = config["train"]["max_grad_norm"]
        parameter_grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float("inf") if threshold is None else threshold,
            error_if_nonfinite=True,
        )
        optimizer.step()
        completed = step
        if step % config["train"]["log_every"] == 0 or step == 1:
            if world_size > 1:
                for value in totals.values():
                    dist.all_reduce(value)
                    value.div_(world_size)
            if rank == 0:
                now = time.perf_counter()
                record = {"step": step, **{k: v.item() for k, v in totals.items()},
                          "parameter_grad_l2": parameter_grad.item(),
                          "steps_per_second": (step - log_step) / (now - log_started),
                          "elapsed_seconds": now - started, "lr": optimizer.param_groups[0]["lr"]}
                append_json(workdir / "train.jsonl", record)
                print(json.dumps(record), flush=True)
            log_started, log_step = time.perf_counter(), step
        should_stop = torch.tensor(int(stop["requested"]), device=device)
        if world_size > 1:
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
        final_step = step == config["train"]["steps"] or bool(should_stop.item())
        if step % config["train"]["save_every"] == 0 or final_step:
            save_checkpoint(model, optimizer, step, config, workdir, rank, world_size)
        if config["train"]["eval_every"] and (
            step % config["train"]["eval_every"] == 0 or final_step
        ):
            # All ranks pause before the next gradient collective.
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                preview(model, noise, reference, validation, config, step, workdir, device)
            if world_size > 1:
                dist.barrier()
        if final_step:
            break
    if rank == 0:
        print(f"Finished at step {completed}. Checkpoint: {workdir}/checkpoints/latest.pt",
              flush=True)


if __name__ == "__main__":
    try:
        run()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
