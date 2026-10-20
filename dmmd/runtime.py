import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(config, distributed=False):
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    rank = int(os.environ.get("RANK", 0)) if distributed else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if distributed else 1
    expected = config["runtime"].get("expected_world_size")
    if distributed and expected is not None and expected != world_size:
        raise ValueError(f"This config requires {expected} processes, got {world_size}")
    device_name = config["runtime"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        local = int(os.environ.get("LOCAL_RANK", 0)) if world_size > 1 else (device.index or 0)
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return device, rank, world_size


def synchronize_gradients(model, world_size):
    """Explicit all-reduce after backward, including second-order GP gradients.

    Avoid DDP forward/reducer assumptions around repeated forwards and
    autograd.grad. MMD statistics are local or global according to mmd_scope.
    """
    if world_size == 1:
        return
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Only use checkpoints produced by this experiment or otherwise trusted.
    return torch.load(path, map_location=device, weights_only=False)


def rng_state():
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def frozen(model):
    mode = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    model.requires_grad_(False)
    try:
        yield model
    finally:
        model.train(mode)
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)


def save_grid(images, path, nrow=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clamp for visualization only; do not change sampler state.
    save_image(images.detach().cpu().clamp(-1, 1).add(1).div(2), path, nrow=nrow)


def append_json(path, record):
    with Path(path).open("a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
