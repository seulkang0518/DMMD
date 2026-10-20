import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

from .config import project_path


class TensorImages(Dataset):
    def __init__(self, size, seed=0):
        generator = torch.Generator().manual_seed(seed)
        # Only a wiring test distribution; not a CIFAR-10 substitute.
        self.images = torch.rand(size, 3, 32, 32, generator=generator) * 2 - 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], 0


def datasets(config, augment=True):
    c = config["data"]
    if c["synthetic"]:
        raw = TensorImages(c["synthetic_size"], config["seed"])
        training = raw
    else:
        common = [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        raw = CIFAR10(project_path(c["root"]), train=True, download=c["download"],
                      transform=transforms.Compose(common))
        transform = common
        if augment and c["random_flip"]:
            transform = [transforms.RandomHorizontalFlip()] + common
        training = CIFAR10(project_path(c["root"]), train=True, download=False,
                           transform=transforms.Compose(transform))
    val_size = c["validation_size"]
    if val_size >= len(raw):
        raise ValueError("validation_size must leave a nonempty training split")
    order = torch.randperm(len(raw), generator=torch.Generator().manual_seed(config["seed"]))
    val_indices = order[:val_size].tolist()
    train_indices = order[val_size:].tolist()
    return (Subset(training, train_indices), Subset(raw, val_indices),
            Subset(raw, train_indices))


def seed_worker(worker_id):
    import random
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def training_loader(dataset, config, rank=0, world_size=1):
    # DistributedSampler also gives repeatable epoch-based order on one GPU.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=config["seed"], drop_last=True)
    workers = config["data"]["num_workers"]
    loader = DataLoader(
        dataset, batch_size=config["train"]["batch_size"], sampler=sampler,
        num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=workers > 0, worker_init_fn=seed_worker,
    )
    if len(loader) == 0:
        raise ValueError("Not enough training images for a full batch on every rank")
    return loader, sampler


def infinite_batches(loader, sampler, completed_steps):
    epoch, offset = divmod(completed_steps, len(loader))
    while True:
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if i >= offset:
                yield batch[0]
        epoch, offset = epoch + 1, 0


def first_images(dataset, count):
    if len(dataset) == 0:
        raise ValueError("The requested image split is empty")
    return torch.stack([dataset[i][0] for i in range(min(count, len(dataset)))])
