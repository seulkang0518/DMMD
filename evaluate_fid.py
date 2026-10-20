#!/usr/bin/env python3
"""Optional FID/IS on saved PNGs, against the full CIFAR-10 training set."""
import argparse
import json

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from dmmd.config import project_path


class Uint8CIFAR(Dataset):
    def __init__(self, root):
        self.images = CIFAR10(root, train=True, download=False).data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[index].copy()).permute(2, 0, 1)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="sample.py output directory")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    try:
        import torch_fidelity
    except ImportError as error:
        raise SystemExit("Install optional metrics: pip install -r requirements-fid.txt") from error
    directory = project_path(args.samples)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("Expected a directory produced by sample.py")
    metadata = json.loads(metadata_path.read_text())
    if metadata["synthetic"]:
        raise ValueError("Synthetic smoke-test images are not eligible for CIFAR-10 FID")
    images = directory / "images"
    count = len(list(images.glob("*.png")))
    if count != metadata["num_samples"] or count < 2:
        raise ValueError("Image count does not match metadata, or there are too few images")
    metrics = torch_fidelity.calculate_metrics(
        input1=str(images), input2=Uint8CIFAR(project_path(args.data_root)),
        cuda=torch.cuda.is_available() and not args.cpu, batch_size=args.batch_size,
        fid=True, isc=True, verbose=True,
    )
    record = {
        "reference": "all 50000 unaugmented CIFAR-10 training images",
        "generated_samples": count,
        "protocol": "uint8 PNG -> torch-fidelity default Inception features",
        "metrics": metrics,
    }
    (directory / "fid.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    run()
