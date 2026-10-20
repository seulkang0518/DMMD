#!/usr/bin/env python3
"""Download CIFAR-10 on a network-enabled login/local machine."""
import argparse

from torchvision.datasets import CIFAR10

from dmmd.config import project_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()
    root = project_path(args.data_root)
    CIFAR10(root, train=True, download=True)
    CIFAR10(root, train=False, download=True)
    print(f"CIFAR-10 is ready at {root}")
