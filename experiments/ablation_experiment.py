from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import create_dataloaders, load_embedding_splits
from src.train import run_ablation_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    train_samples, valid_samples, test_samples = load_embedding_splits(args.pt_path)
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_samples,
        valid_samples,
        test_samples,
        batch_size=args.batch_size,
    )
    run_ablation_experiment(
        train_loader,
        valid_loader,
        test_loader,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
