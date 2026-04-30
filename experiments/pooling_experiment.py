from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import create_dataloaders, load_embedding_splits
from src.train import compute_pos_weight, run_pooling_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", required=True)
    parser.add_argument("--pooling", nargs="+", default=["mean", "topk", "topk_sim", "attention"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--save-dir", default="results/models")
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
    pos_weight = compute_pos_weight(train_samples)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for pooling_type in args.pooling:
        run_pooling_experiment(
            pooling_type,
            train_loader,
            valid_loader,
            test_loader,
            pos_weight=pos_weight,
            device=device,
            epochs=args.epochs,
            topk=args.topk,
            tau=args.tau,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
