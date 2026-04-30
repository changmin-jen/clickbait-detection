from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import create_dataloaders, load_embedding_splits
from src.train import run_ablation_experiment
from src.utils import get_device, set_seed


ABLATION_CONFIGS = {
    "TS": ["ts"],
    "ThK": ["thk"],
    "TK": ["tk"],
    "ThS": ["ths"],
    "TS+ThS": ["ts", "ths"],
    "TK+ThK": ["tk", "thk"],
    "TS+TK": ["ts", "tk"],
    "ThS+ThK": ["ths", "thk"],
    "w/o TS": ["tk", "thk", "ths"],
    "w/o TK": ["ts", "thk", "ths"],
    "w/o ThS": ["ts", "tk", "thk"],
    "w/o ThK": ["ts", "tk", "ths"],
    "All": ["tk", "ts", "thk", "ths"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", required=True)
    parser.add_argument("--experiments", nargs="+", default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("DEVICE:", device)

    train_samples, valid_samples, test_samples = load_embedding_splits(args.pt_path)
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_samples,
        valid_samples,
        test_samples,
        batch_size=args.batch_size,
    )

    results = {}
    for name in args.experiments:
        if name not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation experiment: {name}")

        result = run_ablation_experiment(
            name,
            ABLATION_CONFIGS[name],
            train_loader,
            valid_loader,
            test_loader,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
        )
        results[name] = result.test_at_half

    print("\n===== Summary =====")
    for name, metrics in results.items():
        print(name, metrics)


if __name__ == "__main__":
    main()
