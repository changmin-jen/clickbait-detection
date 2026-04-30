from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn

from .loss import CombinedBCELoss, compute_loss
from .metrics import evaluate, evaluate_model, search_best_threshold
from .model import ModalityAblationModel, PoolingComparisonModel


def compute_pos_weight(train_samples: list[dict]) -> float:
    train_labels = np.array([x["label_id"] for x in train_samples])
    num_pos = (train_labels == 1).sum()
    num_neg = (train_labels == 0).sum()
    return (num_neg / max(num_pos, 1)) if num_pos > 0 else 1.0


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: str | torch.device = "cuda",
    br_w: float = 0.5,
    mu: float = 1.5,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        labels = batch["labels"].float()

        optimizer.zero_grad()
        outputs = model(batch)
        loss = compute_loss(outputs, labels, criterion, br_w=br_w, mu=mu)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_n += batch_size

    return total_loss / max(total_n, 1)


def run_pooling_experiment(
    pooling_type: str,
    train_loader,
    valid_loader,
    test_loader,
    pos_weight: float,
    device: str | torch.device,
    epochs: int = 30,
    topk: int = 3,
    tau: float = 0.5,
    save_dir: str | None = None,
):
    print(f"\n===== Running pooling: {pooling_type} =====")

    model = PoolingComparisonModel(
        d_model=768,
        hidden_dim=64,
        dropout=0.4,
        pooling_type=pooling_type,
        tau=tau,
        topk=topk,
    ).to(device)

    criterion = CombinedBCELoss(
        lambda_br=0.5,
        mu=1.5,
        pos_weight=pos_weight,
        label_smooth=0.01,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_f1 = -1
    best_state = None
    best_thr = 0.5
    patience = 8
    patience_count = 0
    save_path = None

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{pooling_type}_tau{tau}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            outputs = model(batch)
            loss, _, _ = criterion(outputs, batch["labels"])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        _, val_labels, val_probs = evaluate(
            model,
            valid_loader,
            criterion=criterion,
            threshold=0.5,
            device=device,
        )
        cur_thr, cur_best_metrics = search_best_threshold(val_labels, val_probs, metric="f1")
        scheduler.step(cur_best_metrics["f1"])

        print(
            f"[{pooling_type}] Epoch {epoch:02d} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"val_f1={cur_best_metrics['f1']:.4f} | "
            f"val_auc={cur_best_metrics['auc']:.4f}"
        )

        if cur_best_metrics["f1"] > best_val_f1:
            best_val_f1 = cur_best_metrics["f1"]
            best_thr = cur_thr
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "best_thr": best_thr,
                "pooling_type": pooling_type,
                "tau": tau,
            }
            patience_count = 0

            if save_path is not None:
                torch.save(best_state, save_path)
                print(f"Saved best model to: {save_path}")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state["model"])
    test_metrics_05, _, _ = evaluate(
        model,
        test_loader,
        criterion=criterion,
        threshold=0.5,
        device=device,
    )
    test_metrics_best, _, _ = evaluate(
        model,
        test_loader,
        criterion=criterion,
        threshold=best_state["best_thr"],
        device=device,
    )

    print(f"\n[{pooling_type}] Test @ 0.50")
    print(test_metrics_05)
    print(f"\n[{pooling_type}] Test @ Best Threshold from Valid")
    print(test_metrics_best)

    return model, best_state, test_metrics_05, test_metrics_best


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


def run_ablation_experiment(
    train_loader,
    valid_loader,
    test_loader,
    device: str | torch.device,
    epochs: int = 30,
    patience: int = 8,
) -> dict:
    results = {}

    for exp_name, branches in ABLATION_CONFIGS.items():
        print(f"\n===== {exp_name} =====")

        model = ModalityAblationModel(
            d_model=768,
            hidden_dim=64,
            dropout=0.4,
            pooling_type="attention",
            tau=0.5,
            topk=3,
            active_branches=branches,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

        best_val_f1 = -1
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device=device,
                br_w=0.5,
                mu=1.5,
            )
            val_metrics = evaluate_model(model, valid_loader, device=device, threshold=0.5)

            print(
                f"[{exp_name}] epoch {epoch + 1:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_auc={val_metrics['auc']:.4f}"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"[{exp_name}] Early stopping at epoch {epoch + 1}")
                break

        model.load_state_dict(best_state)
        test_metrics = evaluate_model(model, test_loader, device=device, threshold=0.5)
        results[exp_name] = test_metrics

        print(f"\n[{exp_name}] Test @ 0.50")
        print(test_metrics)

    return results
