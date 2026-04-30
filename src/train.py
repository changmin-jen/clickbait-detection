from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .loss import CombinedBCELoss, compute_loss
from .metrics import evaluate, search_best_threshold
from .model import ModalityAblationModel, PoolingComparisonModel
from .utils import move_batch_to_device


@dataclass
class TrainResult:
    model: nn.Module
    best_state: dict
    test_at_half: dict
    test_at_best_threshold: dict | None = None


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
    device: str | torch.device,
) -> float:
    model.train()
    losses = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        loss = criterion(model(batch), batch["labels"])
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def fit_with_validation(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device: str | torch.device,
    epochs: int = 30,
    patience: int = 8,
    threshold_search: bool = True,
    scheduler=None,
    label: str = "model",
):
    best_score = -1.0
    best_threshold = 0.5
    best_state = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, val_labels, val_probs = evaluate(
            model,
            valid_loader,
            device=device,
            criterion=criterion,
            threshold=0.5,
        )

        if threshold_search:
            current_threshold, val_metrics = search_best_threshold(val_labels, val_probs)
        else:
            current_threshold = 0.5
            val_metrics, _, _ = evaluate(
                model,
                valid_loader,
                device=device,
                criterion=criterion,
                threshold=0.5,
            )

        if scheduler is not None:
            scheduler.step(val_metrics["f1"])

        print(
            f"[{label}] epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_score:
            best_score = val_metrics["f1"]
            best_threshold = current_threshold
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "best_val_f1": best_score,
                "best_threshold": best_threshold,
            }
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[{label}] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid best state.")

    model.load_state_dict(best_state["model"])
    return best_state


def run_pooling_experiment(
    pooling_type: str,
    train_loader,
    valid_loader,
    test_loader,
    pos_weight: float,
    device: str | torch.device,
    epochs: int = 30,
    patience: int = 8,
    topk: int = 3,
    tau: float = 0.5,
    save_dir: str | None = None,
) -> TrainResult:
    print(f"\n===== Pooling: {pooling_type} =====")

    model = PoolingComparisonModel(
        d_model=768,
        hidden_dim=64,
        dropout=0.4,
        pooling_type=pooling_type,
        tau=tau,
        topk=topk,
    ).to(device)
    criterion = CombinedBCELoss(
        branch_weight=0.5,
        fused_weight=1.5,
        pos_weight=pos_weight,
        label_smooth=0.01,
        branch_reduction="sum",
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_state = fit_with_validation(
        model,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        device,
        epochs=epochs,
        patience=patience,
        threshold_search=True,
        scheduler=scheduler,
        label=pooling_type,
    )
    best_state.update({"pooling_type": pooling_type, "tau": tau})

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / f"{pooling_type}_tau{tau}_best.pt"
        torch.save(best_state, save_path)
        print(f"Saved best model to: {save_path}")

    test_at_half, _, _ = evaluate(model, test_loader, device=device, criterion=criterion, threshold=0.5)
    test_at_best, _, _ = evaluate(
        model,
        test_loader,
        device=device,
        criterion=criterion,
        threshold=best_state["best_threshold"],
    )

    print(f"\n[{pooling_type}] Test @ 0.50")
    print(test_at_half)
    print(f"\n[{pooling_type}] Test @ Best Threshold from Valid")
    print(test_at_best)

    return TrainResult(model, best_state, test_at_half, test_at_best)


def run_ablation_experiment(
    name: str,
    active_branches: list[str],
    train_loader,
    valid_loader,
    test_loader,
    device: str | torch.device,
    epochs: int = 30,
    patience: int = 8,
) -> TrainResult:
    print(f"\n===== Ablation: {name} =====")

    model = ModalityAblationModel(
        d_model=768,
        hidden_dim=64,
        dropout=0.4,
        pooling_type="attention",
        tau=0.5,
        topk=3,
        active_branches=active_branches,
    ).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

    def criterion(outputs, labels):
        return compute_loss(
            outputs,
            labels.float(),
            bce,
            branch_weight=0.5,
            fused_weight=1.5,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    best_state = fit_with_validation(
        model,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        device,
        epochs=epochs,
        patience=patience,
        threshold_search=False,
        label=name,
    )
    best_state.update({"experiment": name, "active_branches": active_branches})

    test_at_half, _, _ = evaluate(model, test_loader, device=device, criterion=criterion, threshold=0.5)
    print(f"\n[{name}] Test @ 0.50")
    print(test_at_half)

    return TrainResult(model, best_state, test_at_half)
