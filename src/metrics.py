from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .loss import compute_loss


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def search_best_threshold(labels: np.ndarray, probs: np.ndarray, metric: str = "f1"):
    best_thr = 0.5
    best_score = -1
    best_metrics = None

    for thr in np.arange(0.1, 0.91, 0.01):
        metrics = compute_metrics(labels, probs, threshold=thr)
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best_thr = thr
            best_metrics = metrics

    return best_thr, best_metrics


@torch.no_grad()
def evaluate(model, loader, criterion=None, threshold: float = 0.5, device: str | torch.device | None = None):
    model.eval()
    all_labels = []
    all_probs = []
    losses = []

    for batch in loader:
        if device is not None:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

        outputs = model(batch)
        probs = torch.sigmoid(outputs["fused_logit"]).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

        if criterion is not None:
            loss, _, _ = criterion(outputs, batch["labels"])
            losses.append(loss.item())

    metrics = compute_metrics(np.array(all_labels), np.array(all_probs), threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, np.array(all_labels), np.array(all_probs)


@torch.no_grad()
def evaluate_model(model, loader, device: str | torch.device = "cuda", threshold: float = 0.5):
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0
    total_n = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        labels = batch["labels"].float()
        outputs = model(batch)

        loss = compute_loss(outputs, labels, criterion, br_w=0.5, mu=1.5)
        probs = torch.sigmoid(outputs["fuse_logit"])

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_n += batch_size

        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics
