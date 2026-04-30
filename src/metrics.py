from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .utils import move_batch_to_device


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
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
        "auc": auc,
        "acc": accuracy_score(labels, preds),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def search_best_threshold(labels: np.ndarray, probs: np.ndarray, metric: str = "f1"):
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for threshold in np.arange(0.1, 0.91, 0.01):
        metrics = compute_metrics(labels, probs, threshold=threshold)
        if metrics[metric] > best_score:
            best_threshold = threshold
            best_score = metrics[metric]
            best_metrics = metrics

    return best_threshold, best_metrics


@torch.no_grad()
def collect_predictions(model, loader, device: str | torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    probs = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        probs.extend(torch.sigmoid(outputs["fused_logit"]).cpu().numpy().tolist())
        labels.extend(batch["labels"].cpu().numpy().tolist())

    return np.array(labels), np.array(probs)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: str | torch.device,
    criterion=None,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    probs = []
    losses = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        probs.extend(torch.sigmoid(outputs["fused_logit"]).cpu().numpy().tolist())
        labels.extend(batch["labels"].cpu().numpy().tolist())

        if criterion is not None:
            loss = criterion(outputs, batch["labels"])
            if isinstance(loss, tuple):
                loss = loss[0]
            losses.append(loss.item())

    labels = np.array(labels)
    probs = np.array(probs)
    metrics = compute_metrics(labels, probs, threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, labels, probs
