from __future__ import annotations

import torch
import torch.nn as nn


class CombinedBCELoss(nn.Module):
    def __init__(
        self,
        branch_weight: float = 0.5,
        fused_weight: float = 1.5,
        pos_weight: float | None = None,
        label_smooth: float = 0.01,
        branch_reduction: str = "sum",
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.branch_weight = branch_weight
        self.fused_weight = fused_weight
        self.label_smooth = label_smooth
        self.branch_reduction = branch_reduction

        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            weight_device = device if device is not None else "cpu"
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=weight_device),
            )

    def _smooth(self, labels: torch.Tensor) -> torch.Tensor:
        return labels * (1 - self.label_smooth) + 0.5 * self.label_smooth

    def forward(self, outputs: dict, labels: torch.Tensor):
        labels = labels.to(outputs["fused_logit"].device).float()
        labels = self._smooth(labels)

        branch_losses = [
            self.bce(logit, labels)
            for logit in outputs["branch_logits"].values()
        ]
        branch_loss = torch.stack(branch_losses)
        if self.branch_reduction == "mean":
            branch_loss = branch_loss.mean()
        elif self.branch_reduction == "sum":
            branch_loss = branch_loss.sum()
        else:
            raise ValueError(f"Unknown branch_reduction: {self.branch_reduction}")

        fused_loss = self.bce(outputs["fused_logit"], labels)
        total_loss = self.branch_weight * branch_loss + self.fused_weight * fused_loss
        return total_loss, branch_loss.detach(), fused_loss.detach()


def compute_loss(
    outputs: dict,
    labels: torch.Tensor,
    criterion: nn.Module,
    branch_weight: float = 0.5,
    fused_weight: float = 1.5,
) -> torch.Tensor:
    labels = labels.float()
    branch_losses = [
        criterion(logit, labels)
        for logit in outputs["branch_logits"].values()
    ]
    branch_loss = torch.stack(branch_losses).mean()
    fused_loss = criterion(outputs["fused_logit"], labels)
    return branch_weight * branch_loss + fused_weight * fused_loss
