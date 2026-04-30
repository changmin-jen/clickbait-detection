from __future__ import annotations

import torch
import torch.nn as nn


class CombinedBCELoss(nn.Module):
    def __init__(
        self,
        lambda_br: float = 0.5,
        mu: float = 1.5,
        pos_weight: float | None = None,
        label_smooth: float = 0.01,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.lambda_br = lambda_br
        self.mu = mu
        self.label_smooth = label_smooth

        if pos_weight is not None:
            weight_device = device if device is not None else "cpu"
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=weight_device),
            )
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        return y * (1 - self.label_smooth) + 0.5 * self.label_smooth

    def forward(self, outputs: dict, labels: torch.Tensor):
        y_sm = self.smooth_labels(labels.to(outputs["fused_logit"].device))

        loss_br = 0.0
        for logit in outputs["branch_logits"].values():
            loss_br = loss_br + self.bce(logit, y_sm)

        loss_fuse = self.bce(outputs["fused_logit"], y_sm)
        loss = self.lambda_br * loss_br + self.mu * loss_fuse
        return loss, loss_br.detach(), loss_fuse.detach()


def compute_loss(
    outputs: dict,
    labels: torch.Tensor,
    criterion: nn.Module,
    br_w: float = 0.5,
    mu: float = 1.5,
) -> torch.Tensor:
    branch_losses = [criterion(logit, labels) for logit in outputs["branch_logits"].values()]
    branch_loss = torch.stack(branch_losses).mean() if branch_losses else 0.0
    fuse_loss = criterion(outputs["fuse_logit"], labels)
    return br_w * branch_loss + mu * fuse_loss
