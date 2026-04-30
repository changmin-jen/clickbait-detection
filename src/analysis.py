from __future__ import annotations

import torch

from .utils import move_batch_to_device


@torch.no_grad()
def collect_branch_attention(model, loader, device: str | torch.device) -> dict:
    model.eval()
    attention_batches = []
    branch_names = None

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        attention_batches.append(outputs["branch_attention"].detach().cpu())
        branch_names = outputs.get("branch_names", branch_names)

    attention = torch.cat(attention_batches, dim=0)
    return {
        "branch_names": list(branch_names or []),
        "alpha_all": attention,
        "mean_alpha": attention.mean(dim=0),
        "std_alpha": attention.std(dim=0),
    }


def save_branch_attention_plot(attention_summary: dict, save_path: str) -> None:
    import matplotlib.pyplot as plt

    names = attention_summary["branch_names"]
    mean_alpha = attention_summary["mean_alpha"]

    plt.figure(figsize=(6, 4))
    plt.bar(names, mean_alpha.numpy())
    plt.xlabel("Modality Pair")
    plt.ylabel("Average Attention Weight")
    plt.ylim(0, 1.0)

    for idx, value in enumerate(mean_alpha.numpy()):
        plt.text(idx, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
