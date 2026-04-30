from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPooling(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, seq_embs: torch.Tensor, seq_mask: torch.Tensor, query: torch.Tensor | None = None):
        mask = seq_mask.unsqueeze(-1).float()
        pooled = (seq_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(self.eps)

        attn = seq_mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return pooled, attn


class TopKPooling(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

    def forward(self, seq_embs: torch.Tensor, seq_mask: torch.Tensor, query: torch.Tensor | None = None):
        norms = torch.norm(seq_embs, dim=-1).masked_fill(~seq_mask, -1e9)
        k_eff = min(self.k, seq_embs.size(1))
        topk_idx = torch.topk(norms, k=k_eff, dim=1).indices

        gathered = torch.gather(
            seq_embs,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, seq_embs.size(-1)),
        )
        pooled = gathered.mean(dim=1)

        attn = torch.zeros_like(norms)
        attn.scatter_(1, topk_idx, 1.0 / k_eff)
        return pooled, attn


class TopKSimPooling(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

    def forward(self, seq_embs: torch.Tensor, seq_mask: torch.Tensor, query: torch.Tensor):
        seq_norm = F.normalize(seq_embs, p=2, dim=-1)
        q_norm = F.normalize(query, p=2, dim=-1).unsqueeze(1)
        sim = (seq_norm * q_norm).sum(dim=-1).masked_fill(~seq_mask, -1e9)

        k_eff = min(self.k, seq_embs.size(1))
        topk_idx = torch.topk(sim, k=k_eff, dim=1).indices

        gathered = torch.gather(
            seq_embs,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, seq_embs.size(-1)),
        )
        pooled = gathered.mean(dim=1)

        attn = torch.zeros_like(sim)
        attn.scatter_(1, topk_idx, 1.0 / k_eff)
        return pooled, attn


class AttentionPooling(nn.Module):
    def __init__(self, tau: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.tau = tau
        self.eps = eps

    def forward(self, seq_embs: torch.Tensor, seq_mask: torch.Tensor, query: torch.Tensor):
        seq_norm = F.normalize(seq_embs, p=2, dim=-1)
        q_norm = F.normalize(query, p=2, dim=-1).unsqueeze(1)

        sim = (seq_norm * q_norm).sum(dim=-1) / self.tau
        sim = sim.masked_fill(~seq_mask, -1e9)

        attn = F.softmax(sim, dim=-1)
        attn = attn * seq_mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(self.eps)

        pooled = torch.bmm(attn.unsqueeze(1), seq_embs).squeeze(1)
        return pooled, attn
