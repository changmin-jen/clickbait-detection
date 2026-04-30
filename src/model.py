from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import AttentionPooling, MeanPooling, TopKPooling, TopKSimPooling


BRANCH_ORDER = ("tk", "ts", "thk", "ths")


def pair_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)


def build_pooling(pooling_type: str, tau: float = 0.5, topk: int = 3) -> nn.Module:
    if pooling_type == "mean":
        return MeanPooling()
    if pooling_type == "topk":
        return TopKPooling(k=topk)
    if pooling_type == "topk_sim":
        return TopKSimPooling(k=topk)
    if pooling_type == "attention":
        return AttentionPooling(tau=tau)
    raise ValueError(f"Unknown pooling_type: {pooling_type}")


class BranchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        logit = self.classifier(hidden).squeeze(-1)
        return hidden, logit


class BranchFusionModel(nn.Module):
    """Shared model for pooling comparison and modality ablation experiments."""

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 64,
        dropout: float = 0.4,
        pooling_type: str = "attention",
        tau: float = 0.5,
        topk: int = 3,
        active_branches: list[str] | tuple[str, ...] | None = None,
        normalize_pairs: bool = False,
        branch_tau: float = 1.5,
    ):
        super().__init__()
        self.pooling_type = pooling_type
        self.active_branches = tuple(active_branches or BRANCH_ORDER)
        self.branch_tau = branch_tau
        self.pool = build_pooling(pooling_type, tau=tau, topk=topk)

        unknown = set(self.active_branches) - set(BRANCH_ORDER)
        if unknown:
            raise ValueError(f"Unknown branches: {sorted(unknown)}")

        pair_dim = 4 * d_model
        self.pair_norm = nn.LayerNorm(pair_dim) if normalize_pairs else nn.Identity()
        self.branches = nn.ModuleDict({
            name: BranchMLP(pair_dim, hidden_dim, dropout)
            for name in BRANCH_ORDER
        })
        self.branch_attention = nn.Linear(hidden_dim, 1)
        self.fuse_out = nn.Linear(hidden_dim, 1)

    def _pool_sequence(
        self,
        seq_embs: torch.Tensor,
        seq_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pooling_type in {"mean", "topk"}:
            return self.pool(seq_embs, seq_mask, None)
        return self.pool(seq_embs, seq_mask, query)

    def encode_pairs(self, batch: dict) -> dict[str, torch.Tensor]:
        title = batch["title_emb"]
        thumb = batch["thumb_emb"]
        stt_seq = batch["stt_embs"]
        stt_mask = batch["stt_mask"]
        key_seq = batch["kf_embs"]
        key_mask = batch["kf_mask"]

        stt_by_title, attn_s_t = self._pool_sequence(stt_seq, stt_mask, title)
        key_by_title, attn_k_t = self._pool_sequence(key_seq, key_mask, title)
        stt_by_thumb, attn_s_th = self._pool_sequence(stt_seq, stt_mask, thumb)
        key_by_thumb, attn_k_th = self._pool_sequence(key_seq, key_mask, thumb)

        pairs = {
            "tk": pair_features(title, key_by_title),
            "ts": pair_features(title, stt_by_title),
            "thk": pair_features(thumb, key_by_thumb),
            "ths": pair_features(thumb, stt_by_thumb),
        }
        pairs = {name: self.pair_norm(features) for name, features in pairs.items()}
        token_attention = {
            "s_t": attn_s_t,
            "k_t": attn_k_t,
            "s_th": attn_s_th,
            "k_th": attn_k_th,
        }
        return pairs, token_attention

    def forward(self, batch: dict) -> dict:
        pairs, token_attention = self.encode_pairs(batch)

        branch_logits = {}
        hidden_states = []
        for name in self.active_branches:
            hidden, logit = self.branches[name](pairs[name])
            branch_logits[name] = logit
            hidden_states.append(hidden)

        if not hidden_states:
            raise ValueError("At least one active branch is required.")

        hidden_stack = torch.stack(hidden_states, dim=1)
        scores = self.branch_attention(hidden_stack).squeeze(-1)
        alpha = F.softmax(scores / self.branch_tau, dim=1)
        fused_hidden = (hidden_stack * alpha.unsqueeze(-1)).sum(dim=1)
        fused_logit = self.fuse_out(fused_hidden).squeeze(-1)

        return {
            "branch_logits": branch_logits,
            "fused_logit": fused_logit,
            "fuse_logit": fused_logit,
            "branch_attention": alpha,
            "alpha": alpha,
            "branch_names": self.active_branches,
            "token_attention": token_attention,
        }


class PoolingComparisonModel(BranchFusionModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("active_branches", BRANCH_ORDER)
        kwargs.setdefault("normalize_pairs", False)
        super().__init__(*args, **kwargs)


class ModalityAblationModel(BranchFusionModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("pooling_type", "attention")
        kwargs.setdefault("normalize_pairs", True)
        super().__init__(*args, **kwargs)
