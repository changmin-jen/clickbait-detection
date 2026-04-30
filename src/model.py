from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import AttentionPooling, MeanPooling, TopKPooling, TopKSimPooling


def pair_features(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    return torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=-1)


class BranchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.logit = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.net(x)
        return z, self.logit(z).squeeze(-1)


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


class PoolingComparisonModel(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 64,
        dropout: float = 0.4,
        pooling_type: str = "attention",
        tau: float = 0.5,
        topk: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.pair_dim = 4 * d_model
        self.pooling_type = pooling_type
        self.pool = build_pooling(pooling_type, tau=tau, topk=topk)

        self.branch_tk = BranchMLP(self.pair_dim, hidden_dim, dropout)
        self.branch_ts = BranchMLP(self.pair_dim, hidden_dim, dropout)
        self.branch_thk = BranchMLP(self.pair_dim, hidden_dim, dropout)
        self.branch_ths = BranchMLP(self.pair_dim, hidden_dim, dropout)

        self.att_w = nn.Linear(hidden_dim, 1)
        self.fuse_out = nn.Linear(hidden_dim, 1)
        self.branch_tau = 1.5

    def _pool_for_query(self, seq: torch.Tensor, mask: torch.Tensor, query: torch.Tensor):
        if self.pooling_type in ["mean", "topk"]:
            return self.pool(seq, mask, None)
        return self.pool(seq, mask, query)

    def forward(self, batch: dict) -> dict:
        title = batch["title_emb"]
        thumb = batch["thumb_emb"]
        stt_seq = batch["stt_embs"]
        stt_mask = batch["stt_mask"]
        key_seq = batch["kf_embs"]
        key_mask = batch["kf_mask"]

        stt_by_title, attn_s_t = self._pool_for_query(stt_seq, stt_mask, title)
        key_by_title, attn_k_t = self._pool_for_query(key_seq, key_mask, title)
        stt_by_thumb, attn_s_th = self._pool_for_query(stt_seq, stt_mask, thumb)
        key_by_thumb, attn_k_th = self._pool_for_query(key_seq, key_mask, thumb)

        z_tk, s_tk = self.branch_tk(pair_features(title, key_by_title))
        z_ts, s_ts = self.branch_ts(pair_features(title, stt_by_title))
        z_thk, s_thk = self.branch_thk(pair_features(thumb, key_by_thumb))
        z_ths, s_ths = self.branch_ths(pair_features(thumb, stt_by_thumb))

        stacked = torch.stack([z_tk, z_ts, z_thk, z_ths], dim=1)
        att_scores = self.att_w(stacked).squeeze(-1)
        alpha = F.softmax(att_scores / self.branch_tau, dim=1)
        z_fuse = torch.sum(alpha.unsqueeze(-1) * stacked, dim=1)
        fused_logit = self.fuse_out(z_fuse).squeeze(-1)

        return {
            "branch_logits": {
                "tk": s_tk,
                "ts": s_ts,
                "thk": s_thk,
                "ths": s_ths,
            },
            "fused_logit": fused_logit,
            "fuse_logit": fused_logit,
            "branch_attention": alpha,
            "alpha": alpha,
            "token_attention": {
                "s_t": attn_s_t,
                "k_t": attn_k_t,
                "s_th": attn_s_th,
                "k_th": attn_k_th,
            },
        }


class ModalityAblationModel(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 64,
        dropout: float = 0.4,
        pooling_type: str = "attention",
        tau: float = 0.5,
        topk: int = 3,
        active_branches: list[str] | None = None,
    ):
        super().__init__()
        self.pair_dim = 4 * d_model
        self.pooling_type = pooling_type
        self.pool = build_pooling(pooling_type, tau=tau, topk=topk)
        self.active_branches = active_branches or ["tk", "ts", "thk", "ths"]

        self.branch_tk = BranchMLP(self.pair_dim, hidden_dim, dropout)
        self.branch_ts = BranchMLP(self.pair_dim, hidden_dim, dropout)
        self.branch_thk = BranchMLP(self.pair_dim, hidden_dim, dropout)
        self.branch_ths = BranchMLP(self.pair_dim, hidden_dim, dropout)

        self.pair_ln = nn.LayerNorm(self.pair_dim)
        self.att_w = nn.Linear(hidden_dim, 1)
        self.fuse_out = nn.Linear(hidden_dim, 1)

    def _pool_for_query(self, seq: torch.Tensor, mask: torch.Tensor, query: torch.Tensor):
        if self.pooling_type in ["mean", "topk"]:
            return self.pool(seq, mask, None)[0]
        return self.pool(seq, mask, query)[0]

    def encode_pairs(self, batch: dict) -> dict:
        title = batch["title_emb"]
        thumb = batch["thumb_emb"]
        stt_seq = batch["stt_embs"]
        stt_mask = batch["stt_mask"]
        key_seq = batch["kf_embs"]
        key_mask = batch["kf_mask"]

        stt_by_title = self._pool_for_query(stt_seq, stt_mask, title)
        stt_by_thumb = self._pool_for_query(stt_seq, stt_mask, thumb)
        key_by_title = self._pool_for_query(key_seq, key_mask, title)
        key_by_thumb = self._pool_for_query(key_seq, key_mask, thumb)

        return {
            "tk": self.pair_ln(pair_features(title, key_by_title)),
            "ts": self.pair_ln(pair_features(title, stt_by_title)),
            "thk": self.pair_ln(pair_features(thumb, key_by_thumb)),
            "ths": self.pair_ln(pair_features(thumb, stt_by_thumb)),
        }

    def forward(self, batch: dict) -> dict:
        pairs = self.encode_pairs(batch)
        branch_layers = {
            "tk": self.branch_tk,
            "ts": self.branch_ts,
            "thk": self.branch_thk,
            "ths": self.branch_ths,
        }

        branch_outputs = {}
        used_h = []
        for name in self.active_branches:
            h, logit = branch_layers[name](pairs[name])
            branch_outputs[name] = logit
            used_h.append(h)

        if not used_h:
            raise ValueError("No active branches selected.")

        hidden = torch.stack(used_h, dim=1)
        alpha = torch.softmax(self.att_w(hidden).squeeze(-1), dim=1)
        fused = (hidden * alpha.unsqueeze(-1)).sum(dim=1)
        fused_logit = self.fuse_out(fused).squeeze(-1)

        return {
            "branch_logits": branch_outputs,
            "fused_logit": fused_logit,
            "fuse_logit": fused_logit,
            "branch_attention": alpha,
            "alpha": alpha,
        }
