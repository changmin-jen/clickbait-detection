from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset


class ClickbaitEmbeddingDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "id": sample["id"],
            "label": torch.tensor(float(sample["label_id"]), dtype=torch.float32),
            "title_emb": sample["title_emb"].float(),
            "thumb_emb": sample["thumb_emb"].float(),
            "stt_embs": sample["stt_embs"].float(),
            "kf_embs": sample["kf_embs"].float(),
        }


def load_embedding_splits(pt_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    loaded = torch.load(pt_path, map_location="cpu")
    samples = loaded["samples"]
    train_samples = [x for x in samples if x["split"] == "train"]
    valid_samples = [x for x in samples if x["split"] == "valid"]
    test_samples = [x for x in samples if x["split"] == "test"]
    return train_samples, valid_samples, test_samples


def pad_sequence_2d(seq_list: Iterable[torch.Tensor], dim: int = 768) -> tuple[torch.Tensor, torch.Tensor]:
    seq_list = list(seq_list)
    batch_size = len(seq_list)
    max_len = max([x.shape[0] for x in seq_list]) if seq_list else 0

    padded = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, seq in enumerate(seq_list):
        if seq.shape[0] > 0:
            length = seq.shape[0]
            padded[i, :length] = seq
            mask[i, :length] = True

    return padded, mask


def collate_fn(batch: list[dict]) -> dict:
    title_emb = torch.stack([x["title_emb"] for x in batch], dim=0)
    thumb_emb = torch.stack([x["thumb_emb"] for x in batch], dim=0)
    labels = torch.stack([x["label"] for x in batch], dim=0)

    stt_padded, stt_mask = pad_sequence_2d(
        [x["stt_embs"] for x in batch],
        dim=title_emb.shape[-1],
    )
    kf_padded, kf_mask = pad_sequence_2d(
        [x["kf_embs"] for x in batch],
        dim=title_emb.shape[-1],
    )

    return {
        "title_emb": title_emb,
        "thumb_emb": thumb_emb,
        "stt_embs": stt_padded,
        "stt_mask": stt_mask,
        "kf_embs": kf_padded,
        "kf_mask": kf_mask,
        "labels": labels,
        "ids": [x["id"] for x in batch],
    }


def create_dataloaders(
    train_samples: list[dict],
    valid_samples: list[dict],
    test_samples: list[dict],
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        ClickbaitEmbeddingDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        ClickbaitEmbeddingDataset(valid_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        ClickbaitEmbeddingDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader, test_loader
