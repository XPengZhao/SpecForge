"""Token-ID memory for the DSpark Markov head. No target-side capture is required."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class EngramValueProjection(nn.Linear):
    """Keep the initial Engram residual zero, including after HF post_init."""


class EngramMemory(nn.Module):
    def __init__(self, config, hidden_size, markov_rank, eps):
        super().__init__()
        if config.get("hash_version") != "xor_v1":
            raise ValueError("Unsupported Engram hash_version")
        self.heads_per_order = config.get("heads_per_order", 2)
        if type(self.heads_per_order) is not int or self.heads_per_order < 1:
            raise ValueError("Engram heads_per_order must be a positive integer")
        self.bucket_counts = tuple(config["bucket_counts"])
        self.multipliers = tuple(config["hash_multipliers"])
        if (
            len(self.bucket_counts) != 2 * self.heads_per_order
            or len(set(self.bucket_counts)) != len(self.bucket_counts)
            or any(type(n) is not int or n < 2 for n in self.bucket_counts)
        ):
            raise ValueError(
                "Engram requires one distinct bucket count >= 2 per hash head"
            )
        if len(self.multipliers) != 3 or any(
            type(n) is not int or n <= 0 or n % 2 == 0 for n in self.multipliers
        ):
            raise ValueError("Engram requires three positive odd hash multipliers")
        head_dim = config["head_dim"]
        if type(head_dim) is not int or head_dim <= 0:
            raise ValueError("Engram head_dim must be a positive integer")
        width = len(self.bucket_counts) * head_dim
        self.tables = nn.ModuleList(
            nn.Embedding(n, head_dim) for n in self.bucket_counts
        )
        self.query = nn.Linear(hidden_size, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = EngramValueProjection(width, markov_rank, bias=False)
        nn.init.zeros_(self.value.weight)
        self.query_norm = nn.RMSNorm(width, eps=eps)
        self.key_norm = nn.RMSNorm(width, eps=eps)
        self.scale = 1 / math.sqrt(width)

    def bucket_ids(self, input_ids):
        """[batch, sequence, 2 * heads_per_order]; -1 denotes unavailable history for that order."""
        ids = input_ids.long()
        padded = F.pad(ids, (2, 0), value=-1)
        older, prev = padded[:, :-2], padded[:, 1:-1]
        m0, m1, m2 = self.multipliers  # newest to oldest, signed int64
        bigram = (ids * m0) ^ (prev * m1)
        trigram = bigram ^ (older * m2)
        heads = []
        for i, count in enumerate(self.bucket_counts):
            mixed, valid = (
                (bigram, prev >= 0)
                if i < self.heads_per_order
                else (trigram, (older >= 0) & (prev >= 0))
            )
            heads.append(torch.where(valid & (ids >= 0), mixed.remainder(count), -1))
        return torch.stack(heads, dim=-1)

    def forward(self, bucket_ids, hidden_states):
        chunks = []
        for index, table in enumerate(self.tables):
            ids = bucket_ids[..., index]
            row = table(ids.clamp_min(0))
            chunks.append(row * (ids >= 0).unsqueeze(-1))
        memory = torch.cat(chunks, dim=-1)
        q = self.query_norm(self.query(hidden_states))
        k = self.key_norm(self.key(memory))
        gate = ((q.float() * k.float()).sum(-1, keepdim=True) * self.scale).sigmoid()
        value = self.value(memory)
        return gate.to(value.dtype) * value
