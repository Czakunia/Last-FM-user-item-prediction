"""Common graph encoder contract for architecture benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class EncoderOutput:
    user_embedding: Tensor
    candidate_embedding: Tensor
    graph_context: Tensor
    graph_score: Tensor


class BaseGraphEncoder(nn.Module, ABC):
    """Encoders must only consume graph topology — never HCR features."""

    embed_dim: int

    @abstractmethod
    def encode_all(self) -> Tensor:
        """Return flat node embeddings [n_nodes, d] (user | entity layout)."""

    def pair_outputs(
        self,
        z: Tensor,
        user_idx: Tensor,
        item_idx: Tensor,
    ) -> EncoderOutput:
        zu = z[user_idx]
        zi = z[item_idx]
        graph_context = torch.cat([zu, zi, zu * zi, (zu - zi).abs()], dim=-1)
        graph_score = (zu * zi).sum(dim=-1)
        return EncoderOutput(
            user_embedding=zu,
            candidate_embedding=zi,
            graph_context=graph_context,
            graph_score=graph_score,
        )

    @property
    def context_dim(self) -> int:
        return 4 * self.embed_dim
