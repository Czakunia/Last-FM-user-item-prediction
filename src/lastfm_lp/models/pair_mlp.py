"""Shared PairMLP encoder for compact binary blocks (Stage H4 / HT3)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PairMLP(nn.Module):
    """Maps h(j,i) ∈ R^{d_in} → e_ji ∈ R^{d_out} with GELU + LayerNorm."""

    def __init__(
        self,
        in_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims or [16, 16])
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.GELU(),
                nn.LayerNorm(h),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pool_encoded(
    encoded: torch.Tensor,
    mask: torch.Tensor,
    support: torch.Tensor | None = None,
    top_k: int = 3,
) -> torch.Tensor:
    """Pool [B, H, D] with mask [B, H] → [B, 4D] (mean,max,top3,wmean)."""

    B, H, D = encoded.shape
    mask_f = mask.float()
    # mean
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (encoded * mask_f.unsqueeze(-1)).sum(dim=1) / denom
    # max (masked)
    neg_inf = torch.finfo(encoded.dtype).min
    masked_for_max = encoded.masked_fill(mask_f.unsqueeze(-1) < 0.5, neg_inf)
    mx = masked_for_max.max(dim=1).values
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    # top-k by first channel of encoded
    key = encoded[..., 0]
    key = key.masked_fill(mask_f < 0.5, neg_inf)
    k = min(top_k, H)
    top_idx = key.topk(k, dim=1).indices  # [B, k]
    gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, D)
    top_rows = torch.gather(encoded, 1, gather_idx)
    top_mask = torch.gather(mask_f, 1, top_idx)
    top_denom = top_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    top_mean = (top_rows * top_mask.unsqueeze(-1)).sum(dim=1) / top_denom
    # support-weighted
    if support is None:
        w = mask_f
    else:
        w = support.float() * mask_f
    wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    wmean = (encoded * w.unsqueeze(-1)).sum(dim=1) / wsum
    return torch.cat([mean, mx, top_mean, wmean], dim=-1)
