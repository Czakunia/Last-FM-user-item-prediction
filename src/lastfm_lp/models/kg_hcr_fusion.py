"""KG-HCR fusion modules: relation-aware attention + gated residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lastfm_lp.models.mlp_decoder import pair_representation


class HistoryAttentionAggregator(nn.Module):
    """β_{u,j,i} over history items given [zu, zj, zi, e_ji]."""

    def __init__(self, embed_dim: int, pair_feat_dim: int, hidden: int = 64) -> None:
        super().__init__()
        in_dim = 3 * embed_dim + pair_feat_dim
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.out_dim = pair_feat_dim

    def forward(
        self,
        zu: torch.Tensor,  # [B, D]
        zi: torch.Tensor,  # [B, D]
        z_hist: torch.Tensor,  # [B, H, D]
        pair_feats: torch.Tensor,  # [B, H, F]
        mask: torch.Tensor,  # [B, H] 1=valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, D = z_hist.shape
        zu_e = zu.unsqueeze(1).expand(-1, H, -1)
        zi_e = zi.unsqueeze(1).expand(-1, H, -1)
        x = torch.cat([zu_e, z_hist, zi_e, pair_feats], dim=-1)
        logits = self.scorer(x).squeeze(-1)  # [B, H]
        logits = logits.masked_fill(mask <= 0, -1e9)
        beta = torch.softmax(logits, dim=-1)
        beta = beta * mask
        beta = beta / beta.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        agg = torch.sum(beta.unsqueeze(-1) * pair_feats, dim=1)  # [B, F]
        return agg, beta


class GatedKGHCRModel(nn.Module):
    """s = s_KG + g * s_KG-HCR  (F4)."""

    def __init__(
        self,
        embed_dim: int,
        kg_hcr_dim: int,
        gate_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        pair_dim = 4 * embed_dim

        def mlp(in_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = in_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        self.decoder_kg = mlp(pair_dim)
        self.decoder_hcr = mlp(kg_hcr_dim)
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        zu: torch.Tensor,
        zi: torch.Tensor,
        kg_hcr_vec: torch.Tensor,
        gate_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_kg = self.decoder_kg(pair_representation(zu, zi)).squeeze(-1)
        s_hcr = self.decoder_hcr(kg_hcr_vec).squeeze(-1)
        g = torch.sigmoid(self.gate(gate_feats).squeeze(-1))
        return s_kg + g * s_hcr, s_kg, g
