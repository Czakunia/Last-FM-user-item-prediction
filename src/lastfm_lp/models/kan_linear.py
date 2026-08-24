"""Efficient KANLinear: base SiLU path + learnable B-spline path.

Adapted from the efficient-kan reformulation (Blealtan et al.): B-spline
bases are evaluated per input channel, then combined with a linear map.
This avoids expanding to (batch, out, in) activation tensors.

Wave 9 constraints:
- float32
- grid_update=False in K1 (valid/test must not mutate knots)
- grid_range default [-3, 3] to match HCR slot audit
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Sequence[float] = (-3.0, 3.0),
        grid_update: bool = False,
        base_activation: str = "silu",
        base_scale_init: float = 1.0,
        spline_scale_init: float = 0.1,
        use_bias: bool = True,
        scale_noise: float = 0.1,
        grid_eps: float = 0.02,
        enable_standalone_scale_spline: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in/out features must be positive")
        if grid_size < 1:
            raise ValueError("grid_size must be >= 1")
        if spline_order < 1:
            raise ValueError("spline_order must be >= 1")
        if len(grid_range) != 2 or float(grid_range[0]) >= float(grid_range[1]):
            raise ValueError(f"invalid grid_range {grid_range}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.grid_update = bool(grid_update)
        self.scale_noise = float(scale_noise)
        self.scale_base = float(base_scale_init)
        self.scale_spline = float(spline_scale_init)
        self.enable_standalone_scale_spline = bool(enable_standalone_scale_spline)
        self.grid_eps = float(grid_eps)

        act = str(base_activation).lower()
        if act == "silu":
            self.base_activation: nn.Module = nn.SiLU()
        elif act == "gelu":
            self.base_activation = nn.GELU()
        else:
            raise ValueError(f"unsupported base_activation {base_activation}")

        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = (
            (
                torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1)
                * h
                + self.grid_range[0]
            )
            .expand(self.in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.in_features,
                self.grid_size + self.spline_order,
            )
        )
        if self.enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.empty(self.out_features, self.in_features)
            )
        else:
            self.register_parameter("spline_scaler", None)

        self.bias = nn.Parameter(torch.zeros(self.out_features)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(
                        self.grid_size + 1,
                        self.in_features,
                        self.out_features,
                    )
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.spline_scaler is not None:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, : -(k + 1)]
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_num = grid[:, k + 1 :] - x
            right_den = grid[:, k + 1 :] - grid[:, 1:(-k)]
            bases = (left_num / left_den.clamp_min(1e-12)) * bases[:, :, :-1] + (
                right_num / right_den.clamp_min(1e-12)
            ) * bases[:, :, 1:]
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(a, b).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.spline_scaler is None:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.forward_parts(x)
        return out

    def forward_parts(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (total, base_output, spline_output) — bias included in total only."""
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"expected last dim {self.in_features}, got {x.size(-1)}"
            )
        original_shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x2), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x2).view(x2.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        out = base_output + spline_output
        if self.bias is not None:
            out = out + self.bias
        shape = (*original_shape[:-1], self.out_features)
        return (
            out.reshape(*shape),
            base_output.reshape(*shape),
            spline_output.reshape(*shape),
        )

    @torch.no_grad()
    def update_grid_from_data(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Adaptive knot update (K2). No-op when grid_update=False."""
        if not self.grid_update:
            return
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced))

    def regularization_loss(self) -> torch.Tensor:
        """L1 on spline coefficients (efficient-kan style weight L1)."""
        return self.spline_weight.abs().mean()

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor | None = None) -> dict[str, float]:
        out = {
            "base_weight_norm": float(self.base_weight.detach().norm()),
            "spline_coefficient_norm": float(self.spline_weight.detach().norm()),
            "max_abs_spline_coefficient": float(self.spline_weight.detach().abs().max()),
            "fraction_zero_spline_coefficients": float(
                (self.spline_weight.detach().abs() < 1e-12).float().mean()
            ),
        }
        if x is not None and x.numel():
            x2 = x.reshape(-1, self.in_features)
            lo, hi = self.grid_range
            frac = float(((x2 < lo) | (x2 > hi)).float().mean())
            out["input_out_of_grid_fraction"] = frac
            out["nan_count"] = float(torch.isnan(x2).sum())
            out["inf_count"] = float(torch.isinf(x2).sum())
        return out
