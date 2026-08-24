"""KAN tabular predictor using efficient KANLinear (thesis-compatible)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.lastfm_lp.models.kan_linear import KANLinear
from src.lastfm_lp.models.mlp_predictor import TrainConfig


class KANNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, grid_size: int, spline_order: int) -> None:
        super().__init__()
        self.kan1 = KANLinear(in_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order, grid_update=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.kan2 = KANLinear(hidden_dim, 1, grid_size=grid_size, spline_order=spline_order, grid_update=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kan1(x)
        x = self.norm1(x)
        x = self.drop(x)
        x = self.kan2(x).squeeze(-1)
        return x


class KANPredictor:
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float,
        grid_size: int,
        spline_order: int,
        cfg: TrainConfig,
    ) -> None:
        self.scaler = StandardScaler()
        self.cfg = cfg
        self.device = torch.device("cpu")
        torch.manual_seed(cfg.seed)
        self.model = KANNet(in_dim, hidden_dim, dropout, grid_size, spline_order).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        Xv = self.scaler.transform(X_val).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y.astype(np.float32)))
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()
        best_state = None
        best_val = float("inf")
        patience_left = self.cfg.patience
        history = []
        for epoch in range(self.cfg.max_epochs):
            self.model.train()
            total = 0.0
            n = 0
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(xb)
                n += len(xb)
            self.model.eval()
            with torch.no_grad():
                vlogits = self.model(torch.from_numpy(Xv))
                vloss = float(loss_fn(vlogits, torch.from_numpy(y_val.astype(np.float32))).item())
            history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_loss": vloss})
            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"history": history, "best_val_loss": best_val, "epochs_ran": len(history)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xs = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(Xs))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float64)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))
