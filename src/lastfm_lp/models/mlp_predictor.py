"""MLP tabular predictor (Stage A/B)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class MLPNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 40
    batch_size: int = 4096
    patience: int = 5
    seed: int = 2026


class MLPPredictor:
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float, cfg: TrainConfig) -> None:
        self.scaler = StandardScaler()
        self.cfg = cfg
        self.device = torch.device("cpu")
        torch.manual_seed(cfg.seed)
        self.model = MLPNet(in_dim, hidden_dims, dropout).to(self.device)
        self.in_dim = in_dim

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
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(xb)
                n += len(xb)
            self.model.eval()
            with torch.no_grad():
                vlogits = self.model(torch.from_numpy(Xv).to(self.device))
                vloss = float(loss_fn(vlogits, torch.from_numpy(y_val.astype(np.float32)).to(self.device)).item())
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
            logits = self.model(torch.from_numpy(Xs).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float64)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))
