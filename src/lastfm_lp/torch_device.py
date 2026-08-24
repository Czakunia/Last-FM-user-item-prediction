"""Resolve training device: cuda → mps → cpu (or explicit override)."""

from __future__ import annotations

import torch


def resolve_torch_device(name: str | None = "auto") -> torch.device:
    """Pick compute device.

    ``name``:
      - ``auto`` / ``None`` / ``""``: CUDA if available, else MPS, else CPU
      - ``cuda`` / ``mps`` / ``cpu``: force that backend (fallback to CPU if missing)
    """

    key = (name or "auto").strip().lower()
    if key in {"auto", "best", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if key.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(key if ":" in key else "cuda")
        print("[device] CUDA requested but unavailable → cpu")
        return torch.device("cpu")
    if key == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[device] MPS requested but unavailable → cpu")
        return torch.device("cpu")
    if key == "cpu":
        return torch.device("cpu")
    print(f"[device] unknown '{name}' → auto")
    return resolve_torch_device("auto")
