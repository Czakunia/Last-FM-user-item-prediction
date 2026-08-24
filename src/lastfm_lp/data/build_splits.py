"""Per-user validation holdout from train; original test preserved."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.lastfm_lp.data.load_kgat_lastfm import LastFMData


def build_per_user_splits(
    data: LastFMData,
    *,
    validation_ratio: float = 0.1,
    min_train_items: int = 5,
    seed: int = 2026,
) -> dict[str, Any]:
    """For each user, hold out a fraction of train items as validation.

    Remaining items become model_train history. Official test set untouched.
    """
    rng = np.random.default_rng(seed)
    model_train: dict[int, set[int]] = {}
    valid: dict[int, set[int]] = {}
    skipped = 0

    for u, items in data.train_by_user.items():
        items_list = sorted(items)
        if len(items_list) < min_train_items:
            model_train[u] = set(items_list)
            valid[u] = set()
            skipped += 1
            continue
        n_val = max(1, int(round(len(items_list) * validation_ratio)))
        n_val = min(n_val, len(items_list) - min_train_items + 1)
        n_val = max(1, n_val)
        chosen = set(rng.choice(items_list, size=n_val, replace=False).tolist())
        valid[u] = chosen
        model_train[u] = set(items_list) - chosen

    test = {u: set(v) for u, v in data.test_by_user.items()}
    return {
        "model_train": model_train,
        "valid": valid,
        "test": test,
        "meta": {
            "n_users": data.n_users,
            "n_items": data.n_items,
            "validation_ratio": validation_ratio,
            "min_train_items": min_train_items,
            "seed": seed,
            "users_without_val_holdout": skipped,
            "n_train_interactions": int(sum(len(v) for v in model_train.values())),
            "n_valid_interactions": int(sum(len(v) for v in valid.values())),
            "n_test_interactions": int(sum(len(v) for v in test.values())),
        },
    }


def select_eval_users(
    splits: dict[str, Any],
    *,
    n_eval_users: int | None,
    seed: int = 2026,
) -> list[int]:
    """Select eval users. ``n_eval_users=None`` → all eligible users (FULL_SOTA)."""
    rng = np.random.default_rng(seed)
    candidates = [
        u
        for u, items in splits["model_train"].items()
        if len(items) >= 1
        and (len(splits["valid"].get(u, ())) > 0 or len(splits["test"].get(u, ())) > 0)
    ]
    candidates = sorted(candidates)
    if n_eval_users is None or len(candidates) <= int(n_eval_users):
        return candidates
    pick = rng.choice(candidates, size=int(n_eval_users), replace=False)
    return sorted(int(u) for u in pick)


def save_splits(splits: dict[str, Any], out_dir: str | Path, eval_users: list[int]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _dump_user_sets(mapping: dict[int, set[int]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for u in sorted(mapping):
                items = " ".join(str(i) for i in sorted(mapping[u]))
                f.write(f"{u} {items}\n" if items else f"{u}\n")

    _dump_user_sets(splits["model_train"], out_dir / "model_train.txt")
    _dump_user_sets(splits["valid"], out_dir / "valid.txt")
    _dump_user_sets(splits["test"], out_dir / "test.txt")
    (out_dir / "eval_users.json").write_text(
        json.dumps({"eval_users": eval_users}, indent=2), encoding="utf-8"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(splits["meta"], indent=2), encoding="utf-8"
    )


def load_user_sets(path: Path) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            u = int(parts[0])
            out[u] = {int(x) for x in parts[1:]}
    return out
