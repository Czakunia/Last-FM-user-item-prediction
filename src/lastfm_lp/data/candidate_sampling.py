"""Candidate pair sampling with leakage-safe negatives."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def sample_negatives_for_user(
    *,
    user_id: int,
    n_items: int,
    n_neg: int,
    forbidden: set[int],
    popularity: np.ndarray | None = None,
    mode: str = "random",
    rng: np.random.Generator,
) -> list[int]:
    """Sample negative item ids not in forbidden set."""
    if n_neg <= 0:
        return []
    if mode == "popularity_matched" and popularity is not None:
        weights = popularity.astype(np.float64) + 1.0
        for i in forbidden:
            if 0 <= i < n_items:
                weights[i] = 0.0
        total = weights.sum()
        if total <= 0:
            mode = "random"
        else:
            probs = weights / total
            # with replacement then unique fill
            chosen: list[int] = []
            seen = set(forbidden)
            tries = 0
            while len(chosen) < n_neg and tries < n_neg * 20:
                i = int(rng.choice(n_items, p=probs))
                tries += 1
                if i in seen:
                    continue
                seen.add(i)
                chosen.append(i)
            if len(chosen) >= n_neg:
                return chosen[:n_neg]
            mode = "random"

    # random (default / fallback). kg_hard reserved for later stages.
    chosen = []
    seen = set(forbidden)
    while len(chosen) < n_neg:
        i = int(rng.integers(0, n_items))
        if i in seen:
            continue
        seen.add(i)
        chosen.append(i)
    return chosen


def build_pair_table(
    *,
    users: Iterable[int],
    positives_by_user: dict[int, set[int]],
    history_by_user: dict[int, set[int]],
    n_items: int,
    n_neg_per_pos: int,
    max_positives_per_user: int | None,
    popularity: np.ndarray,
    mode: str,
    seed: int,
    split_name: str,
) -> dict[str, np.ndarray]:
    """Build arrays: user_id, item_id, label, split."""
    rng = np.random.default_rng(seed + hash(split_name) % 10_000)
    users_out: list[int] = []
    items_out: list[int] = []
    labels_out: list[int] = []

    for u in users:
        pos = sorted(positives_by_user.get(u, ()))
        if not pos:
            continue
        if max_positives_per_user is not None and len(pos) > max_positives_per_user:
            pos = rng.choice(pos, size=max_positives_per_user, replace=False).tolist()
            pos = sorted(int(x) for x in pos)
        hist = history_by_user.get(u, set())
        forbidden = set(hist) | set(positives_by_user.get(u, set()))
        # also forbid official other positives already known? keep simple: hist ∪ all split pos
        for p in pos:
            users_out.append(u)
            items_out.append(int(p))
            labels_out.append(1)
            negs = sample_negatives_for_user(
                user_id=u,
                n_items=n_items,
                n_neg=n_neg_per_pos,
                forbidden=forbidden,
                popularity=popularity,
                mode=mode,
                rng=rng,
            )
            forbidden.update(negs)
            for n in negs:
                users_out.append(u)
                items_out.append(int(n))
                labels_out.append(0)

    return {
        "user_id": np.asarray(users_out, dtype=np.int32),
        "item_id": np.asarray(items_out, dtype=np.int32),
        "label": np.asarray(labels_out, dtype=np.int8),
    }
