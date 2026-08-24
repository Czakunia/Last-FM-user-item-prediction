"""Load KGAT Last-FM working copy into memory structures."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LastFMData:
    path: Path
    n_users: int
    n_items: int
    n_entities: int
    n_relations: int
    train_by_user: dict[int, set[int]]
    test_by_user: dict[int, set[int]]
    train_pairs: np.ndarray  # (N,2) user,item
    test_pairs: np.ndarray
    kg: pd.DataFrame
    item_kg_degree: np.ndarray
    relation_counts: dict[int, int]
    relation_names: dict[int, str]


def _read_interactions(path: Path) -> tuple[dict[int, set[int]], np.ndarray]:
    by_user: dict[int, set[int]] = {}
    pairs: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            items = {int(x) for x in parts[1:]}
            by_user[u] = items
            for i in items:
                pairs.append((u, i))
    arr = np.asarray(pairs, dtype=np.int32)
    return by_user, arr


def load_lastfm(data_dir: str | Path) -> LastFMData:
    data_dir = Path(data_dir)
    train_by_user, train_pairs = _read_interactions(data_dir / "train.txt")
    test_by_user, test_pairs = _read_interactions(data_dir / "test.txt")

    users = pd.read_csv(data_dir / "user_list.txt", sep=r"\s+", engine="python")
    items = pd.read_csv(data_dir / "item_list.txt", sep=r"\s+", engine="python")
    # entity_list can have broken spaces — count remaps robustly
    ent_ids = []
    with (data_dir / "entity_list.txt").open("r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    ent_ids.append(int(parts[-1]))
                except ValueError:
                    continue
    relations = pd.read_csv(data_dir / "relation_list.txt", sep=r"\s+", engine="python")
    kg = pd.read_csv(
        data_dir / "kg_final.txt",
        sep=r"\s+",
        header=None,
        names=["head", "relation", "tail"],
        dtype={"head": np.int32, "relation": np.int32, "tail": np.int32},
    )

    n_users = int(users["remap_id"].nunique())
    n_items = int(items["remap_id"].nunique())
    n_entities = int(len(set(ent_ids)))
    n_relations = int(relations["remap_id"].nunique())

    deg = np.zeros(n_entities, dtype=np.int32)
    for h, t in zip(kg["head"].to_numpy(), kg["tail"].to_numpy()):
        if 0 <= h < n_entities:
            deg[h] += 1
        if 0 <= t < n_entities:
            deg[t] += 1
    item_kg_degree = deg[:n_items].astype(np.int32)

    rel_counts = kg["relation"].value_counts().to_dict()
    rel_names = {
        int(r): str(n).split("/")[-1]
        for r, n in zip(relations["remap_id"], relations["org_id"])
    }

    return LastFMData(
        path=data_dir,
        n_users=n_users,
        n_items=n_items,
        n_entities=n_entities,
        n_relations=n_relations,
        train_by_user=train_by_user,
        test_by_user=test_by_user,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        kg=kg,
        item_kg_degree=item_kg_degree,
        relation_counts={int(k): int(v) for k, v in rel_counts.items()},
        relation_names=rel_names,
    )


def item_popularity(train_by_user: dict[int, set[int]], n_items: int) -> np.ndarray:
    pop = np.zeros(n_items, dtype=np.int32)
    for items in train_by_user.values():
        for i in items:
            if 0 <= i < n_items:
                pop[i] += 1
    return pop
