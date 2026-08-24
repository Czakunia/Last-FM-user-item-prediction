"""Efficient KG neighborhood index for item–item path descriptors.

Uses official relation_list.txt remap ids (Last-FM: 9 Freebase music relations).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

# Frozen structural path classes (names only; relation ids come from data)
PATH_CLASSES = (
    "shared_entity_two_hop",
    "relational_three_hop",
    "no_path",
)


class KGPathIndex:
    def __init__(self, kg: pd.DataFrame, n_entities: int, n_relations: int) -> None:
        self.n_entities = int(n_entities)
        self.n_relations = int(n_relations)
        # undirected neighbors
        self.neighbors: list[set[int]] = [set() for _ in range(n_entities)]
        # neighbors by relation (undirected: both endpoints see the edge under r)
        self.nbr_by_rel: list[list[set[int]]] = [
            [set() for _ in range(n_relations)] for _ in range(n_entities)
        ]
        for h, r, t in zip(
            kg["head"].to_numpy(), kg["relation"].to_numpy(), kg["tail"].to_numpy()
        ):
            h, r, t = int(h), int(r), int(t)
            if not (0 <= h < n_entities and 0 <= t < n_entities):
                continue
            if h == t:
                continue
            if not (0 <= r < n_relations):
                continue
            self.neighbors[h].add(t)
            self.neighbors[t].add(h)
            self.nbr_by_rel[h][r].add(t)
            self.nbr_by_rel[t][r].add(h)

        self._cache: dict[tuple[int, int], dict[str, float]] = {}

    def pair_features(self, j: int, i: int) -> dict[str, float]:
        """KG descriptors for ordered history item j and candidate i."""
        j, i = int(j), int(i)
        if j == i:
            return self._empty(connected=True, shortest=0.0)
        key = (j, i) if j < i else (i, j)
        # directed cache key for asymmetric fields; shared topology is symmetric
        dkey = (j, i)
        if dkey in self._cache:
            return self._cache[dkey]

        nj = self.neighbors[j]
        ni = self.neighbors[i]
        shared = nj & ni
        shared_count = len(shared)
        # direct edge?
        direct = i in nj
        connected = direct or shared_count > 0

        # 3-hop proxy only if no 1/2-hop (cost control)
        path3 = 0
        if shared_count == 0 and not direct:
            # sample-limited: |N(a) ∩ N(i)| over neighbors a of j
            for a in list(nj)[:40]:
                if a == i:
                    continue
                inter_n = len(self.neighbors[a] & ni)
                if j in self.neighbors[a] and j in ni:
                    inter_n = max(inter_n - 1, 0)
                path3 += inter_n
                if path3 > 1000:
                    path3 = 1000
                    break

        # relation overlap / multihot on 2-hop context
        rel_hit = np.zeros(self.n_relations, dtype=np.float32)
        overlap = 0.0
        for r in range(self.n_relations):
            # shared entities reachable via relation r from either side
            sj = self.nbr_by_rel[j][r]
            si = self.nbr_by_rel[i][r]
            # same-relation shared neighbor OR complementary roles into shared entity
            inter_r = sj & si
            # also: entity e in shared where j-e or i-e used any rel — mark rels used into shared
            if inter_r or (shared and (sj & shared or si & shared)):
                rel_hit[r] = 1.0
                overlap += 1.0
        # refine: for each shared entity, mark relations used by j and by i
        for e in list(shared)[:64]:
            for r in range(self.n_relations):
                if e in self.nbr_by_rel[j][r] or e in self.nbr_by_rel[i][r]:
                    rel_hit[r] = 1.0

        if direct:
            shortest = 1.0
        elif shared_count > 0:
            shortest = 2.0
        elif path3 > 0:
            shortest = 3.0
            connected = True
        else:
            shortest = 0.0  # sentinel: no short path
            connected = False

        out: dict[str, float] = {
            "kg_connected": 1.0 if connected else 0.0,
            "kg_shortest_path": shortest if connected else 0.0,
            "kg_path_count_len2": float(shared_count),
            "kg_path_count_len3": float(min(path3, 5000)),
            "kg_shared_entity_count": float(shared_count),
            "kg_relation_overlap": float(rel_hit.sum()),
            "log1p_path_count_len2": float(np.log1p(shared_count)),
            "log1p_path_count_len3": float(np.log1p(min(path3, 5000))),
        }
        for r in range(self.n_relations):
            out[f"kg_rel_{r}"] = float(rel_hit[r])

        self._cache[dkey] = out
        return out

    def _empty(self, *, connected: bool, shortest: float) -> dict[str, float]:
        out = {
            "kg_connected": 1.0 if connected else 0.0,
            "kg_shortest_path": shortest,
            "kg_path_count_len2": 0.0,
            "kg_path_count_len3": 0.0,
            "kg_shared_entity_count": 0.0,
            "kg_relation_overlap": 0.0,
            "log1p_path_count_len2": 0.0,
            "log1p_path_count_len3": 0.0,
        }
        for r in range(self.n_relations):
            out[f"kg_rel_{r}"] = 0.0
        return out

    @property
    def feature_names(self) -> list[str]:
        return [
            "kg_connected",
            "kg_shortest_path",
            "kg_path_count_len2",
            "kg_path_count_len3",
            "kg_shared_entity_count",
            "kg_relation_overlap",
            "log1p_path_count_len2",
            "log1p_path_count_len3",
        ] + [f"kg_rel_{r}" for r in range(self.n_relations)]
