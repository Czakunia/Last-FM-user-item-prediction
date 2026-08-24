"""Build undirected CKG edge_index: users ∪ entities, interact + KG edges."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.lastfm_lp.data.load_kgat_lastfm import LastFMData, load_lastfm


def build_ckg_edge_index(
    data: LastFMData,
    model_train: dict[int, set[int]],
    *,
    max_kg_edges: int | None = None,
    seed: int = 2026,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Node layout: [0, n_users) users; [n_users, n_users+n_entities) entities/items.

    Item i maps to node n_users + i.
    """
    n_users = data.n_users
    n_entities = data.n_entities
    n_nodes = n_users + n_entities

    src: list[int] = []
    dst: list[int] = []

    # user–item interactions (undirected)
    for u, items in model_train.items():
        for i in items:
            if 0 <= i < data.n_items:
                a, b = int(u), n_users + int(i)
                src.extend([a, b])
                dst.extend([b, a])

    # KG triplets (undirected for GraphSAGE)
    kg = data.kg
    h = kg["head"].to_numpy()
    t = kg["tail"].to_numpy()
    if max_kg_edges is not None and len(h) > max_kg_edges:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(h), size=max_kg_edges, replace=False)
        h, t = h[idx], t[idx]
    for hi, ti in zip(h, t):
        a, b = n_users + int(hi), n_users + int(ti)
        if a == b:
            continue
        src.extend([a, b])
        dst.extend([b, a])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    meta = {
        "n_users": n_users,
        "n_items": data.n_items,
        "n_entities": n_entities,
        "n_nodes": n_nodes,
        "n_edges_undirected_stubs": int(edge_index.size(1)),
        "item_offset": n_users,
    }
    return edge_index, meta


def user_item_to_nodes(
    users: np.ndarray,
    items: np.ndarray,
    item_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.from_numpy(users.astype(np.int64))
    i = torch.from_numpy(items.astype(np.int64)) + int(item_offset)
    return u, i


def build_typed_ckg(
    data: LastFMData,
    model_train: dict[int, set[int]],
    *,
    max_kg_edges: int | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    """Typed directed CKG for R-GCN / HGT.

    Relation IDs (R-GCN):
      0: user→item interact
      1: item→user interact_rev
      2+r: KG relation r (head→tail)
      2+n_rel+r: KG relation r reverse (tail→head)

    Hetero edge keys for HGT use local ids within node type.
    """
    n_users = data.n_users
    n_entities = data.n_entities
    n_rel = data.n_relations
    n_nodes = n_users + n_entities

    src: list[int] = []
    dst: list[int] = []
    etype: list[int] = []

    # hetero local indices
    ui_src, ui_dst = [], []
    iu_src, iu_dst = [], []
    kg_edges: dict[str, tuple[list[int], list[int]]] = {}

    for u, items in model_train.items():
        for i in items:
            if 0 <= i < data.n_items:
                u_i, e_i = int(u), int(i)
                # flat
                src.extend([u_i, n_users + e_i])
                dst.extend([n_users + e_i, u_i])
                etype.extend([0, 1])
                # hetero
                ui_src.append(u_i)
                ui_dst.append(e_i)
                iu_src.append(e_i)
                iu_dst.append(u_i)

    h = data.kg["head"].to_numpy()
    r = data.kg["relation"].to_numpy()
    t = data.kg["tail"].to_numpy()
    if max_kg_edges is not None and len(h) > max_kg_edges:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(h), size=max_kg_edges, replace=False)
        h, r, t = h[idx], r[idx], t[idx]

    for hi, ri, ti in zip(h, r, t):
        hi, ri, ti = int(hi), int(ri), int(ti)
        if hi == ti:
            continue
        a, b = n_users + hi, n_users + ti
        src.extend([a, b])
        dst.extend([b, a])
        etype.extend([2 + ri, 2 + n_rel + ri])
        key = f"rel_{ri}"
        key_rev = f"rel_{ri}_rev"
        if key not in kg_edges:
            kg_edges[key] = ([], [])
            kg_edges[key_rev] = ([], [])
        kg_edges[key][0].append(hi)
        kg_edges[key][1].append(ti)
        kg_edges[key_rev][0].append(ti)
        kg_edges[key_rev][1].append(hi)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    n_relations = 2 + 2 * n_rel

    edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {
        ("user", "interact", "entity"): torch.tensor([ui_src, ui_dst], dtype=torch.long),
        ("entity", "interact_rev", "user"): torch.tensor([iu_src, iu_dst], dtype=torch.long),
    }
    for key, (s, d) in kg_edges.items():
        edge_index_dict[("entity", key, "entity")] = torch.tensor([s, d], dtype=torch.long)

    metadata = (
        ["user", "entity"],
        list(edge_index_dict.keys()),
    )

    meta = {
        "n_users": n_users,
        "n_items": data.n_items,
        "n_entities": n_entities,
        "n_nodes": n_nodes,
        "n_relations": n_relations,
        "n_kg_relations": n_rel,
        "n_edges": int(edge_index.size(1)),
        "item_offset": n_users,
        "metadata": metadata,
    }
    return {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_index_dict": edge_index_dict,
        "metadata": metadata,
        "meta": meta,
    }


def load_data_and_graph(cfg: dict[str, Any], model_train: dict[int, set[int]]) -> dict[str, Any]:
    data = load_lastfm(cfg["data"]["path"])
    gcfg = cfg.get("models", {}).get("graphsage", {})
    edge_index, meta = build_ckg_edge_index(
        data,
        model_train,
        max_kg_edges=gcfg.get("max_kg_edges"),
        seed=cfg["split"]["seed"],
    )
    return {"data": data, "edge_index": edge_index, "meta": meta}


def load_data_and_typed_graph(
    cfg: dict[str, Any],
    model_train: dict[int, set[int]],
    *,
    max_kg_edges: int | None = None,
) -> dict[str, Any]:
    data = load_lastfm(cfg["data"]["path"])
    typed = build_typed_ckg(
        data,
        model_train,
        max_kg_edges=max_kg_edges,
        seed=cfg["split"]["seed"],
    )
    return {"data": data, **typed}
