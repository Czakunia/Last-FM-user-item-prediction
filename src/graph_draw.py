"""Budowa grafu CKG / KG, metryki oraz rysunek podgrafu."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.io import ROOT, load_config, load_dataset

# kolory wierzchołków
COLOR = {"user": "#2F6F8F", "item": "#C45C26", "entity": "#7A9A6D"}


def build_kg_digraph(kg: pd.DataFrame, item_ids: set[int] | None = None) -> nx.MultiDiGraph:
    """Directed multi-graph knowledge graph (h -r-> t)."""
    G = nx.MultiDiGraph()
    item_ids = item_ids or set()
    nodes: dict[int, str] = {}
    for h, r, t in zip(kg["head"].to_numpy(), kg["relation"].to_numpy(), kg["tail"].to_numpy()):
        h, r, t = int(h), int(r), int(t)
        for n in (h, t):
            if n not in nodes:
                nodes[n] = "item" if n in item_ids else "entity"
                G.add_node(f"e{n}", kind=nodes[n], raw_id=n)
        G.add_edge(f"e{h}", f"e{t}", relation=r, etype="kg")
    return G


def build_bipartite_interactions(
    interactions: pd.DataFrame,
    edge_attr: str = "interact",
) -> nx.Graph:
    """Undirected bipartite user–item graph."""
    G = nx.Graph()
    for u, i in zip(interactions["user_id"].to_numpy(), interactions["item_id"].to_numpy()):
        u, i = int(u), int(i)
        un, inn = f"u{u}", f"e{i}"
        if un not in G:
            G.add_node(un, kind="user", raw_id=u)
        if inn not in G:
            G.add_node(inn, kind="item", raw_id=i)
        G.add_edge(un, inn, etype=edge_attr)
    return G


def sample_user_ego_ckg(
    train: pd.DataFrame,
    kg: pd.DataFrame,
    user_id: int | None = None,
    max_history: int = 8,
    max_kg_per_item: int = 4,
    seed: int = 42,
) -> tuple[nx.Graph, dict[str, Any]]:
    """
    Collaborative Knowledge Graph ego-sample wokół jednego użytkownika:
    user — items z historii — sąsiedzi KG tych itemów.
    """
    rng = np.random.default_rng(seed)
    users = train["user_id"].unique()
    if user_id is None:
        # wybierz usera ze średnią historią (czytelny rysunek)
        deg = train.groupby("user_id").size()
        candidates = deg[(deg >= 10) & (deg <= 40)].index.to_numpy()
        if len(candidates) == 0:
            candidates = users
        user_id = int(rng.choice(candidates))

    hist = train.loc[train["user_id"] == user_id, "item_id"].drop_duplicates().to_numpy()
    if len(hist) > max_history:
        hist = rng.choice(hist, size=max_history, replace=False)

    # indeks KG: head -> list[(rel, tail)] i odwrotnie dla itemów jako tail
    out_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    in_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for h, r, t in zip(kg["head"].to_numpy(), kg["relation"].to_numpy(), kg["tail"].to_numpy()):
        h, r, t = int(h), int(r), int(t)
        out_adj[h].append((r, t))
        in_adj[t].append((r, h))

    G = nx.Graph()
    meta = {"user_id": user_id, "history_items": [int(x) for x in hist], "kg_edges": 0}
    u_node = f"u{user_id}"
    G.add_node(u_node, kind="user", raw_id=user_id, label=f"user {user_id}")

    for i in hist:
        i = int(i)
        i_node = f"e{i}"
        G.add_node(i_node, kind="item", raw_id=i, label=f"item {i}")
        G.add_edge(u_node, i_node, etype="interact", relation=-1)

        # próbka sąsiadów KG
        neigh: list[tuple[str, int, int, int]] = []  # (direction, other, rel, item)
        for r, t in out_adj.get(i, [])[: 50]:
            neigh.append(("out", t, r, i))
        for r, h in in_adj.get(i, [])[: 50]:
            neigh.append(("in", h, r, i))
        if len(neigh) > max_kg_per_item:
            idx = rng.choice(len(neigh), size=max_kg_per_item, replace=False)
            neigh = [neigh[j] for j in idx]

        for direction, other, r, item in neigh:
            o_node = f"e{other}"
            if o_node not in G:
                # other może być itemem lub encją
                kind = "item" if other in set(hist) else "entity"
                # jeśli other jest w zakresie itemów historii już dodany jako item
                G.add_node(o_node, kind=kind, raw_id=other, label=f"ent {other}")
            a, b = (i_node, o_node) if direction == "out" else (o_node, i_node)
            # undirected rysunek — zachowaj relation na krawędzi
            if not G.has_edge(i_node, o_node):
                G.add_edge(i_node, o_node, etype="kg", relation=int(r))
                meta["kg_edges"] += 1

    meta["n_nodes"] = G.number_of_nodes()
    meta["n_edges"] = G.number_of_edges()
    return G, meta


def compute_graph_stats(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kg: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_entities: int,
) -> dict[str, Any]:
    """Policz graf bez materializacji pełnego NetworkX (za duży)."""
    interactions = pd.concat([train, test], ignore_index=True)
    # bipartite
    n_ui = len(interactions)
    user_deg = interactions.groupby("user_id").size()
    item_deg = interactions.groupby("item_id").size()

    # KG as simple digraph stats
    n_trip = len(kg)
    heads = kg["head"].nunique()
    tails = kg["tail"].nunique()
    kg_nodes = pd.unique(pd.concat([kg["head"], kg["tail"]], ignore_index=True))
    # approx connectedness via giant weakly component on sample is expensive;
    # report structural counts instead

    # collaborative: users + entities, edges = UI + KG (undirected view of KG)
    ckg_nodes = n_users + n_entities
    ckg_edges = n_ui + n_trip  # UI undirected + KG directed triplets as edges

    return {
        "bipartite_user_item": {
            "n_nodes": n_users + n_items,
            "n_users": n_users,
            "n_items": n_items,
            "n_edges": int(n_ui),
            "avg_user_degree": float(user_deg.mean()),
            "avg_item_degree": float(item_deg.mean()),
            "density": float(n_ui / (n_users * n_items)) if n_users and n_items else 0.0,
        },
        "knowledge_graph": {
            "n_nodes": int(len(kg_nodes)),
            "n_entities_listed": n_entities,
            "n_directed_edges": int(n_trip),
            "n_heads": int(heads),
            "n_tails": int(tails),
            "n_relations": int(kg["relation"].nunique()),
            "avg_out_degree": float(n_trip / max(heads, 1)),
            "density_directed": float(n_trip / (len(kg_nodes) ** 2)) if len(kg_nodes) else 0.0,
        },
        "collaborative_kg_view": {
            "n_nodes": int(ckg_nodes),
            "n_edges_approx": int(ckg_edges),
            "edge_breakdown": {"user_item": int(n_ui), "kg_triplets": int(n_trip)},
            "note": "CKG = users ∪ entities; edges = interactions ∪ KG triplets",
        },
    }


def draw_ego_graph(
    G: nx.Graph,
    out_path: Path,
    title: str,
    rel_names: dict[int, str] | None = None,
) -> None:
    rel_names = rel_names or {}
    kinds = nx.get_node_attributes(G, "kind")
    colors = [COLOR.get(kinds.get(n, "entity"), "#888888") for n in G.nodes()]
    sizes = []
    for n in G.nodes():
        k = kinds.get(n, "entity")
        sizes.append({"user": 900, "item": 500, "entity": 280}.get(k, 250))

    pos = nx.spring_layout(G, seed=42, k=1.4 / max(np.sqrt(G.number_of_nodes()), 1), iterations=80)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    # krawędzie: interact vs kg
    interact_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "interact"]
    kg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "kg"]
    nx.draw_networkx_edges(G, pos, edgelist=interact_edges, ax=ax, width=2.0, edge_color="#2F6F8F", alpha=0.85)
    nx.draw_networkx_edges(G, pos, edgelist=kg_edges, ax=ax, width=1.2, edge_color="#9AA59A", style="dashed", alpha=0.75)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, edgecolors="white", linewidths=0.8)

    labels = {}
    for n, data in G.nodes(data=True):
        if data.get("kind") == "user":
            labels[n] = f"u{data['raw_id']}"
        elif data.get("kind") == "item":
            labels[n] = f"i{data['raw_id']}"
        else:
            labels[n] = f"e{data['raw_id']}"
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

    # legenda ręczna
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend = [
        Patch(facecolor=COLOR["user"], edgecolor="white", label="user"),
        Patch(facecolor=COLOR["item"], edgecolor="white", label="item"),
        Patch(facecolor=COLOR["entity"], edgecolor="white", label="entity (KG)"),
        Line2D([0], [0], color="#2F6F8F", lw=2, label="interakcja u–i"),
        Line2D([0], [0], color="#9AA59A", lw=1.5, ls="--", label="krawędź KG"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=True, fontsize=9)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_schema_diagram(out_path: Path, dataset: str) -> None:
    """Schematyczny rysunek struktury CKG (nie pełne dane)."""
    G = nx.DiGraph()
    G.add_node("U", label="Users", kind="user")
    G.add_node("I", label="Items", kind="item")
    G.add_node("E", label="Entities\n(KG)", kind="entity")
    G.add_edge("U", "I", label="interact\n(train/test)")
    G.add_edge("I", "E", label="item ⊆ entity")
    G.add_edge("E", "E", label="(h,r,t)\nkg_final")

    pos = {"U": (0.0, 0.6), "I": (1.0, 0.6), "E": (2.0, 0.35)}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [COLOR[G.nodes[n]["kind"]] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=3200, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(
        G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()}, ax=ax, font_size=10, font_color="white"
    )
    # self-loop on E — draw manually-ish via connectionstyle
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=[("U", "I"), ("I", "E")],
        arrows=True, arrowstyle="-|>", arrowsize=18, width=2.2, edge_color="#444444",
        connectionstyle="arc3,rad=0.0", node_size=3200,
    )
    # self loop
    ax.annotate(
        "",
        xy=(2.15, 0.45),
        xytext=(1.85, 0.45),
        arrowprops=dict(arrowstyle="-|>", color="#444444", lw=2.0, connectionstyle="arc3,rad=0.9"),
    )
    ax.text(2.0, 0.72, "(h, r, t)\nkg_final", ha="center", va="bottom", fontsize=9, color="#333333")
    ax.text(0.5, 0.78, "interact\n(train/test)", ha="center", fontsize=9, color="#333333")
    ax.text(1.5, 0.52, "item ⊆ entity", ha="center", fontsize=9, color="#333333")
    ax.set_title(f"Schemat Collaborative Knowledge Graph — {dataset}")
    ax.set_xlim(-0.4, 2.6)
    ax.set_ylim(0.0, 1.05)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_draw(dataset: str | None = None, user_id: int | None = None) -> dict[str, Any]:
    cfg = load_config()
    data = load_dataset(cfg, dataset)
    name = data["name"]
    fig_dir = ROOT / cfg["paths"]["figures"] / name
    report_dir = ROOT / cfg["paths"]["reports"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train, test, kg = data["train"], data["test"], data["kg"]
    n_users = int(data["users"]["remap_id"].nunique())
    n_items = int(data["items"]["remap_id"].nunique())
    n_entities = int(data["entities"]["remap_id"].nunique())

    stats = compute_graph_stats(train, test, kg, n_users, n_items, n_entities)

    rel_names = {
        int(r): str(n).split("/")[-1]
        for r, n in zip(data["relations"]["remap_id"], data["relations"]["org_id"])
    }

    # 1) schemat
    draw_schema_diagram(fig_dir / "ckg_schema.png", name)

    # 2) ego sample
    G_ego, meta = sample_user_ego_ckg(
        train, kg, user_id=user_id, max_history=8, max_kg_per_item=4, seed=cfg.get("random_seed", 42)
    )
    draw_ego_graph(
        G_ego,
        fig_dir / "ckg_ego_sample.png",
        title=f"{name}: podgraf CKG wokół user {meta['user_id']} "
        f"(|V|={meta['n_nodes']}, |E|={meta['n_edges']})",
        rel_names=rel_names,
    )

    # 3) mały wycinek samego KG (losowe trójki wokół itemów z ego)
    item_ids = set(meta["history_items"])
    kg_sub = kg[kg["head"].isin(item_ids) | kg["tail"].isin(item_ids)].head(80)
    G_kg = nx.DiGraph()
    for h, r, t in zip(kg_sub["head"], kg_sub["relation"], kg_sub["tail"]):
        h, r, t = int(h), int(r), int(t)
        for n, kind in ((h, "item" if h in item_ids else "entity"), (t, "item" if t in item_ids else "entity")):
            nid = f"e{n}"
            if nid not in G_kg:
                G_kg.add_node(nid, kind=kind, raw_id=n)
        G_kg.add_edge(f"e{h}", f"e{t}", relation=int(r), etype="kg")
    # rysuj jako undirected look z directed edges
    pos = nx.spring_layout(G_kg, seed=1, k=1.2 / max(np.sqrt(G_kg.number_of_nodes()), 1))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    colors = [COLOR.get(G_kg.nodes[n]["kind"], "#888") for n in G_kg.nodes()]
    nx.draw_networkx_nodes(G_kg, pos, ax=ax, node_color=colors, node_size=350, edgecolors="white")
    nx.draw_networkx_edges(G_kg, pos, ax=ax, arrows=True, arrowsize=10, width=1.0, edge_color="#666666", alpha=0.7, node_size=350)
    labels = {n: f"{G_kg.nodes[n]['kind'][0]}{G_kg.nodes[n]['raw_id']}" for n in G_kg.nodes()}
    nx.draw_networkx_labels(G_kg, pos, labels=labels, ax=ax, font_size=7)
    ax.set_title(f"{name}: wycinek KG wokół itemów historii (|V|={G_kg.number_of_nodes()}, |E|={G_kg.number_of_edges()})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "kg_snippet.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    payload = {"dataset": name, "stats": stats, "ego_sample": meta, "figures": {
        "schema": str((fig_dir / "ckg_schema.png").relative_to(ROOT)),
        "ego": str((fig_dir / "ckg_ego_sample.png").relative_to(ROOT)),
        "kg_snippet": str((fig_dir / "kg_snippet.png").relative_to(ROOT)),
    }}
    out_json = report_dir / f"{name}_graph_build.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = [
        f"# Budowa i policzenie grafu — `{name}`",
        "",
        "## Collaborative Knowledge Graph (policzone)",
        "",
        f"- węzły CKG (users ∪ entities): **{stats['collaborative_kg_view']['n_nodes']:,}**",
        f"- krawędzie (interakcje + tripletów KG): **{stats['collaborative_kg_view']['n_edges_approx']:,}**",
        f"  - user–item: {stats['bipartite_user_item']['n_edges']:,}",
        f"  - KG directed: {stats['knowledge_graph']['n_directed_edges']:,}",
        "",
        "## Bipartite user–item",
        "",
        f"- |V| = {stats['bipartite_user_item']['n_nodes']:,} "
        f"(users {stats['bipartite_user_item']['n_users']:,} + items {stats['bipartite_user_item']['n_items']:,})",
        f"- |E| = {stats['bipartite_user_item']['n_edges']:,}",
        f"- avg deg(user) = {stats['bipartite_user_item']['avg_user_degree']:.2f}",
        f"- avg deg(item) = {stats['bipartite_user_item']['avg_item_degree']:.2f}",
        f"- density = {stats['bipartite_user_item']['density']:.6e}",
        "",
        "## Knowledge Graph",
        "",
        f"- |V| (w tripletach) = {stats['knowledge_graph']['n_nodes']:,}",
        f"- |E| directed = {stats['knowledge_graph']['n_directed_edges']:,}",
        f"- relacji = {stats['knowledge_graph']['n_relations']}",
        f"- avg out-degree (po head) ≈ {stats['knowledge_graph']['avg_out_degree']:.2f}",
        "",
        "## Rysunki (podgraf — pełny graf jest za duży)",
        "",
        f"- schemat: `{payload['figures']['schema']}`",
        f"- ego CKG user {meta['user_id']}: `{payload['figures']['ego']}`",
        f"- wycinek KG: `{payload['figures']['kg_snippet']}`",
        "",
    ]
    (report_dir / f"{name}_graph_build.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Zbuduj, policz i narysuj graf CKG/KG")
    p.add_argument("--dataset", default=None)
    p.add_argument("--user-id", type=int, default=None)
    args = p.parse_args()
    out = run_draw(args.dataset, args.user_id)
    s = out["stats"]
    print(f"[{out['dataset']}] CKG nodes={s['collaborative_kg_view']['n_nodes']:,} "
          f"edges≈{s['collaborative_kg_view']['n_edges_approx']:,}")
    print("Figury:", out["figures"])


if __name__ == "__main__":
    main()
