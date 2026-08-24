"""Eksploracja Collaborative Knowledge Graph (KGAT)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.io import (
    ROOT,
    interaction_degree_maps,
    kg_degree_maps,
    load_config,
    load_dataset,
    load_schema,
)

sns.set_theme(style="whitegrid", context="notebook")


def _ensure_dirs(cfg: dict[str, Any]) -> tuple[Path, Path]:
    fig_dir = ROOT / cfg["paths"]["figures"]
    report_dir = ROOT / cfg["paths"]["reports"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, report_dir


def describe_series(values: np.ndarray | list[int], name: str) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"name": name, "count": 0}
    return {
        "name": name,
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "zeros": int((arr == 0).sum()),
    }


def plot_degree_hist(
    degrees: list[int] | np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 50,
    log_scale: bool = True,
) -> None:
    arr = np.asarray(list(degrees), dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if arr.size == 0:
        ax.set_title(title + " (brak danych)")
    else:
        ax.hist(arr, bins=bins, color="#2F6F8F", edgecolor="white", alpha=0.9)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("liczba wierzchołków (log)" if log_scale else "liczba wierzchołków")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_relation_bars(
    relation_counts: pd.DataFrame,
    out_path: Path,
    top_k: int = 20,
) -> None:
    df = relation_counts.head(top_k).copy()
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(df))))
    sns.barplot(data=df, y="relation_name", x="count", ax=ax, color="#C45C26")
    ax.set_xlabel("liczba tripletów")
    ax.set_ylabel("relacja")
    ax.set_title(f"Top-{min(top_k, len(df))} relacji w KG")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_ccdf(degrees: list[int] | np.ndarray, title: str, out_path: Path) -> None:
    """Complementary CDF stopni — diagnostyka ciężkich ogonów."""
    arr = np.sort(np.asarray(list(degrees), dtype=float))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if arr.size == 0:
        ax.set_title(title + " (brak danych)")
    else:
        y = 1.0 - np.arange(1, arr.size + 1) / arr.size
        ax.loglog(arr, y, color="#1F4E5F", lw=1.8)
        ax.set_xlabel("stopień")
        ax.set_ylabel("P(X ≥ k)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def variable_inventory(data: dict[str, Any], schema: dict[str, Any]) -> pd.DataFrame:
    """Typy zmiennych, liczebności i przykładowe wartości."""
    rows = []

    def add(name, dtype, n_unique, n_obs, example, source):
        rows.append(
            {
                "variable": name,
                "dtype": dtype,
                "n_unique": n_unique,
                "n_observations": n_obs,
                "example": example,
                "source": source,
            }
        )

    train, test, kg = data["train"], data["test"], data["kg"]
    users, items, entities, relations = (
        data["users"],
        data["items"],
        data["entities"],
        data["relations"],
    )

    add("user_id", "categorical_id/int", users["remap_id"].nunique(), len(train) + len(test), int(users["remap_id"].iloc[0]), "user_list / interactions")
    add("item_id", "categorical_id/int", items["remap_id"].nunique(), len(train) + len(test), int(items["remap_id"].iloc[0]), "item_list / interactions")
    add("entity_id", "categorical_id/int", entities["remap_id"].nunique(), len(kg) * 2, int(entities["remap_id"].iloc[0]), "entity_list / kg")
    add("relation_id", "categorical_id/int", relations["remap_id"].nunique(), len(kg), int(relations["remap_id"].iloc[0]), "relation_list / kg")
    add("freebase_id (item)", "categorical_id/string", items["freebase_id"].nunique(), len(items), str(items["freebase_id"].iloc[0]), "item_list")
    add("org_id (user)", "categorical_id/string|int", users["org_id"].nunique(), len(users), str(users["org_id"].iloc[0]), "user_list")
    add("kg.head", "categorical_id/int", kg["head"].nunique(), len(kg), int(kg["head"].iloc[0]), "kg_final")
    add("kg.tail", "categorical_id/int", kg["tail"].nunique(), len(kg), int(kg["tail"].iloc[0]), "kg_final")
    add("kg.relation", "categorical_id/int", kg["relation"].nunique(), len(kg), int(kg["relation"].iloc[0]), "kg_final")
    add("interact.split", "categorical {train,test}", 2, len(train) + len(test), "train|test", "train.txt / test.txt")

    # derived
    u_deg, i_deg = interaction_degree_maps(pd.concat([train, test], ignore_index=True))
    kg_deg = kg_degree_maps(kg)
    add("user_degree", "numeric_derived/int", len(u_deg), len(u_deg), int(next(iter(u_deg.values()))), "interactions")
    add("item_degree", "numeric_derived/int", len(i_deg), len(i_deg), int(next(iter(i_deg.values()))), "interactions")
    add("entity_degree", "numeric_derived/int", len(kg_deg["total"]), len(kg_deg["total"]), int(next(iter(kg_deg["total"].values()))), "kg")

    _ = schema  # schema dostępny pod dalsze rozszerzenia
    return pd.DataFrame(rows)


def explore(dataset: str | None = None) -> dict[str, Any]:
    cfg = load_config()
    schema = load_schema()
    fig_dir, report_dir = _ensure_dirs(cfg)
    data = load_dataset(cfg, dataset)
    name = data["name"]
    exp = cfg.get("exploration", {})
    bins = int(exp.get("degree_hist_bins", 50))
    log_scale = bool(exp.get("degree_log_scale", True))
    top_k_rel = int(exp.get("top_k_relations", 20))
    top_k_hubs = int(exp.get("top_k_hubs", 20))

    train, test, kg = data["train"], data["test"], data["kg"]
    users, items, entities, relations = (
        data["users"],
        data["items"],
        data["entities"],
        data["relations"],
    )
    interactions = pd.concat([train, test], ignore_index=True)

    # --- liczebności wierzchołków / krawędzi ---
    item_ids = set(items["remap_id"].astype(int))
    entity_ids = set(entities["remap_id"].astype(int))
    non_item_entities = entity_ids - item_ids

    counts = {
        "dataset": name,
        "n_users": int(users["remap_id"].nunique()),
        "n_items": int(items["remap_id"].nunique()),
        "n_entities": int(entities["remap_id"].nunique()),
        "n_non_item_entities": int(len(non_item_entities)),
        "n_relations": int(relations["remap_id"].nunique()),
        "n_train_interactions": int(len(train)),
        "n_test_interactions": int(len(test)),
        "n_interactions_total": int(len(interactions)),
        "n_kg_triplets": int(len(kg)),
        "n_users_in_train": int(train["user_id"].nunique()),
        "n_users_in_test": int(test["user_id"].nunique()),
        "n_items_in_train": int(train["item_id"].nunique()),
        "n_items_in_test": int(test["item_id"].nunique()),
        "items_subset_of_entities": item_ids.issubset(entity_ids),
    }

    # porównanie z papierem
    paper = schema.get("paper_statistics", {}).get(name, {})
    paper_check = {
        key: {"paper": paper.get(key), "observed": counts.get(mapped)}
        for key, mapped in [
            ("n_users", "n_users"),
            ("n_items", "n_items"),
            ("n_interactions", "n_interactions_total"),
            ("n_entities", "n_entities"),
            ("n_relations", "n_relations"),
            ("n_triplets", "n_kg_triplets"),
        ]
        if paper
    }

    # --- stopnie ---
    user_deg, item_deg = interaction_degree_maps(interactions)
    train_user_deg, train_item_deg = interaction_degree_maps(train)
    kg_deg = kg_degree_maps(kg)

    degree_stats = {
        "user_degree_all": describe_series(list(user_deg.values()), "user_degree_all"),
        "item_degree_all": describe_series(list(item_deg.values()), "item_degree_all"),
        "user_degree_train": describe_series(list(train_user_deg.values()), "user_degree_train"),
        "item_degree_train": describe_series(list(train_item_deg.values()), "item_degree_train"),
        "entity_out_degree": describe_series(list(kg_deg["out"].values()), "entity_out_degree"),
        "entity_in_degree": describe_series(list(kg_deg["in"].values()), "entity_in_degree"),
        "entity_total_degree": describe_series(list(kg_deg["total"].values()), "entity_total_degree"),
    }

    # huby
    top_users = sorted(user_deg.items(), key=lambda x: x[1], reverse=True)[:top_k_hubs]
    top_items = sorted(item_deg.items(), key=lambda x: x[1], reverse=True)[:top_k_hubs]
    top_entities = sorted(kg_deg["total"].items(), key=lambda x: x[1], reverse=True)[:top_k_hubs]

    # relacje
    rel_map = dict(zip(relations["remap_id"].astype(int), relations["org_id"].astype(str)))
    rel_counts = (
        kg["relation"]
        .value_counts()
        .rename_axis("relation_id")
        .reset_index(name="count")
    )
    rel_counts["relation_name"] = rel_counts["relation_id"].map(
        lambda r: rel_map.get(int(r), str(r)).split("/")[-1]
    )
    rel_counts["relation_uri"] = rel_counts["relation_id"].map(lambda r: rel_map.get(int(r), str(r)))

    # typy zmiennych
    variables = variable_inventory(data, schema)

    # gęstość / sparsity
    n_u, n_i = counts["n_users"], counts["n_items"]
    density_ui = counts["n_interactions_total"] / (n_u * n_i) if n_u and n_i else 0.0
    n_e = counts["n_entities"]
    density_kg = counts["n_kg_triplets"] / (n_e * n_e) if n_e else 0.0

    # --- wykresy ---
    prefix = fig_dir / name
    prefix.mkdir(parents=True, exist_ok=True)
    if exp.get("save_figures", True):
        plot_degree_hist(list(user_deg.values()), f"{name}: stopnie użytkowników", "degree (interakcje)", prefix / "user_degree_hist.png", bins, log_scale)
        plot_degree_hist(list(item_deg.values()), f"{name}: stopnie itemów", "degree (interakcje)", prefix / "item_degree_hist.png", bins, log_scale)
        plot_degree_hist(list(kg_deg["total"].values()), f"{name}: stopnie encji KG", "degree (in+out)", prefix / "entity_degree_hist.png", bins, log_scale)
        plot_ccdf(list(user_deg.values()), f"{name}: CCDF stopni użytkowników", prefix / "user_degree_ccdf.png")
        plot_ccdf(list(item_deg.values()), f"{name}: CCDF stopni itemów", prefix / "item_degree_ccdf.png")
        plot_ccdf(list(kg_deg["total"].values()), f"{name}: CCDF stopni encji KG", prefix / "entity_degree_ccdf.png")
        plot_relation_bars(rel_counts, prefix / "relation_counts.png", top_k=top_k_rel)

        # pie node types
        fig, ax = plt.subplots(figsize=(6, 6))
        sizes = [counts["n_users"], counts["n_items"], counts["n_non_item_entities"]]
        labels = ["users", "items", "other entities"]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#2F6F8F", "#C45C26", "#7A9A6D"], startangle=90)
        ax.set_title(f"{name}: skład wierzchołków CKG")
        fig.tight_layout()
        fig.savefig(prefix / "node_type_composition.png", dpi=140)
        plt.close(fig)

    report = {
        "counts": counts,
        "paper_check": paper_check,
        "sparsity": {
            "user_item_density": density_ui,
            "user_item_sparsity": 1.0 - density_ui,
            "kg_density": density_kg,
            "kg_sparsity": 1.0 - density_kg,
        },
        "degree_stats": degree_stats,
        "top_user_hubs": [{"user_id": u, "degree": d} for u, d in top_users],
        "top_item_hubs": [{"item_id": i, "degree": d} for i, d in top_items],
        "top_entity_hubs": [{"entity_id": e, "degree": d} for e, d in top_entities],
        "relation_counts": rel_counts.to_dict(orient="records"),
        "variables": variables.to_dict(orient="records"),
        "node_types": schema.get("node_types", {}),
        "edge_types": schema.get("edge_types", {}),
    }

    if exp.get("save_report", True):
        out_json = report_dir / f"{name}_exploration.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # czytelny markdown
        md_path = report_dir / f"{name}_exploration.md"
        md = _to_markdown(report, variables, rel_counts, degree_stats)
        md_path.write_text(md, encoding="utf-8")

        variables.to_csv(report_dir / f"{name}_variables.csv", index=False)
        rel_counts.to_csv(report_dir / f"{name}_relations.csv", index=False)

    return report


def _to_markdown(
    report: dict[str, Any],
    variables: pd.DataFrame,
    rel_counts: pd.DataFrame,
    degree_stats: dict[str, Any],
) -> str:
    c = report["counts"]
    s = report["sparsity"]
    lines = [
        f"# Eksploracja grafu: `{c['dataset']}`",
        "",
        "## Liczebności wierzchołków i krawędzi",
        "",
        f"- users: **{c['n_users']:,}**",
        f"- items: **{c['n_items']:,}**",
        f"- entities (KG): **{c['n_entities']:,}** (w tym non-item: {c['n_non_item_entities']:,})",
        f"- relations: **{c['n_relations']:,}**",
        f"- interakcje train/test/total: {c['n_train_interactions']:,} / {c['n_test_interactions']:,} / {c['n_interactions_total']:,}",
        f"- tripletów KG: **{c['n_kg_triplets']:,}**",
        f"- items ⊆ entities: {c['items_subset_of_entities']}",
        "",
        "## Gęstość / sparsity",
        "",
        f"- user–item density: {s['user_item_density']:.6e} (sparsity {s['user_item_sparsity']:.6f})",
        f"- KG density: {s['kg_density']:.6e} (sparsity {s['kg_sparsity']:.6f})",
        "",
        "## Statystyki stopni",
        "",
        "| zmienna | count | min | median | mean | p90 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, st in degree_stats.items():
        lines.append(
            f"| {st['name']} | {st.get('count', 0)} | {st.get('min', float('nan')):.1f} | "
            f"{st.get('median', float('nan')):.1f} | {st.get('mean', float('nan')):.2f} | "
            f"{st.get('p90', float('nan')):.1f} | {st.get('p99', float('nan')):.1f} | "
            f"{st.get('max', float('nan')):.1f} |"
        )

    lines += ["", "## Typy zmiennych", "", variables.to_markdown(index=False), ""]
    lines += ["## Rozkład relacji KG", "", rel_counts[["relation_id", "relation_name", "count"]].to_markdown(index=False), ""]

    if report.get("paper_check"):
        lines += ["## Porównanie ze statystykami z papieru", ""]
        for k, v in report["paper_check"].items():
            lines.append(f"- {k}: paper={v['paper']}, observed={v['observed']}")
        lines.append("")

    lines += [
        "## Huby (top)",
        "",
        f"- top users: {report['top_user_hubs'][:5]}",
        f"- top items: {report['top_item_hubs'][:5]}",
        f"- top entities: {report['top_entity_hubs'][:5]}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="EDA grafu KGAT")
    parser.add_argument("--dataset", default=None, help="last-fm | amazon-book | yelp2018")
    args = parser.parse_args()
    report = explore(args.dataset)
    c = report["counts"]
    print(f"[{c['dataset']}] users={c['n_users']:,} items={c['n_items']:,} "
          f"entities={c['n_entities']:,} relations={c['n_relations']:,} "
          f"interactions={c['n_interactions_total']:,} triplets={c['n_kg_triplets']:,}")
    print("Raport zapisany w outputs/reports/")


if __name__ == "__main__":
    main()
