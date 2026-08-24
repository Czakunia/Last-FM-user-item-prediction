"""Stage 0 audit → outputs/lastfm/audit/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lastfm_lp.data.load_kgat_lastfm import LastFMData, item_popularity


def run_audit(
    data: LastFMData,
    splits: dict[str, Any],
    eval_users: list[int],
    out_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = splits["model_train"]
    valid = splits["valid"]
    test = splits["test"]
    pop = item_popularity(train, data.n_items)

    hist_lens = np.array([len(train[u]) for u in sorted(train)], dtype=np.int32)
    item_cov_kg = int((data.item_kg_degree[: data.n_items] > 0).sum())

    # cold-start: users with tiny history; items never in model_train
    cold_users = int((hist_lens <= 5).sum())
    cold_items = int((pop == 0).sum())

    # label balance on eval subsample preview (positives only counts)
    n_val_pos = sum(len(valid.get(u, ())) for u in eval_users)
    n_test_pos = sum(len(test.get(u, ())) for u in eval_users)

    report = {
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_entities": data.n_entities,
        "n_relations": data.n_relations,
        "n_train_interactions_model": int(sum(len(v) for v in train.values())),
        "n_valid_interactions": int(sum(len(v) for v in valid.values())),
        "n_test_interactions": int(sum(len(v) for v in test.values())),
        "history_length": {
            "min": int(hist_lens.min()) if len(hist_lens) else 0,
            "median": float(np.median(hist_lens)) if len(hist_lens) else 0,
            "mean": float(hist_lens.mean()) if len(hist_lens) else 0,
            "p90": float(np.percentile(hist_lens, 90)) if len(hist_lens) else 0,
            "max": int(hist_lens.max()) if len(hist_lens) else 0,
        },
        "item_popularity": {
            "min": int(pop.min()),
            "median": float(np.median(pop)),
            "mean": float(pop.mean()),
            "p90": float(np.percentile(pop, 90)),
            "max": int(pop.max()),
        },
        "item_kg_degree": {
            "min": int(data.item_kg_degree.min()),
            "median": float(np.median(data.item_kg_degree)),
            "mean": float(data.item_kg_degree.mean()),
            "max": int(data.item_kg_degree.max()),
            "items_with_kg_edge": item_cov_kg,
            "coverage": float(item_cov_kg / max(data.n_items, 1)),
        },
        "relation_types": data.relation_names,
        "relation_counts": data.relation_counts,
        "cold_start": {
            "users_history_le_5": cold_users,
            "items_unseen_in_model_train": cold_items,
        },
        "eval_subsample": {
            "n_eval_users": len(eval_users),
            "n_valid_positives_on_eval_users": n_val_pos,
            "n_test_positives_on_eval_users": n_test_pos,
        },
        "label_protocol_note": (
            "Negatives are sampled at candidate-building time; "
            "positives come from valid/test; history from model_train only."
        ),
    }

    (out_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # figures
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(hist_lens, bins=50, color="#2F6F8F", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("User history length (model_train)")
    ax.set_xlabel("|H_u|")
    fig.tight_layout()
    fig.savefig(out_dir / "history_length_hist.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pop[pop > 0], bins=50, color="#C45C26", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Item popularity (model_train)")
    ax.set_xlabel("pop(i)")
    fig.tight_layout()
    fig.savefig(out_dir / "item_popularity_hist.png", dpi=140)
    plt.close(fig)

    rel_df = pd.DataFrame(
        [
            {"relation_id": k, "name": data.relation_names.get(k, str(k)), "count": v}
            for k, v in sorted(data.relation_counts.items(), key=lambda x: -x[1])
        ]
    )
    rel_df.to_csv(out_dir / "relation_counts.csv", index=False)

    md = [
        "# Last-FM audit (Stage 0)",
        "",
        f"- users: **{report['n_users']:,}**",
        f"- items: **{report['n_items']:,}**",
        f"- entities: **{report['n_entities']:,}**",
        f"- relations: **{report['n_relations']:,}**",
        f"- model_train / valid / test interactions: "
        f"{report['n_train_interactions_model']:,} / "
        f"{report['n_valid_interactions']:,} / "
        f"{report['n_test_interactions']:,}",
        f"- history length median/mean/max: "
        f"{report['history_length']['median']:.1f} / "
        f"{report['history_length']['mean']:.1f} / "
        f"{report['history_length']['max']}",
        f"- item KG coverage: {report['item_kg_degree']['coverage']:.4f}",
        f"- cold users (|H|≤5): {cold_users:,}; cold items: {cold_items:,}",
        f"- eval users: {len(eval_users):,}",
        "",
    ]
    (out_dir / "audit_report.md").write_text("\n".join(md), encoding="utf-8")
    return report
