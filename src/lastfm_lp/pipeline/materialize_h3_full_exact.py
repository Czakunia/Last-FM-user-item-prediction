"""FULL exact H3 materialization via sorted_batch user blocks.

Writes 8-D H3_RAW plus derived signed (3-D) and signed_energy (6-D) matrices.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.lastfm_lp.binary.hcr_triple import H3_RAW_FEATURE_ORDER, history_unordered_pairs
from src.lastfm_lp.binary.hcr_triple_user_batch import (
    build_cross_fit_triple_user_batch,
    compute_h3_user_block,
)
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit

SIGNED_COLS = [0, 2, 4]  # mean/max/top3mean a111 (drop wmean≡mean)
SIGNED_ENERGY_COLS = [0, 1, 2, 3, 4, 5]  # drop wmean* duplicates

SIGNED_NAMES = [H3_RAW_FEATURE_ORDER[i] for i in SIGNED_COLS]
SIGNED_ENERGY_NAMES = [H3_RAW_FEATURE_ORDER[i] for i in SIGNED_ENERGY_COLS]


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _group_rows(users: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for r, u in enumerate(users.tolist()):
        groups[int(u)].append(r)
    return {u: np.asarray(rows, dtype=np.int64) for u, rows in groups.items()}


def _fingerprint_pairs(bundle: dict[str, Any]) -> dict[str, Any]:
    feat = Path(bundle["cfg"]["paths"]["features"])
    out = {}
    for name in ("train_pairs.npz", "val_pairs.npz", "test_pairs.npz", "user_folds.json"):
        p = feat / name
        if p.exists():
            out[name] = _sha_file(p)
    return out


def materialize_h3_full_exact(
    bundle: dict[str, Any],
    *,
    out_dir: Path,
    backend: str = "sorted_batch",
    max_history: int = 25,
    splits: tuple[str, ...] = ("train", "val", "test"),
    log_path: Path | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path) if log_path else out_dir / "FULL_H3_MATERIALIZATION_LOG.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if (out_dir / "COMPLETE").exists():
        log(f"[H3_FULL] reuse existing COMPLETE at {out_dir}")
        return out_dir

    t0 = time.time()
    cf = ensure_cross_fit(bundle)
    log(f"[H3_FULL] building cross-fit triple indices backend={backend} …")
    triples = build_cross_fit_triple_user_batch(bundle, backend=backend)  # type: ignore[arg-type]
    model_train = bundle["model_train"]
    fp = _fingerprint_pairs(bundle)

    support_rows: list[dict[str, Any]] = []

    for split in splits:
        key = "val_pairs" if split == "val" else f"{split}_pairs"
        pairs = bundle[key]
        users = pairs["user_id"]
        items = pairs["item_id"]
        labels = pairs["label"]
        n = int(len(labels))
        x_path = out_dir / f"X_{split}.npy"
        prog_path = out_dir / f"progress_{split}.json"
        min_path = out_dir / f"min_a111_{split}.npy"
        maxabs_path = out_dir / f"max_abs_a111_{split}.npy"
        npairs_path = out_dir / f"n_pairs_used_{split}.npy"

        if (out_dir / f"COMPLETE_{split}").exists() and x_path.exists():
            log(f"[H3_FULL] skip {split} (COMPLETE)")
            continue

        if x_path.exists() and prog_path.exists():
            X = np.load(x_path, mmap_mode="r+")
            min_a = np.load(min_path, mmap_mode="r+")
            max_abs = np.load(maxabs_path, mmap_mode="r+")
            n_pairs_used = np.load(npairs_path, mmap_mode="r+")
            next_user_i = int(json.loads(prog_path.read_text())["next_user_i"])
        else:
            X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(n, 8))
            X[:] = 0
            min_a = np.lib.format.open_memmap(min_path, mode="w+", dtype=np.float64, shape=(n,))
            max_abs = np.lib.format.open_memmap(maxabs_path, mode="w+", dtype=np.float64, shape=(n,))
            n_pairs_used = np.lib.format.open_memmap(npairs_path, mode="w+", dtype=np.int32, shape=(n,))
            min_a[:] = 0
            max_abs[:] = 0
            n_pairs_used[:] = 0
            next_user_i = 0

        groups = _group_rows(users)
        user_list = sorted(groups.keys())
        log(f"[H3_FULL] {split}: rows={n:,} users={len(user_list):,} resume_user_i={next_user_i}")

        logical = 0
        t_split = time.time()
        for ui in range(next_user_i, len(user_list)):
            u = user_list[ui]
            ridx = groups[u]
            hist = model_train.get(u, set())
            hp = history_unordered_pairs(hist, max_history=max_history)
            logical += int(hp.shape[0]) * int(len(ridx))
            pw = cf.index_for(u, split=split)
            block = compute_h3_user_block(
                user_id=u,
                history_items=hist,
                candidate_items=items[ridx],
                labels=labels[ridx],
                split=split,
                triples=triples,
                pairwise=pw,
                max_history=max_history,
            )
            assert block["index_key"] == triples.expected_index_key(u, split)
            X[ridx] = block["features"]
            min_a[ridx] = block["min_a111"]
            max_abs[ridx] = block["max_abs_a111"]
            n_pairs_used[ridx] = block["n_pairs_used"]

            # light support audit sample (positives only, capped)
            if split == "test" and len(support_rows) < 50_000:
                for local, r in enumerate(ridx.tolist()):
                    if int(labels[r]) != 1:
                        continue
                    support_rows.append(
                        {
                            "split": split,
                            "user": u,
                            "item": int(items[r]),
                            "mean_a111": float(block["features"][local, 0]),
                            "max_a111": float(block["features"][local, 2]),
                            "min_a111": float(block["min_a111"][local]),
                            "max_abs_a111": float(block["max_abs_a111"][local]),
                            "n_pairs_used": int(block["n_pairs_used"][local]),
                            "n_i": int(bundle["popularity"][int(items[r])]),
                        }
                    )

            if (ui + 1) % 200 == 0 or ui + 1 == len(user_list):
                X.flush()
                min_a.flush()
                max_abs.flush()
                n_pairs_used.flush()
                prog_path.write_text(
                    json.dumps(
                        {
                            "next_user_i": ui + 1,
                            "n_users": len(user_list),
                            "n_rows": n,
                            "logical_triples_so_far": logical,
                        }
                    ),
                    encoding="utf-8",
                )
                elapsed = time.time() - t_split
                rate_u = (ui + 1 - next_user_i) / max(elapsed, 1e-9)
                eta = (len(user_list) - ui - 1) / max(rate_u, 1e-9)
                log(
                    f"[H3_FULL] {split} users {ui+1}/{len(user_list)} "
                    f"({100*(ui+1)/len(user_list):.1f}%) "
                    f"users/s={rate_u:.2f} eta_s={eta:.0f} logical={logical:,}"
                )

        X.flush()
        # derived views
        Xs = np.asarray(X[:, SIGNED_COLS], dtype=np.float32)
        Xe = np.asarray(X[:, SIGNED_ENERGY_COLS], dtype=np.float32)
        np.save(out_dir / f"X_{split}_signed.npy", Xs)
        np.save(out_dir / f"X_{split}_signed_energy.npy", Xe)
        (out_dir / f"COMPLETE_{split}").write_text("ok\n", encoding="utf-8")
        log(f"[H3_FULL] {split} done in {(time.time()-t_split)/3600:.3f}h")

    (out_dir / "feature_names.json").write_text(
        json.dumps(H3_RAW_FEATURE_ORDER, indent=2), encoding="utf-8"
    )
    (out_dir / "feature_names_signed.json").write_text(
        json.dumps(SIGNED_NAMES, indent=2), encoding="utf-8"
    )
    (out_dir / "feature_names_signed_energy.json").write_text(
        json.dumps(SIGNED_ENERGY_NAMES, indent=2), encoding="utf-8"
    )
    # aliases expected by user
    for split in splits:
        # already saved as X_{split}_signed.npy
        pass

    meta = {
        "protocol": "LASTFM_FULL_SOTA_V2_RUN_FULL_H3_EXACT",
        "backend": backend,
        "max_history": max_history,
        "feature_order_raw": H3_RAW_FEATURE_ORDER,
        "feature_order_signed": SIGNED_NAMES,
        "feature_order_signed_energy": SIGNED_ENERGY_NAMES,
        "signed_cols": SIGNED_COLS,
        "signed_energy_cols": SIGNED_ENERGY_COLS,
        "wmean_equals_mean": True,
        "pair_table_fingerprints": fp,
        "crossfit": bundle["cfg"].get("hcr", {}).get("cross_fitting"),
        "elapsed_hours": (time.time() - t0) / 3600.0,
    }
    (out_dir / "H3_FULL_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "H3_FULL_FEATURE_ORDER_SIGNED.json").write_text(
        json.dumps(SIGNED_NAMES, indent=2), encoding="utf-8"
    )
    (out_dir / "H3_FULL_FEATURE_ORDER_SIGNED_ENERGY.json").write_text(
        json.dumps(SIGNED_ENERGY_NAMES, indent=2), encoding="utf-8"
    )
    if support_rows:
        import csv

        with (out_dir / "H3_FULL_SUPPORT_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(support_rows[0].keys()))
            w.writeheader()
            w.writerows(support_rows)
    (out_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
    log(f"[H3_FULL] ALL COMPLETE in {(time.time()-t0)/3600:.3f}h → {out_dir}")
    return out_dir
