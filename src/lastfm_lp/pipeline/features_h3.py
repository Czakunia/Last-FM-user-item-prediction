"""Materialize pooled H3_RAW (8-D) under outputs/lastfm_full_v2/features/H3_raw/."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.lastfm_lp.binary.cross_fit import CrossFitHCRBundle
from src.lastfm_lp.binary.hcr_triple import (
    H3_RAW_FEATURE_ORDER,
    a111_energy_from_n_jki_matrix,
    history_unordered_pairs,
    pool_h3_raw_matrix,
)
from src.lastfm_lp.binary.hcr_triple_index import BackendName, TripleStatsIndex, build_triple_index
from src.lastfm_lp.binary.user_hcr_aggregation import _truncate_history
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit


class CrossFitTripleBundle:
    def __init__(
        self,
        *,
        full: TripleStatsIndex,
        folds: list[TripleStatsIndex],
        cf: CrossFitHCRBundle,
    ) -> None:
        self.full = full
        self.folds = folds
        self.cf = cf

    def index_for(self, user_id: int, split: str) -> TripleStatsIndex:
        if split in {"val", "valid", "validation", "test"}:
            return self.full
        return self.folds[int(self.cf.user_to_fold[int(user_id)])]


def build_cross_fit_triples(
    bundle: dict[str, Any],
    *,
    backend: BackendName = "bitset_popcount",
    max_history_for_pairs: int = 60,
    cache_size: int = 250_000,
) -> CrossFitTripleBundle:
    cfg = bundle["cfg"]
    model_train = bundle["model_train"]
    n_items = int(len(bundle["popularity"]))
    seed = int(cfg.get("hcr", {}).get("cross_fitting", {}).get("seed", 2026))
    cf = ensure_cross_fit(bundle)

    full = build_triple_index(
        model_train,
        n_items,
        backend=backend,
        max_history_for_pairs=max_history_for_pairs,
        seed=seed,
        cache_size=cache_size,
    )
    folds: list[TripleStatsIndex] = []
    n_folds = cf.n_folds
    user_to_fold = cf.user_to_fold
    for k in range(n_folds):
        keep = {u for u, f in user_to_fold.items() if f != k}
        sub = {u: model_train[u] for u in keep if u in model_train}
        folds.append(
            build_triple_index(
                sub,
                n_items,
                backend=backend,
                max_history_for_pairs=max_history_for_pairs,
                seed=seed,
                cache_size=cache_size,
            )
        )
    return CrossFitTripleBundle(full=full, folds=folds, cf=cf)


def aggregate_h3_user_batch(
    history: set[int],
    candidate_ids: np.ndarray,
    triple_index: TripleStatsIndex,
    *,
    max_history: int = 25,
    cooc_n11_fn=None,
    pairwise_index=None,
) -> np.ndarray:
    """Return (C, 8) H3_RAW pooled features for one user × many candidates."""

    pairs = history_unordered_pairs(history, max_history=max_history)
    cands = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    if pairs.shape[0] == 0 or C == 0:
        return np.zeros((C, 8), dtype=np.float32)

    n_jki = triple_index.n_jki_block(pairs, cands)
    M = pairs.shape[0]
    n_j = triple_index.popularity[pairs[:, 0]].astype(np.int64).reshape(M, 1)
    n_k = triple_index.popularity[pairs[:, 1]].astype(np.int64).reshape(M, 1)
    n_i = triple_index.popularity[cands].astype(np.int64).reshape(1, C)

    if cooc_n11_fn is not None:
        n_jk = np.array(
            [cooc_n11_fn(int(pairs[m, 0]), int(pairs[m, 1])) for m in range(M)],
            dtype=np.int64,
        ).reshape(M, 1)
        n_ji = np.zeros((M, C), dtype=np.int64)
        n_ki = np.zeros((M, C), dtype=np.int64)
        for m in range(M):
            j, k = int(pairs[m, 0]), int(pairs[m, 1])
            for c in range(C):
                i = int(cands[c])
                n_ji[m, c] = cooc_n11_fn(j, i)
                n_ki[m, c] = cooc_n11_fn(k, i)
    elif pairwise_index is not None:
        # HxC blocks for (hist vs cands); n_jk from hist×hist
        hist_ids = np.asarray(sorted({int(x) for x in _truncate_history(history, max_history)}), dtype=np.int64)
        # map item → row in hist_ids
        pos = {int(h): t for t, h in enumerate(hist_ids)}
        n11_hc = pairwise_index.cooccurrence_block(hist_ids, cands)
        n11_hh = pairwise_index.cooccurrence_block(hist_ids, hist_ids)
        n_jk = np.array(
            [n11_hh[pos[int(pairs[m, 0])], pos[int(pairs[m, 1])]] for m in range(M)],
            dtype=np.int64,
        ).reshape(M, 1)
        n_ji = np.zeros((M, C), dtype=np.int64)
        n_ki = np.zeros((M, C), dtype=np.int64)
        for m in range(M):
            n_ji[m] = n11_hc[pos[int(pairs[m, 0])]]
            n_ki[m] = n11_hc[pos[int(pairs[m, 1])]]
    else:
        raise ValueError("need cooc_n11_fn or pairwise_index")

    a, e = a111_energy_from_n_jki_matrix(
        n_jki, n_jk, n_ji, n_ki, n_j, n_k, n_i, triple_index.n_users
    )
    # pool over M for each candidate → (C, 8)
    out = np.zeros((C, 8), dtype=np.float32)
    for c in range(C):
        out[c] = pool_h3_raw_matrix(a[:, c], e[:, c])[0]
    return out


def build_h3_pooled_rows(
    pairs: dict[str, np.ndarray],
    *,
    split: str,
    cf: CrossFitHCRBundle,
    triples: CrossFitTripleBundle,
    model_train: dict[int, set[int]],
    max_history: int = 25,
    out_dir: Path | None = None,
    chunk_size: int = 50_000,
    desc: str = "H3_raw",
) -> np.ndarray:
    n = len(pairs["label"])
    n_cols = 8
    users = pairs["user_id"]
    items = pairs["item_id"]
    labels = pairs["label"]

    out_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"X_{split}.npy"
        done = out_dir / f"COMPLETE_{split}"
        if done.exists() and out_path.exists():
            X = np.load(out_path, mmap_mode="r")
            if X.shape == (n, n_cols):
                print(f"[H3] skip {split}: already complete")
                return X
        X = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=(n, n_cols)
        )
        start = 0
        prog = out_dir / f"progress_{split}.json"
        if prog.exists():
            start = int(json.loads(prog.read_text()).get("next_row", 0))
            print(f"[H3] resume {split} from {start}/{n}")
    else:
        X = np.zeros((n, n_cols), dtype=np.float32)
        start = 0

    pbar = tqdm(total=n, initial=start, desc=desc, mininterval=2.0)
    for chunk_start in range(start, n, int(chunk_size)):
        chunk_end = min(chunk_start + int(chunk_size), n)
        user_groups: dict[int, list[int]] = defaultdict(list)
        for r in range(chunk_start, chunk_end):
            user_groups[int(users[r])].append(r)

        for u, rows in user_groups.items():
            t_idx = triples.index_for(u, split)
            p_idx = cf.index_for(u, split=split)
            base = set(model_train.get(u, ()))
            normal_rows: list[int] = []
            removal: dict[int, list[int]] = defaultdict(list)
            for r in rows:
                i = int(items[r])
                y = int(labels[r])
                if y == 1 and i in base:
                    removal[i].append(r)
                else:
                    normal_rows.append(r)

            if normal_rows:
                r_idx = np.asarray(normal_rows, dtype=np.int64)
                feats = aggregate_h3_user_batch(
                    base,
                    items[r_idx],
                    t_idx,
                    max_history=max_history,
                    pairwise_index=p_idx,
                )
                X[r_idx] = feats
            for cand_i, rlist in removal.items():
                hist = base - {int(cand_i)}
                feat = aggregate_h3_user_batch(
                    hist,
                    np.asarray([cand_i], dtype=np.int64),
                    t_idx,
                    max_history=max_history,
                    pairwise_index=p_idx,
                )[0]
                for r in rlist:
                    X[r] = feat

        if out_dir is not None:
            X.flush()
            (out_dir / f"progress_{split}.json").write_text(
                json.dumps({"next_row": chunk_end, "n_rows": n, "n_cols": n_cols}),
                encoding="utf-8",
            )
        pbar.update(chunk_end - chunk_start)
    pbar.close()

    if out_dir is not None:
        (out_dir / "feature_names.json").write_text(
            json.dumps(H3_RAW_FEATURE_ORDER, indent=2), encoding="utf-8"
        )
        (out_dir / f"COMPLETE_{split}").write_text("ok\n", encoding="utf-8")
        cache_notes = {
            "full": triples.full.cache_stats(),
            "backend": triples.full.backend,
        }
        (out_dir / "cache_stats.json").write_text(
            json.dumps(cache_notes, indent=2), encoding="utf-8"
        )
        return np.load(out_path, mmap_mode="r")
    return X


def materialize_h3_raw(
    bundle: dict[str, Any],
    *,
    out_dir: Path,
    backend: BackendName = "sorted_intersect",
    max_history: int = 25,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cf = ensure_cross_fit(bundle)
    print(f"[H3] building cross-fit triple indices backend={backend} …")
    triples = build_cross_fit_triples(bundle, backend=backend)
    model_train = bundle["model_train"]
    for split in splits:
        key = f"{split}_pairs" if split != "valid" else "val_pairs"
        if split == "val":
            key = "val_pairs"
        pairs = bundle[key]
        print(f"[H3] materialize {split} rows={len(pairs['label']):,} …")
        build_h3_pooled_rows(
            pairs,
            split=split,
            cf=cf,
            triples=triples,
            model_train=model_train,
            max_history=max_history,
            out_dir=out_dir,
            desc=f"H3_{split}",
        )
    meta = {
        "protocol": "LASTFM_FULL_SOTA_V2_HIGHER_ORDER_HCR",
        "feature_order": H3_RAW_FEATURE_ORDER,
        "backend": backend,
        "max_history": max_history,
        "note": "wmean ≡ mean for d=2; no shrinkage in H3_RAW",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return out_dir
