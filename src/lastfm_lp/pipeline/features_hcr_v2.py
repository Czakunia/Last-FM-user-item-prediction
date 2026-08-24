"""Materialize Stage-H / HT pooled orthonormal-HCR side features."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.lastfm_lp.binary.cross_fit import CrossFitHCRBundle, build_cross_fit_bundle, bundle_meta
from src.lastfm_lp.binary.user_hcr_aggregation import (
    BlockKind,
    aggregate_a11_energy_user_batch,
    aggregate_a11_energy_vectorized,
    aggregate_block_raw,
    raw_pool_feature_names,
)

STAGE_BLOCK: dict[str, BlockKind] = {
    "HT0": "legacy_scalar",
    "HT1": "a11",
    "HT2": "compact8",
    "HT3": "compact8",  # PairMLP path materializes history bank separately
    "H0": "legacy_scalar",
    "H1": "a11",
    "H2": "a11_energy",
    "H3": "compact8",
    "H4": "compact8",
    "H5": "compact8",
    "H6": "compact8",
}

# Fast path: batch CSR gather + vectorized a11 (same pooling / protocol).
_FAST_KINDS = frozenset({"a11", "a11_energy", "legacy_scalar"})


def _hist_for_row(
    u: int,
    i: int,
    y: int,
    model_train: dict[int, set[int]],
) -> set[int]:
    hist = set(model_train.get(u, ()))
    if y == 1 and i in hist:
        hist = hist - {i}
    return hist


def ensure_cross_fit(bundle: dict[str, Any]) -> CrossFitHCRBundle:
    if "cross_fit" in bundle:
        return bundle["cross_fit"]
    cfg = bundle["cfg"]
    n_items = int(bundle["popularity"].shape[0])
    hcfg = cfg.get("hcr", {})
    cf_cfg = cfg.get("cross_fitting") or hcfg.get("cross_fitting") or {}
    cache_dir = Path(cfg["paths"]["features"]) / "cross_fit_cache"
    print("[H] building/loading 5-fold cross-fit HCR indices …")
    cf = build_cross_fit_bundle(
        bundle["model_train"],
        n_items,
        n_folds=int(cf_cfg.get("n_user_folds", 5)),
        seed=int(cf_cfg.get("seed", cfg["split"]["seed"])),
        smoothing=float(hcfg.get("smoothing_for_unstable_channels", hcfg.get("smoothing", 0.5))),
        max_history_for_pairs=60,
        full_index=bundle["index"],
        cache_dir=cache_dir,
    )
    meta_path = Path(cfg["paths"]["features"]) / "cross_fit_meta.json"
    meta_path.write_text(json.dumps(bundle_meta(cf), indent=2), encoding="utf-8")
    fold_path = Path(cfg["paths"]["features"]) / "user_folds.json"
    fold_path.write_text(json.dumps({str(k): v for k, v in cf.user_to_fold.items()}), encoding="utf-8")
    bundle["cross_fit"] = cf
    return cf


def _needs_uncertainty(kind: BlockKind) -> bool:
    return kind in {"compact8", "compact10"}


def _pair_fn_factory(
    cf: CrossFitHCRBundle,
    user_id: int,
    split: str,
    *,
    kind: BlockKind = "a11",
    drop_a11: bool = False,
    shuffle_a11: dict[tuple[int, int], float] | None = None,
    uncertainty_mode: str = "support_proxy",
):
    """pair lookup (legacy row-wise path; kept for non-fast kinds + regression).

    ``uncertainty_mode``:
      - ``support_proxy`` (default, fast): 1/sqrt(n11+1) — no 5-fold SD
      - ``cross_fold_sd``: expensive SD of a11 across leave-one-fold indices
    """
    from dataclasses import replace

    use_unc = _needs_uncertainty(kind) and uncertainty_mode == "cross_fold_sd"

    def pair_fn(j: int, i: int):
        if use_unc:
            m = cf.pair_with_uncertainty(user_id, j, i, split=split)
        else:
            m = cf.pair(user_id, j, i, split=split)
            if _needs_uncertainty(kind) and m.uncertainty == 0.0:
                # cheap, leakage-safe proxy (same as Stage F)
                m = replace(m, uncertainty=float(1.0 / np.sqrt(m.n11 + 1.0)))
        if shuffle_a11 is not None:
            key = (min(j, i), max(j, i))
            if key in shuffle_a11:
                a = float(shuffle_a11[key])
                m = replace(m, hcr_a11=a, hcr_energy=a * a, phi=a)
        if drop_a11:
            m = replace(m, hcr_a11=0.0, hcr_energy=0.0, phi=0.0)
        return m

    return pair_fn


def _progress_path(out_dir: Path, split: str) -> Path:
    return out_dir / f"X_{split}.progress.json"


def _split_array_path(out_dir: Path, split: str) -> Path:
    return out_dir / f"X_{split}.npy"


def _write_progress(
    out_dir: Path,
    split: str,
    *,
    next_row: int,
    n_rows: int,
    n_cols: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomic progress write (tmp → rename). ``next_row`` == completed_until_row."""

    payload: dict[str, Any] = {
        "completed_until_row": int(next_row),
        "next_row": int(next_row),  # backward-compatible alias
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "split": split,
        "output_shape": [int(n_rows), int(n_cols)],
    }
    if extra:
        payload.update(extra)
    path = _progress_path(out_dir, split)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _read_progress(out_dir: Path, split: str) -> dict[str, Any] | None:
    path = _progress_path(out_dir, split)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    # normalize next_row
    if "next_row" not in raw and "completed_until_row" in raw:
        raw["next_row"] = raw["completed_until_row"]
    return raw


def _split_is_complete(out_dir: Path, split: str, n_rows: int, n_cols: int) -> bool:
    path = _split_array_path(out_dir, split)
    if not path.exists():
        return False
    prog = _read_progress(out_dir, split)
    if prog is not None:
        return (
            int(prog.get("next_row", -1)) == n_rows
            and int(prog.get("n_rows", -1)) == n_rows
            and int(prog.get("n_cols", -1)) == n_cols
        )
    # Legacy full save (no progress file): accept matching shape.
    try:
        X = np.load(path, mmap_mode="r")
        return tuple(X.shape) == (n_rows, n_cols)
    except Exception:
        return False


def _open_memmap_split(
    out_dir: Path,
    split: str,
    n_rows: int,
    n_cols: int,
) -> tuple[np.memmap, int]:
    """Open/create memmap; return (array, start_row) for resume."""

    path = _split_array_path(out_dir, split)
    prog = _read_progress(out_dir, split)
    if path.exists() and prog is not None:
        if prog.get("n_rows") == n_rows and prog.get("n_cols") == n_cols:
            X = np.load(path, mmap_mode="r+")
            start = int(min(max(prog.get("next_row", 0), 0), n_rows))
            return X, start
    # Fresh or incompatible → recreate
    X = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(n_rows, n_cols)
    )
    X[:] = 0
    X.flush()
    _write_progress(out_dir, split, next_row=0, n_rows=n_rows, n_cols=n_cols)
    return X, 0


def _index_key_for_user(cf: CrossFitHCRBundle, user_id: int, split: str) -> int:
    """Stable key: -1 = full_index, else fold id."""

    if split in {"val", "valid", "validation", "test"}:
        return -1
    fold = cf.user_to_fold.get(int(user_id))
    if fold is None:
        return -1
    return int(fold)


def _index_from_key(cf: CrossFitHCRBundle, key: int):
    if key < 0:
        return cf.full_index
    return cf.fold_indices[key]


def build_pooled_rows_legacy(
    pairs: dict[str, np.ndarray],
    *,
    split: str,
    cf: CrossFitHCRBundle,
    model_train: dict[int, set[int]],
    kind: BlockKind,
    max_history: int,
    drop_a11: bool = False,
    shuffle_a11: dict[tuple[int, int], float] | None = None,
    uncertainty_mode: str = "support_proxy",
    row_indices: np.ndarray | None = None,
    desc: str = "hcr_legacy",
) -> np.ndarray:
    """Original row-wise path (per-row ``_pair_fn_factory`` + ``cf.pair``)."""

    names = raw_pool_feature_names(kind)
    if row_indices is None:
        row_indices = np.arange(len(pairs["label"]), dtype=np.int64)
    X = np.zeros((len(row_indices), len(names)), dtype=np.float32)
    for t, r in enumerate(tqdm(row_indices, desc=desc, mininterval=2.0)):
        r = int(r)
        u = int(pairs["user_id"][r])
        i = int(pairs["item_id"][r])
        y = int(pairs["label"][r])
        hist = _hist_for_row(u, i, y, model_train)
        pair_fn = _pair_fn_factory(
            cf,
            u,
            split,
            kind=kind,
            drop_a11=drop_a11,
            shuffle_a11=shuffle_a11,
            uncertainty_mode=uncertainty_mode,
        )
        X[t] = aggregate_block_raw(
            hist,
            i,
            cf.full_index,
            max_history=max_history,
            kind=kind,
            pair_fn=pair_fn,
        )
    return X


def build_pooled_rows_fast_v1(
    pairs: dict[str, np.ndarray],
    *,
    split: str,
    cf: CrossFitHCRBundle,
    model_train: dict[int, set[int]],
    kind: BlockKind,
    max_history: int,
    drop_a11: bool = False,
    shuffle_a11: dict[tuple[int, int], float] | None = None,
    out_dir: Path | None = None,
    chunk_size: int = 65_536,
    desc: str = "hcr_fast_v1",
) -> np.ndarray:
    """FAST V1: group by (index, candidate); per-row vectorized history gather."""

    if kind not in _FAST_KINDS:
        raise ValueError(f"fast path does not support kind={kind}")

    n = len(pairs["label"])
    n_cols = len(raw_pool_feature_names(kind))
    users = pairs["user_id"]
    items = pairs["item_id"]
    labels = pairs["label"]

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if _split_is_complete(out_dir, split, n, n_cols):
            print(f"[H] skip {split}: already complete ({n}×{n_cols})")
            return np.load(_split_array_path(out_dir, split), mmap_mode="r")
        X, start = _open_memmap_split(out_dir, split, n, n_cols)
        if start > 0:
            print(f"[H] resume {split} from row {start}/{n}")
    else:
        X = np.zeros((n, n_cols), dtype=np.float32)
        start = 0

    pbar = tqdm(total=n, initial=start, desc=desc, mininterval=2.0)
    for chunk_start in range(start, n, int(chunk_size)):
        chunk_end = min(chunk_start + int(chunk_size), n)
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for r in range(chunk_start, chunk_end):
            u = int(users[r])
            i = int(items[r])
            key = (_index_key_for_user(cf, u, split), i)
            groups[key].append(r)

        for (idx_key, cand), rows in groups.items():
            index = _index_from_key(cf, idx_key)
            row_cols, row_data = index.cooc_row(int(cand))
            for r in rows:
                u = int(users[r])
                i = int(items[r])
                y = int(labels[r])
                hist = _hist_for_row(u, i, y, model_train)
                X[r] = aggregate_a11_energy_vectorized(
                    hist,
                    i,
                    index,
                    max_history=max_history,
                    kind=kind,
                    drop_a11=drop_a11,
                    shuffle_a11=shuffle_a11,
                    row_cols=row_cols,
                    row_data=row_data,
                )
        if out_dir is not None:
            X.flush()
            _write_progress(
                out_dir,
                split,
                next_row=chunk_end,
                n_rows=n,
                n_cols=n_cols,
                extra={"engine": "fast_v1", "feature_kind": kind},
            )
        pbar.update(chunk_end - chunk_start)
    pbar.close()

    if out_dir is not None:
        _write_progress(
            out_dir,
            split,
            next_row=n,
            n_rows=n,
            n_cols=n_cols,
            extra={"engine": "fast_v1", "feature_kind": kind},
        )
        return np.load(_split_array_path(out_dir, split), mmap_mode="r")
    return X


# Backward-compatible alias
build_pooled_rows_fast = build_pooled_rows_fast_v1


def build_pooled_rows_fast_v2(
    pairs: dict[str, np.ndarray],
    *,
    split: str,
    cf: CrossFitHCRBundle,
    model_train: dict[int, set[int]],
    kind: BlockKind,
    max_history: int,
    drop_a11: bool = False,
    shuffle_a11: dict[tuple[int, int], float] | None = None,
    out_dir: Path | None = None,
    chunk_size: int = 100_000,
    desc: str = "hcr_fast_v2",
) -> np.ndarray:
    """FAST V2: group by user; HxC cooc block + matrix a11/pool.

    Rows with ``y==1`` and candidate ∈ model_train use ``hist = base - {i}``
    (identical to ``_hist_for_row``); all other rows of the user share ``base``.
    """

    if kind not in _FAST_KINDS:
        raise ValueError(f"fast_v2 does not support kind={kind}")

    n = len(pairs["label"])
    n_cols = len(raw_pool_feature_names(kind))
    users = pairs["user_id"]
    items = pairs["item_id"]
    labels = pairs["label"]

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if _split_is_complete(out_dir, split, n, n_cols):
            print(f"[H] skip {split}: already complete ({n}×{n_cols})")
            return np.load(_split_array_path(out_dir, split), mmap_mode="r")
        X, start = _open_memmap_split(out_dir, split, n, n_cols)
        if start > 0:
            print(f"[H] resume {split} from row {start}/{n}")
    else:
        X = np.zeros((n, n_cols), dtype=np.float32)
        start = 0

    pbar = tqdm(total=n, initial=start, desc=desc, mininterval=2.0)
    for chunk_start in range(start, n, int(chunk_size)):
        chunk_end = min(chunk_start + int(chunk_size), n)
        # Group row indices by user (preserve original row ids for write-back).
        user_groups: dict[int, list[int]] = defaultdict(list)
        for r in range(chunk_start, chunk_end):
            user_groups[int(users[r])].append(r)

        for u, rows in user_groups.items():
            idx_key = _index_key_for_user(cf, u, split)
            index = _index_from_key(cf, idx_key)
            base = set(model_train.get(u, ()))
            # Split rows: positives whose candidate is in model_train need
            # hist = base - {i} (same rule as _hist_for_row). Others share base.
            normal_rows: list[int] = []
            removal: dict[int, list[int]] = defaultdict(list)  # candidate → rows
            for r in rows:
                i = int(items[r])
                y = int(labels[r])
                if y == 1 and i in base:
                    removal[i].append(r)
                else:
                    normal_rows.append(r)

            if normal_rows:
                r_idx = np.asarray(normal_rows, dtype=np.int64)
                feats = aggregate_a11_energy_user_batch(
                    base,
                    items[r_idx],
                    index,
                    max_history=max_history,
                    kind=kind,
                    drop_a11=drop_a11,
                    shuffle_a11=shuffle_a11,
                )
                X[r_idx] = feats

            for cand_i, rlist in removal.items():
                hist = base - {int(cand_i)}
                feat = aggregate_a11_energy_user_batch(
                    hist,
                    np.asarray([cand_i], dtype=np.int64),
                    index,
                    max_history=max_history,
                    kind=kind,
                    drop_a11=drop_a11,
                    shuffle_a11=shuffle_a11,
                )[0]
                for r in rlist:
                    X[r] = feat

        if out_dir is not None:
            X.flush()
            _write_progress(
                out_dir,
                split,
                next_row=chunk_end,
                n_rows=n,
                n_cols=n_cols,
                extra={"engine": "fast_v2", "feature_kind": kind},
            )
        pbar.update(chunk_end - chunk_start)
    pbar.close()

    if out_dir is not None:
        _write_progress(
            out_dir,
            split,
            next_row=n,
            n_rows=n,
            n_cols=n_cols,
            extra={"engine": "fast_v2", "feature_kind": kind},
        )
        return np.load(_split_array_path(out_dir, split), mmap_mode="r")
    return X


def _resolve_engine(cfg: dict[str, Any], split: str, kind: BlockKind) -> str:
    mat = cfg.get("hcr_materialization") or {}
    default = str(mat.get("engine", "fast_v2"))
    if kind not in _FAST_KINDS:
        return "legacy"
    if split == "train":
        return str(mat.get("train_engine", "fast_v1"))
    return default


def materialize_pooled_sides(
    bundle: dict[str, Any],
    stage: str,
    *,
    tag: str | None = None,
    drop_a11: bool = False,
    shuffle_a11: dict[tuple[int, int], float] | None = None,
    use_fast_path: bool | None = None,
    chunk_size: int | None = None,
    engine: str | None = None,
) -> dict[str, Any]:
    """Build mean/max/top3/wmean pooled vectors for each split.

    Engines: ``legacy`` | ``fast_v1`` | ``fast_v2`` (config ``hcr_materialization``).
    Train defaults to ``fast_v1`` (self-removal); val/test to ``fast_v2``.
    """

    kind = STAGE_BLOCK[stage]
    cfg = bundle["cfg"]
    mat_cfg = cfg.get("hcr_materialization") or {}
    max_history = int(cfg.get("models", {}).get("stage_h", {}).get("max_history", 25))
    unc_mode = str(cfg.get("hcr", {}).get("uncertainty", "support_proxy"))
    if unc_mode == "cross_fold_sd_a11":
        unc_mode = "cross_fold_sd"
    tag = tag or f"{stage}_{kind}"
    out_dir = Path(cfg["paths"]["features"]) / "stage_h" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    names = raw_pool_feature_names(kind)
    names_path = out_dir / "feature_names.json"
    names_path.write_text(json.dumps(names), encoding="utf-8")

    n_train = len(bundle["train_pairs"]["label"])
    n_val = len(bundle["val_pairs"]["label"])
    n_test = len(bundle["test_pairs"]["label"])
    n_cols = len(names)
    complete_marker = out_dir / "COMPLETE"
    if (
        complete_marker.exists()
        and _split_is_complete(out_dir, "train", n_train, n_cols)
        and _split_is_complete(out_dir, "val", n_val, n_cols)
        and _split_is_complete(out_dir, "test", n_test, n_cols)
    ):
        return {
            "X_train": np.load(out_dir / "X_train.npy"),
            "X_val": np.load(out_dir / "X_val.npy"),
            "X_test": np.load(out_dir / "X_test.npy"),
            "y_train": bundle["train_pairs"]["label"],
            "y_val": bundle["val_pairs"]["label"],
            "y_test": bundle["test_pairs"]["label"],
            "feature_names": names,
            "kind": kind,
        }

    cf = ensure_cross_fit(bundle)
    if chunk_size is None:
        chunk_size = int(mat_cfg.get("chunk_size", 100_000))
    if use_fast_path is False:
        force_engine = "legacy"
    else:
        force_engine = engine

    print(
        f"[H] materialize {tag} kind={kind} uncertainty={unc_mode} "
        f"engine_cfg={mat_cfg.get('engine', 'fast_v2')} chunk={chunk_size}"
    )

    def build(pairs: dict[str, np.ndarray], split: str) -> np.ndarray:
        eng = force_engine or _resolve_engine(cfg, split, kind)
        print(f"[H] {split}: engine={eng}")
        use_memmap = bool(mat_cfg.get("memmap", True))
        od = out_dir if use_memmap else None

        if eng == "legacy" or kind not in _FAST_KINDS:
            if od is not None and _split_is_complete(od, split, len(pairs["label"]), n_cols):
                print(f"[H] skip {split}: already complete")
                return np.load(_split_array_path(od, split), mmap_mode="r")
            if od is None:
                return build_pooled_rows_legacy(
                    pairs,
                    split=split,
                    cf=cf,
                    model_train=bundle["model_train"],
                    kind=kind,
                    max_history=max_history,
                    drop_a11=drop_a11,
                    shuffle_a11=shuffle_a11,
                    uncertainty_mode=unc_mode,
                    desc=f"{tag} {split}",
                )
            Xmm, start = _open_memmap_split(od, split, len(pairs["label"]), n_cols)
            nn = len(pairs["label"])
            for chunk_start in range(start, nn, int(chunk_size)):
                chunk_end = min(chunk_start + int(chunk_size), nn)
                rows = np.arange(chunk_start, chunk_end, dtype=np.int64)
                Xmm[chunk_start:chunk_end] = build_pooled_rows_legacy(
                    pairs,
                    split=split,
                    cf=cf,
                    model_train=bundle["model_train"],
                    kind=kind,
                    max_history=max_history,
                    drop_a11=drop_a11,
                    shuffle_a11=shuffle_a11,
                    uncertainty_mode=unc_mode,
                    row_indices=rows,
                    desc=f"{tag} {split}[{chunk_start}:{chunk_end}]",
                )
                Xmm.flush()
                _write_progress(
                    od,
                    split,
                    next_row=chunk_end,
                    n_rows=nn,
                    n_cols=n_cols,
                    extra={"engine": "legacy", "feature_kind": kind},
                )
            _write_progress(
                od,
                split,
                next_row=nn,
                n_rows=nn,
                n_cols=n_cols,
                extra={"engine": "legacy", "feature_kind": kind},
            )
            return np.load(_split_array_path(od, split), mmap_mode="r")

        builder = build_pooled_rows_fast_v2 if eng == "fast_v2" else build_pooled_rows_fast_v1
        return builder(
            pairs,
            split=split,
            cf=cf,
            model_train=bundle["model_train"],
            kind=kind,
            max_history=max_history,
            drop_a11=drop_a11,
            shuffle_a11=shuffle_a11,
            out_dir=od,
            chunk_size=int(chunk_size),
            desc=f"{tag} {split}",
        )

    Xtr = build(bundle["train_pairs"], "train")
    Xva = build(bundle["val_pairs"], "val")
    Xte = build(bundle["test_pairs"], "test")
    complete_marker.write_text("ok\n", encoding="utf-8")
    return {
        "X_train": np.asarray(Xtr),
        "X_val": np.asarray(Xva),
        "X_test": np.asarray(Xte),
        "y_train": bundle["train_pairs"]["label"],
        "y_val": bundle["val_pairs"]["label"],
        "y_test": bundle["test_pairs"]["label"],
        "feature_names": names,
        "kind": kind,
    }


def build_a11_shuffle_map(
    bundle: dict[str, Any],
    *,
    n_buckets: int = 20,
    seed: int = 2026,
) -> dict[tuple[int, int], float]:
    """Shuffle a11 inside buckets of similar n11 and popularity."""

    rng = np.random.default_rng(seed)
    cf = ensure_cross_fit(bundle)
    pop = bundle["popularity"]
    # Collect unique history–candidate pairs from train table (capped).
    pairs = bundle["train_pairs"]
    seen: dict[tuple[int, int], tuple[float, int, int, int]] = {}
    for r in range(len(pairs["label"])):
        u = int(pairs["user_id"][r])
        i = int(pairs["item_id"][r])
        y = int(pairs["label"][r])
        hist = _hist_for_row(u, i, y, bundle["model_train"])
        for j in list(hist)[:25]:
            key = (min(j, i), max(j, i))
            if key in seen:
                continue
            m = cf.full_index.pair(j, i)
            seen[key] = (m.hcr_a11, m.n11, int(pop[j]), int(pop[i]))
        if len(seen) >= 50_000:
            break

    keys = list(seen.keys())
    a11s = np.array([seen[k][0] for k in keys], dtype=np.float64)
    n11s = np.array([seen[k][1] for k in keys], dtype=np.float64)
    # bucket by log1p n11
    edges = np.quantile(n11s, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1
    edges[-1] += 1
    buckets = np.digitize(n11s, edges) - 1
    shuffled = a11s.copy()
    for b in range(n_buckets):
        idx = np.where(buckets == b)[0]
        if len(idx) < 2:
            continue
        shuffled[idx] = rng.permutation(a11s[idx])
    return {keys[t]: float(shuffled[t]) for t in range(len(keys))}
