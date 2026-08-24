"""Artist mapping, train-only user–artist matrix, leave-one-fold signed A11."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.lastfm_lp.binary.binary_measures import a11_energy_from_n11_matrix
from src.lastfm_lp.binary.cross_fit import assign_user_folds
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.measure_race_selection import deterministic_topk_indices
from src.lastfm_lp.clean_v2.constants import CLEAN_V2_TOP_K
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d_matrix
from src.lastfm_lp.clean_v2.routing import score_matrix_clean_v2

ARTIST_RELATION_URI = "http://rdf.freebase.com/ns/music.recording.artist"
ARTIST_RELATION_ID = 2
N_ITEMS_DEFAULT = 48123


@dataclass
class ArtistMapping:
    n_items: int
    n_artists: int
    artist_entity_ids: np.ndarray  # (n_artists,) original entity remap
    entity_to_remap: dict[int, int]
    artists_of_item: list[np.ndarray]  # item -> artist remaps (int32)
    primary_artist: np.ndarray  # (n_items,) remap or -1
    n_artists_of_item: np.ndarray
    n_items_with_artist: int
    n_items_without_artist: int
    n_single_artist: int
    n_multi_artist: int


@dataclass
class ArtistA11Fold:
    N: int
    pop: np.ndarray  # (n_art,)
    n11: np.ndarray  # (n_art, n_art) int32
    a11: np.ndarray  # (n_art, n_art) float32
    user_artists: dict[int, np.ndarray]


def build_artist_mapping(
    kg_path: Path,
    item_ids: set[int],
    *,
    n_items: int = N_ITEMS_DEFAULT,
    relation_id: int = ARTIST_RELATION_ID,
) -> ArtistMapping:
    item_to: dict[int, set[int]] = defaultdict(set)
    with Path(kg_path).open() as f:
        for line in f:
            h, r, t = line.split()
            if int(r) != int(relation_id):
                continue
            h_i, t_i = int(h), int(t)
            if h_i in item_ids and t_i not in item_ids:
                item_to[h_i].add(t_i)
    all_art = sorted({a for s in item_to.values() for a in s})
    entity_to_remap = {e: i for i, e in enumerate(all_art)}
    artists_of_item: list[np.ndarray] = [np.zeros(0, dtype=np.int32) for _ in range(n_items)]
    primary = np.full(n_items, -1, dtype=np.int32)
    n_of = np.zeros(n_items, dtype=np.int16)
    n_with = n_single = n_multi = 0
    for i, ents in item_to.items():
        if not (0 <= i < n_items):
            continue
        rem = np.asarray(sorted(entity_to_remap[e] for e in ents), dtype=np.int32)
        artists_of_item[i] = rem
        n_of[i] = rem.size
        if rem.size:
            primary[i] = int(rem[0])
            n_with += 1
            if rem.size == 1:
                n_single += 1
            else:
                n_multi += 1
    n_without = int(n_items - n_with)
    return ArtistMapping(
        n_items=n_items,
        n_artists=len(all_art),
        artist_entity_ids=np.asarray(all_art, dtype=np.int32),
        entity_to_remap=entity_to_remap,
        artists_of_item=artists_of_item,
        primary_artist=primary,
        n_artists_of_item=n_of,
        n_items_with_artist=n_with,
        n_items_without_artist=n_without,
        n_single_artist=n_single,
        n_multi_artist=n_multi,
    )


def user_artist_sets(
    model_train: dict[int, set[int]],
    mapping: ArtistMapping,
) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for u, items in model_train.items():
        arts: set[int] = set()
        for i in items:
            if 0 <= int(i) < mapping.n_items:
                arts.update(int(a) for a in mapping.artists_of_item[int(i)].tolist())
        out[int(u)] = arts
    return out


def build_user_artist_binary(
    model_train: dict[int, set[int]],
    mapping: ArtistMapping,
) -> dict[str, Any]:
    ua = user_artist_sets(model_train, mapping)
    n_users = len(model_train)
    n_art = mapping.n_artists
    nnz = int(sum(len(s) for s in ua.values()))
    arts_per_u = np.asarray([len(ua[u]) for u in model_train], dtype=np.int32)
    users_per_a = np.zeros(n_art, dtype=np.int32)
    for s in ua.values():
        for a in s:
            users_per_a[a] += 1
    return {
        "n_users": n_users,
        "n_artists": n_art,
        "nnz": nnz,
        "arts_per_user": arts_per_u,
        "users_per_artist": users_per_a,
        "user_artists": ua,
    }


def _a11_from_n11(n11: np.ndarray, pop: np.ndarray, n_users: int) -> np.ndarray:
    a11, _ = a11_energy_from_n11_matrix(n11, pop, pop, n_users)
    return np.asarray(a11, dtype=np.float64)


def _fold_tables(
    user_artists: dict[int, set[int]],
    keep_users: list[int],
    n_art: int,
) -> ArtistA11Fold:
    ua: dict[int, np.ndarray] = {}
    pop = np.zeros(n_art, dtype=np.int32)
    n11 = np.zeros((n_art, n_art), dtype=np.int32)
    for u in keep_users:
        arts = sorted(user_artists.get(int(u), ()))
        arr = np.asarray(arts, dtype=np.int32)
        ua[int(u)] = arr
        if arr.size == 0:
            continue
        pop[arr] += 1
        n11[np.ix_(arr, arr)] += 1
    N = len(keep_users)
    a11 = _a11_from_n11(n11.astype(np.float64), pop.astype(np.float64), N).astype(np.float64)
    np.fill_diagonal(a11, 0.0)  # never feed A11(a,a)=1 into the model
    return ArtistA11Fold(N=N, pop=pop, n11=n11, a11=a11, user_artists=ua)


def build_artist_crossfit(
    model_train: dict[int, set[int]],
    mapping: ArtistMapping,
    *,
    user_to_fold: dict[int, int] | None = None,
    n_folds: int = 5,
    seed: int = 2026,
) -> tuple[dict[int, int], list[ArtistA11Fold], dict[int, set[int]]]:
    users = sorted(model_train)
    if user_to_fold is None:
        user_to_fold = assign_user_folds(users, n_folds=n_folds, seed=seed)
    ua = user_artist_sets(model_train, mapping)
    folds: list[ArtistA11Fold] = []
    for k in range(n_folds):
        keep = [u for u in users if user_to_fold.get(u, -1) != k]
        folds.append(_fold_tables(ua, keep, mapping.n_artists))
    return user_to_fold, folds, ua


def s_artist_from_sets(
    a_h: np.ndarray,
    a_x: np.ndarray,
    a11: np.ndarray,
) -> tuple[float, str]:
    """Mean A11 over distinct artist pairs. Returns (score, reason)."""
    if a_h.size == 0 or a_x.size == 0:
        return 0.0, "missing_artist"
    vals: list[float] = []
    for a in a_h.tolist():
        for b in a_x.tolist():
            if int(a) == int(b):
                continue
            vals.append(float(a11[int(a), int(b)]))
    if not vals:
        return 0.0, "same_artist_only"
    return float(np.mean(vals)), "ok"


def s_artist_matrix(
    hist: np.ndarray,
    cands: np.ndarray,
    mapping: ArtistMapping,
    fold: ArtistA11Fold,
) -> tuple[np.ndarray, np.ndarray]:
    """(K, C) artist A11 scores + reason codes 0=ok, 1=missing, 2=same_only."""
    hist = np.asarray(hist, dtype=np.int64).reshape(-1)
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    K, C = hist.size, cands.size
    out = np.zeros((K, C), dtype=np.float32)
    reason = np.zeros((K, C), dtype=np.int8)
    if K == 0 or C == 0:
        return out, reason
    prim = mapping.primary_artist
    n_of = mapping.n_artists_of_item
    a11 = fold.a11
    ah = prim[np.clip(hist, 0, mapping.n_items - 1)]
    ax = prim[np.clip(cands, 0, mapping.n_items - 1)]
    nh = n_of[np.clip(hist, 0, mapping.n_items - 1)]
    nx = n_of[np.clip(cands, 0, mapping.n_items - 1)]
    # fast path: both single-artist
    ah2 = ah.reshape(K, 1)
    ax2 = ax.reshape(1, C)
    nh2 = nh.reshape(K, 1)
    nx2 = nx.reshape(1, C)
    both_single = (nh2 == 1) & (nx2 == 1)
    missing = (nh2 == 0) | (nx2 == 0)
    same = both_single & (ah2 == ax2) & (ah2 >= 0)
    ok = both_single & (ah2 >= 0) & (ax2 >= 0) & (ah2 != ax2)
    reason[missing] = 1
    reason[same] = 2
    if ok.any():
        ii, jj = np.nonzero(ok)
        out[ii, jj] = a11[ah[ii], ax[jj]]
    # multi-artist cells
    multi_h = np.flatnonzero(nh != 1)
    multi_c = np.flatnonzero(nx != 1)
    if multi_h.size or multi_c.size:
        for ki in range(K):
            if nh[ki] == 1 and not multi_c.size:
                continue
            a_h = mapping.artists_of_item[int(hist[ki])]
            for cj in range(C):
                if nh[ki] == 1 and nx[cj] == 1:
                    continue
                s, why = s_artist_from_sets(a_h, mapping.artists_of_item[int(cands[cj])], a11)
                out[ki, cj] = s
                reason[ki, cj] = {"ok": 0, "missing_artist": 1, "same_artist_only": 2}[why]
    return out, reason


def select_top25_and_item_a11(
    history_items: set[int] | list[int],
    cands: np.ndarray,
    index: PairwiseStatsIndex,
    *,
    max_history: int = CLEAN_V2_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sel_ids KxC with -1 pad, item_a11 KxC). Routing = signed item A11."""
    hist = np.asarray(sorted(int(h) for h in history_items), dtype=np.int64)
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    K = min(int(max_history), int(hist.size))
    sel = np.full((max(K, 1) if hist.size else 0, C), -1, dtype=np.int64)
    item_a = np.zeros((sel.shape[0], C), dtype=np.float32)
    if hist.size == 0 or C == 0:
        return sel, item_a
    # candidates in history need per-column exclusion
    hset = set(int(x) for x in hist.tolist())
    in_hist = np.array([int(x) in hset for x in cands.tolist()], dtype=bool)
    if (~in_hist).any() and hist.size:
        cols = np.flatnonzero(~in_hist)
        n11 = index.cooccurrence_block(hist, cands[cols])
        a11, _ = a11_energy_from_n11_matrix(
            n11, index.popularity[hist], index.popularity[cands[cols]], index.n_users
        )
        if hist.size <= max_history:
            sel[:, cols] = hist.reshape(-1, 1)
            item_a[:, cols] = a11.astype(np.float32)
        else:
            scores = score_matrix_clean_v2(
                n11,
                index.popularity[hist],
                index.popularity[cands[cols]],
                index.n_users,
                policy="a11_top25",
            )
            idx = deterministic_topk_indices(scores, hist, k=max_history)
            sel[:, cols] = hist[idx]
            item_a[:, cols] = np.take_along_axis(a11, idx, axis=0).astype(np.float32)
    for local, x in enumerate(cands.tolist()):
        if not in_hist[local]:
            continue
        h2 = hist[hist != int(x)]
        if h2.size == 0:
            continue
        n11 = index.cooccurrence_block(h2, np.asarray([int(x)], dtype=np.int64))
        a11, _ = a11_energy_from_n11_matrix(
            n11, index.popularity[h2], index.popularity[np.asarray([int(x)])], index.n_users
        )
        if h2.size <= max_history:
            sel[: h2.size, local] = h2
            item_a[: h2.size, local] = a11[:, 0].astype(np.float32)
        else:
            scores = score_matrix_clean_v2(
                n11,
                index.popularity[h2],
                index.popularity[np.asarray([int(x)])],
                index.n_users,
                policy="a11_top25",
            )
            idx = deterministic_topk_indices(scores, h2, k=max_history)[:, 0]
            sel[:, local] = h2[idx]
            item_a[:, local] = a11[idx, 0].astype(np.float32)
    return sel, item_a


def artist_scores_on_selection(
    sel: np.ndarray,
    cands: np.ndarray,
    mapping: ArtistMapping,
    fold: ArtistA11Fold,
) -> tuple[np.ndarray, dict[str, int]]:
    """(K, C) artist A11 on selected history. Fast primary path + multi fix."""
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    if sel.size == 0 or C == 0:
        return np.zeros((0, C), dtype=np.float32), {"ok": 0, "missing": 0, "same": 0}
    K = int(sel.shape[0])
    scores = np.zeros((K, C), dtype=np.float32)
    prim = mapping.primary_artist
    n_of = mapping.n_artists_of_item
    a11 = fold.a11
    valid = sel >= 0
    ah = np.full((K, C), -1, dtype=np.int32)
    nh = np.zeros((K, C), dtype=np.int16)
    safe = np.clip(sel, 0, mapping.n_items - 1)
    ah[valid] = prim[safe[valid]]
    nh[valid] = n_of[safe[valid]]
    ax = prim[np.clip(cands, 0, mapping.n_items - 1)]
    nx = n_of[np.clip(cands, 0, mapping.n_items - 1)]
    missing = valid & ((nh == 0) | (nx.reshape(1, C) == 0))
    same = valid & (nh == 1) & (nx.reshape(1, C) == 1) & (ah == ax.reshape(1, C)) & (ah >= 0)
    ok = valid & (nh == 1) & (nx.reshape(1, C) == 1) & (ah >= 0) & (ax.reshape(1, C) >= 0) & (ah != ax.reshape(1, C))
    if ok.any():
        ii, jj = np.nonzero(ok)
        scores[ii, jj] = a11[ah[ii, jj], ax[jj]]
    multi = valid & ((nh > 1) | (nx.reshape(1, C) > 1))
    n_multi = 0
    if multi.any():
        ii, jj = np.nonzero(multi)
        for k, c in zip(ii.tolist(), jj.tolist()):
            h = int(sel[k, c])
            x = int(cands[c])
            s, why = s_artist_from_sets(
                mapping.artists_of_item[h], mapping.artists_of_item[x], a11
            )
            scores[k, c] = s
            n_multi += 1
            if why == "missing_artist":
                missing[k, c] = True
            elif why == "same_artist_only":
                same[k, c] = True
    stats = {
        "ok": int(ok.sum()) + n_multi,
        "missing": int(missing.sum()),
        "same": int(same.sum()),
    }
    return scores, stats


def pool_artist_for_selection(
    sel: np.ndarray,
    cands: np.ndarray,
    mapping: ArtistMapping,
    fold: ArtistA11Fold,
) -> tuple[np.ndarray, dict[str, int]]:
    """Pool artist A11 on already-selected Top25 → (C, 3). Pad rows ignored."""
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    scores, stats = artist_scores_on_selection(sel, cands, mapping, fold)
    if scores.size == 0:
        return np.zeros((C, 3), dtype=np.float32), stats
    # pool_signed_a11_3d_matrix uses all rows; zero-pad would shrink the mean.
    # Replace pad with NaN and nanmean-equivalent: copy valid-only per col if needed.
    valid = sel >= 0
    if valid.all():
        return pool_signed_a11_3d_matrix(scores), stats
    s = np.where(valid, scores.astype(np.float64), np.nan)
    out = np.zeros((C, 3), dtype=np.float32)
    nval = valid.sum(axis=0)
    ok = nval > 0
    if ok.any():
        out[ok, 0] = np.nanmean(s[:, ok], axis=0).astype(np.float32)
        out[ok, 1] = np.nanmax(s[:, ok], axis=0).astype(np.float32)
        s_fill = np.where(valid, scores.astype(np.float64), -np.inf)
        order = np.argsort(-s_fill, axis=0)
        kk = min(3, int(scores.shape[0]))
        top = np.take_along_axis(s_fill, order[:kk, :], axis=0)
        rows = np.arange(kk, dtype=np.int64)[:, None]
        keep = rows < np.minimum(nval, kk)[None, :]
        top = np.where(keep, top, np.nan)
        out[ok, 2] = np.nanmean(top[:, ok], axis=0).astype(np.float32)
    return out, stats
