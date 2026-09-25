#!/usr/bin/env python3
"""R3 negative-type audit + hardneg generation manifest (post data build).

Replays PositiveLocalHardNegativeSampler with the SAME seed/order as 01_build
to recover source counts (pairs file may not store sources).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_true_final_hardneg_joint_v1" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_rank_v2" / "src"))

from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402

from rankv2.pos_hard_negatives import PositiveLocalHardNegativeSampler  # noqa: E402
from tfhn.paths import ART, SPLITS_DIR  # noqa: E402


def main() -> None:
    assert (ART / "DATA_READY.flag").exists() or (ART / "hardneg_train_pairs.npz").exists()
    pairs = np.load(ART / "hardneg_train_pairs.npz")
    users, items, labels = pairs["user_id"], pairs["item_id"], pairs["label"]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    assert len(labels) == n_pos * 5, "expected 1 pos + 4 neg blocks"

    seed = int(os.environ.get("TFHN_SEED", "303"))
    nn = np.load(ART / "item_a11_neighbors.npz")
    mt = load_user_sets(SPLITS_DIR / "model_train.txt")
    va = load_user_sets(SPLITS_DIR / "valid.txt")
    te = load_user_sets(SPLITS_DIR / "test.txt") if (SPLITS_DIR / "test.txt").exists() else {}
    cfg = load_protocol_config(PROTOCOL_CONFIG)
    bundle = load_prepared(cfg, verify=False)

    # positives in pair order
    pos_u, pos_i = [], []
    for g in range(n_pos):
        base = g * 5
        assert int(labels[base]) == 1
        pos_u.append(int(users[base]))
        pos_i.append(int(items[base]))
        for k in range(1, 5):
            assert int(labels[base + k]) == 0
            j = int(items[base + k])
            u = int(users[base])
            if j in mt.get(u, set()):
                raise RuntimeError(f"leak: neg {j} in train hist u={u}")
            if j == int(items[base]):
                raise RuntimeError(f"pos==neg u={u} i={j}")

    rng = np.random.default_rng(seed)
    samp = PositiveLocalHardNegativeSampler(
        neighbors=nn["neighbors"],
        neighbor_scores=nn["scores"],
        item_pop=nn["pop"],
        model_train=mt,
        n_items=int(len(nn["pop"])),
        rng=rng,
        band_lo=0.05,
        band_hi=0.30,
    )
    src_counts = Counter()
    set_mismatch = 0
    for g, (u, i) in enumerate(tqdm(zip(pos_u, pos_i), total=len(pos_u), desc="audit-replay")):
        out = samp.sample(u, i, n_neg=4)
        for s in out["sources"]:
            src_counts[s] += 1
        stored = set(int(x) for x in items[g * 5 + 1 : g * 5 + 5])
        got = set(out["items"])
        if stored != got:
            set_mismatch += 1

    # leakage vs val/test positives
    val_pairs = {(u, i) for u, xs in va.items() for i in xs}
    test_pairs = {(u, i) for u, xs in te.items() for i in xs}
    train_pos = {(int(u), int(i)) for u, i in zip(pos_u, pos_i)}
    leak_val = len(train_pos & val_pairs)
    leak_test = len(train_pos & test_pairs)

    per_pos = {k: src_counts[k] / max(n_pos, 1) for k in ("hard_local", "pop", "random")}
    manifest = {
        "sampler": "R3_positive_local_A11_semi_hard_5_30_plus_pop_random",
        "band": [0.05, 0.30],
        "target_mix_per_pos": {"hard_local": 2, "pop": 1, "random": 1},
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_neg_per_pos": 4,
        "source_counts": dict(src_counts),
        "source_mean_per_pos": per_pos,
        "replay_set_mismatch_groups": int(set_mismatch),
        "leak_neg_in_user_hist": 0,
        "leak_train_pos_in_val": int(leak_val),
        "leak_train_pos_in_test": int(leak_test),
        "external_labels_used": False,
        "seed": seed,
        "positives_source": "bundle.train_pairs (TRUE FINAL)",
        "PASS_MIX_APPROX": (
            abs(per_pos.get("hard_local", 0) - 2.0) < 0.15
            and abs(per_pos.get("pop", 0) - 1.0) < 0.15
            and abs(per_pos.get("random", 0) - 1.0) < 0.15
        ),
        "NOTE": (
            "source counts from RNG replay with TFHN_SEED; "
            "set_mismatch may be >0 if dedupe/fallback order differs"
        ),
    }
    (ART / "HARDNEG_GENERATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if leak_val or leak_test:
        raise SystemExit("leakage detected vs val/test positives")


if __name__ == "__main__":
    main()
