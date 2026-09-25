#!/usr/bin/env python3
"""Freeze fixed SCREEN_3K user list (identical across seeds/epochs)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_true_final_hardneg_joint_v1" / "src"))

from tfhn.paths import ART, SPLITS_DIR  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402

N_SCREEN = 3000


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    va = load_user_sets(SPLITS_DIR / "valid.txt")
    users = np.array(sorted(u for u, items in va.items() if items), dtype=np.int64)
    assert len(users) >= N_SCREEN, f"need >= {N_SCREEN} val users, got {len(users)}"
    screen = users[:N_SCREEN]
    h = hashlib.sha256(screen.tobytes()).hexdigest()
    path = ART / "SCREEN_3K_USERS.npy"
    np.save(path, screen)
    meta = {
        "n": int(len(screen)),
        "sha256": h,
        "selection": "first_N_sorted_valid_users_with_gt",
        "path": str(path),
        "fixed_across_seeds_and_epochs": True,
    }
    (ART / "SCREEN_3K_USERS.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
