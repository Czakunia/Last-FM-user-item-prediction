#!/usr/bin/env python3
"""03 — Build item A11 neighbourhoods and hard-negative R3 training pairs.

What this script does
---------------------
This is the *final* training-negative protocol of the thesis (not random
negatives).

1. Build item–item A11 / Pearson φ neighbourhoods on model-train
   (``00_build_a11_neighbors.py``).
2. For each positive (u, i+), sample four negatives (R3 mixture):
   two from the 5th–30th percentile band of φ(j, i+), one popularity-weighted,
   one uniform, excluding ``H_u ∪ {i+}``.
3. Materialise A5 / H3 / L2 on those pairs
   (``01_build_hardneg_data.py``) and freeze a SCREEN_3K user list
   (``00_freeze_screen3k_users.py``).

Inputs
------
- model-train splits
- ~500k positives from the frozen train-pair table

Outputs
-------
- ``experiments/lastfm_true_final_hardneg_joint_v1/artifacts/item_a11_neighbors.npz``
- ``.../artifacts/hardneg_train_pairs.npz``
- ``.../artifacts/features/{A,H,L}_train.npy`` and scalers
- ``.../artifacts/SCREEN_3K_USERS.npy``
- ``.../artifacts/DATA_READY.flag``

Seeds for *training* (101, 202, 303, 404, 505) are separate from the
sampler seed that freezes the pair table.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "lastfm_true_final_hardneg_joint_v1"
PY = Path(sys.executable)


def run(script: str) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [str(PY), "-u", str(EXP / "scripts" / script)]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    run("00_build_a11_neighbors.py")
    run("00_freeze_screen3k_users.py")
    run("01_build_hardneg_data.py")
    run("01b_audit_negatives.py")
