#!/usr/bin/env python3
"""04 — Train HGT + A5 + H3 + LEG from scratch (official seeds).

What this script does
---------------------
Trains the frozen joint ranker:

    s(u,X) = MLP([q_struct ‖ A5 ‖ H3]) + δ_LEG(L2)

* HGT: width 64, 2 layers, 2 heads, dropout 0.1, ID embeddings (no SVD feed).
* Pair readout: [e_u ‖ e_X ‖ e_u⊙e_X ‖ |e_u−e_X| ‖ <e_u,e_X>] (257-D).
* Fusion MLP: 265 → 128 → 64 → 1 (GELU, dropout 0.2).
* LEG residual: Linear(1→16)→GELU→Linear(16→1), last layer zero-init.
* Loss: weighted BCE + 0.5 BPR on the R3 table from step 03.
* Optimizer: Adam 1e-3, weight decay 1e-4.
* 20 epochs, checkpoints every epoch, no train-time early stop.
* Official seeds: 101, 202, 303, 404, 505 (set ``TFHN_SEEDS``).

The graph encoder never sees A5/H3; those enter only at the fusion decoder.

Inputs
------
- hard-negative pair table and scaled A/H/L from step 03
- CKG from Last-FM* train + kg_final (max 250_000 KG edges)

Outputs
-------
- ``experiments/lastfm_true_final_hardneg_joint_v1/artifacts/seed{SEED}/checkpoints/epoch_*.pt``

Does not touch the sealed test labels.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "lastfm_true_final_hardneg_joint_v1"
PY = Path(sys.executable)
SEEDS = os.environ.get("TFHN_SEEDS", "101,202,303,404,505")


def main() -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TFHN_SEEDS"] = SEEDS
    env.setdefault("TFHN_MAX_EPOCHS", "20")
    env.setdefault("TFHN_DEVICE", "cpu")
    env.setdefault("TFHN_LAMBDA_BPR", "0.5")
    print(f"TRAIN seeds={SEEDS} device={env['TFHN_DEVICE']}", flush=True)
    subprocess.check_call(
        [str(PY), "-u", str(EXP / "scripts" / "02_train_from_scratch.py")],
        cwd=str(ROOT),
        env=env,
    )


if __name__ == "__main__":
    main()
