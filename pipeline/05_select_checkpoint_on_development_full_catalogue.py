#!/usr/bin/env python3
"""05 — Select the checkpoint on development full-catalogue NDCG@20.

What this script does
---------------------
For each seed:

1. Score SCREEN_3K users after trained epochs (cheap proxy).
2. Keep the top-2 SCREEN epochs (best and second).
3. Run **full-catalogue** ranking on the validation split for those epochs
   (~23k users, catalogue minus H_u).
4. Freeze ``argmax`` full-catalogue NDCG@20 as ``selected_epoch``.

This is development selection. The sealed extra file is still not used.

Expected official development full-catalogue NDCG@20 (hard-neg, 5 seeds):
**0.2819 ± 0.0039**.

Inputs
------
- checkpoints from step 04
- SCREEN_3K user list from step 03
- valid.txt

Outputs
-------
- ``.../artifacts/seed{SEED}/FULLCAT.json`` (selected epoch + metrics)
- ``FINAL_TRUE_FINAL_HARDNEG_MANIFEST.json`` after ``04_summarize.py``
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
    for seed in SEEDS.split(","):
        seed = seed.strip()
        print(f"FULLCAT DEV seed={seed}", flush=True)
        env["TFHN_SEED"] = seed
        subprocess.check_call(
            [str(PY), "-u", str(EXP / "scripts" / "03_eval_fullcat.py")],
            cwd=str(ROOT),
            env=env,
        )
    subprocess.check_call(
        [str(PY), "-u", str(EXP / "scripts" / "04_summarize.py")],
        cwd=str(ROOT),
        env=env,
    )


if __name__ == "__main__":
    main()
