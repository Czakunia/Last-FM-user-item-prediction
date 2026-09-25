#!/usr/bin/env python3
"""06 — Refit on TRAIN_EXTERNAL and evaluate the sealed Last-FM* holdout.

What this script does
---------------------
Honest extra protocol:

* ``TRAIN_EXTERNAL = model_train ∪ valid`` (graph, histories, A5/H3/L2,
  hard-negatives). The sealed ``test.txt`` is never used for fitting.
* From-scratch refit for ``selected_epoch + 1`` epochs (the DEV-selected
  epoch plus one, matching the published protocol).
* Score the full eligible catalogue for every test user
  (n ≈ 23_529, up to 48_123 items).
* Report NDCG@20 / Recall@20 / MRR.

Official 5-seed mean NDCG@20: **0.2015**
(seeds 101/202/303/404/505; see ``expected_results.json``).

Gates (must be set in the environment, same as the thesis freeze):

    GO_TRUE_FINAL_EXTERNAL=YES
    CONFIRM_UNSEAL_LASTFM_TEST=YES

This script does not overwrite a previous official ``07_EXTERNAL`` tree
unless those gates are set. Seeds 404/505 use ``08_run_extra_seeds_external.py``.

Inputs
------
- DEV ``selected_epoch`` per seed from step 05
- Last-FM* ``test.txt`` (first and only use of the holdout)

Outputs
-------
- ``LASTFM_TRUE_FINAL/JOINT_HARDNEG_R3_FROM_SCRATCH_V1/07_EXTERNAL/``
- ``.../08_EXTRA_SEEDS_404_505/`` (seeds 404 and 505)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "lastfm_true_final_hardneg_joint_v1"
PY = Path(sys.executable)


def main() -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env.get("GO_TRUE_FINAL_EXTERNAL") != "YES":
        raise SystemExit("Set GO_TRUE_FINAL_EXTERNAL=YES to unseal extra")
    if env.get("CONFIRM_UNSEAL_LASTFM_TEST") != "YES":
        raise SystemExit("Set CONFIRM_UNSEAL_LASTFM_TEST=YES")
    print("SEALED EXTRA seeds 101/202/303", flush=True)
    subprocess.check_call(
        [str(PY), "-u", str(EXP / "scripts" / "07_run_sealed_external.py")],
        cwd=str(ROOT),
        env=env,
    )
    print("SEALED EXTRA seeds 404/505", flush=True)
    subprocess.check_call(
        [str(PY), "-u", str(EXP / "scripts" / "08_run_extra_seeds_external.py")],
        cwd=str(ROOT),
        env=env,
    )


if __name__ == "__main__":
    main()
