#!/usr/bin/env python3
"""LAST_FM_04_fullrank_validation_frozen_checkpoints_20260824

Step 4 — Full-catalog ranking on frozen checkpoints (publication metric).

Scores every item minus H_u in model_train for each validation user.
Uses frozen checkpoints from steps 2–3; does not retrain.

Outputs (JSON/CSV committed when present locally):
  TRUE FINAL     → LASTFM_TRUE_FINAL/NOLEAK_FULLRANK_VALIDATION_V1/
  HGT only       → LASTFM_TRUE_FINAL/NOLEAK_FULLRANK_HGT_ONLY_V1/
  HGT + A5       → LASTFM_TRUE_FINAL/NOLEAK_FULLRANK_HGT_A5_V1/
  HGT + A5 + H3  → LASTFM_TRUE_FINAL/NOLEAK_FULLRANK_HGT_A5_H3_V1/

Do not mix sampled NDCG@20 (~0.87) with full-catalog NDCG@20 (~0.28).
TEST remains locked — never scored.

Prerequisite: checkpoints from steps 2–3.
"""

from __future__ import annotations

from pathlib import Path

from _lastfm_pipeline_common_20260824 import ROOT, run_script

TRUE_FINAL_TRAIN = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_TRAINING_V1"
TRUE_FINAL_FR = ROOT / "LASTFM_TRUE_FINAL" / "NOLEAK_FULLRANK_VALIDATION_V1"
HGT_TRAIN = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HGT_ONLY_RANKER_V1"
HGT_FR = ROOT / "LASTFM_TRUE_FINAL" / "NOLEAK_FULLRANK_HGT_ONLY_V1"
A5_TRAIN = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HGT_A5_RANKER_V1"
A5_FR = ROOT / "LASTFM_TRUE_FINAL" / "NOLEAK_FULLRANK_HGT_A5_V1"
H3_TRAIN = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HGT_A5_H3_NOLEG_V1"
H3_FR = ROOT / "LASTFM_TRUE_FINAL" / "NOLEAK_FULLRANK_HGT_A5_H3_V1"

COMMON = {
    "LASTFM_SKIP_WAIT": "1",
    "LASTFM_CAND_BATCH": "4096",
}


def fullrank(tag: str, train_out: Path, fr_out: Path, extra: dict[str, str]) -> None:
    env = {
        **COMMON,
        "LASTFM_FINAL_OUT": str(train_out),
        "LASTFM_TRUE_FINAL_OUT": str(train_out),
        "LASTFM_FULLRANK_OUT": str(fr_out),
        "LASTFM_TRAIN_WAIT_SCRIPT": "run_lastfm_joint_hgt_a5_ranker_v1.py",
        **extra,
    }
    print(f"[step 4] full-rank {tag} → {fr_out.name}", flush=True)
    run_script("run_lastfm_noleak_fullrank_validation_v1.py", extra_env=env)


if __name__ == "__main__":
    print("[step 4] full-catalog validation (all frozen layers)", flush=True)
    fullrank("TRUE FINAL", TRUE_FINAL_TRAIN, TRUE_FINAL_FR, {})
    fullrank("HGT only", HGT_TRAIN, HGT_FR, {"LASTFM_HGT_ONLY": "1"})
    fullrank("HGT+A5", A5_TRAIN, A5_FR, {"LASTFM_NO_DIST": "1"})
    fullrank("HGT+A5+H3", H3_TRAIN, H3_FR, {"LASTFM_NO_LEG": "1"})
    print("[step 4] complete", flush=True)
