#!/usr/bin/env python3
"""LAST_FM_02_train_true_final_joint_model_HGT_A5_H3_LEG_20260824

Step 2 — Train the frozen TRUE FINAL ranker (publication reference model).

Architecture (fixed in config, do not retune):
  HGT 64D / 2 layers / 2 heads + pair256 + graph_dot + A5 + H3 + LEG_K2 residual
  LateFusion decoder 265→128→64→1

Trains seeds 101, 202, 303 on model_train; early stopping on sampled val NDCG@20.
Checkpoints: LASTFM_TRUE_FINAL/JOINT_TRAINING_V1/06_CHECKPOINTS/ (local, gitignored).

Prerequisite: steps 0–1 complete.
One GPU job at a time (MPS on Apple Silicon).
"""

from __future__ import annotations

from _lastfm_pipeline_common_20260824 import run_script

if __name__ == "__main__":
    run_script("run_lastfm_true_final_joint_training_v1.py")
