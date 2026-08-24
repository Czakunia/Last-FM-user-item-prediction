#!/usr/bin/env python3
"""LAST_FM_03_train_ablation_layers_sampled_val_NDCG_20260824

Step 3 — Progressive ablation training (sampled validation NDCG@20).

Trains three frozen variants for the thesis proxy comparison:
  • HGT only          → LASTFM_TRUE_FINAL/JOINT_HGT_ONLY_RANKER_V1/
  • HGT + A5          → LASTFM_TRUE_FINAL/JOINT_HGT_A5_RANKER_V1/
  • HGT + A5 + H3     → LASTFM_TRUE_FINAL/JOINT_HGT_A5_H3_NOLEG_V1/  (no LEG branch)

Same HGT backbone and training protocol as step 2; only decoder input dims change.
Skips layers whose checkpoints already exist.

Prerequisite: steps 0–1 complete (step 2 optional but recommended first).
Does not overwrite TRUE FINAL from step 2.
"""

from __future__ import annotations

import json
from pathlib import Path

from _lastfm_pipeline_common_20260824 import ROOT, SEEDS, base_env, run_script

HGT_OUT = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HGT_ONLY_RANKER_V1"
A5_OUT = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HGT_A5_RANKER_V1"
H3_OUT = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HGT_A5_H3_NOLEG_V1"


def layer_complete(out: Path) -> bool:
    seed_dirs = {101: "01_SEED101", 202: "02_SEED202", 303: "03_SEED303"}
    return all((out / "06_CHECKPOINTS" / f"final_seed{s}_best.pt").exists() for s in SEEDS) and all(
        (out / seed_dirs[s] / "seed_summary.json").exists() for s in SEEDS
    )


def train_layer(layer: str, out: Path, out_env: str) -> None:
    if layer_complete(out):
        print(f"[step 3] layer={layer} already trained — skip", flush=True)
        return
    run_script(
        "run_lastfm_joint_hgt_a5_ranker_v1.py",
        extra_env={"LASTFM_LAYER": layer, out_env: str(out), "LASTFM_TRUE_FINAL_FINALIZE": "0"},
    )
    if not layer_complete(out):
        raise SystemExit(f"[step 3] layer={layer} finished but checkpoints missing")


def dump_sampled() -> None:
    print("\n========== SAMPLED NDCG@20 (ablation layers) ==========", flush=True)
    for name, out in (("HGT", HGT_OUT), ("HGT+A5", A5_OUT), ("HGT+A5+H3", H3_OUT)):
        print(f"--- {name} ---", flush=True)
        for s, folder in ((101, "01_SEED101"), (202, "02_SEED202"), (303, "03_SEED303")):
            p = out / folder / "seed_summary.json"
            if not p.exists():
                print(f"  seed {s}: MISSING", flush=True)
                continue
            row = json.loads(p.read_text())
            print(
                f"  seed {s}: sampled_NDCG@20={row.get('best_sampled_NDCG@20'):.6f} "
                f"best_ep={row.get('best_epoch')} stop={row.get('stopping_epoch')}",
                flush=True,
            )
    print("TRUE FINAL (step 2) uses LEG; compare its seed_summary separately.", flush=True)
    print("=======================================================\n", flush=True)


if __name__ == "__main__":
    print("[step 3] progressive ablation training", flush=True)
    train_layer("a5", A5_OUT, "LASTFM_HGT_A5_OUT")
    train_layer("hgt", HGT_OUT, "LASTFM_HGT_ONLY_OUT")
    train_layer("h3", H3_OUT, "LASTFM_HGT_H3_OUT")
    dump_sampled()
