#!/usr/bin/env python3
"""Multi-seed summary + FINAL_TRUE_FINAL_HARDNEG_MANIFEST. STOP before external."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_true_final_hardneg_joint_v1" / "src"))

from tfhn.ckpt_io import copy_scalers_to_audit, ensure_joint_dirs, write_json  # noqa: E402
from tfhn.paths import ARCH, ART, B0_MEAN, B0_SEED303, JOINT_OUT, REP  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    d = ensure_joint_dirs()
    copy_scalers_to_audit(ART / "features")
    rows = []
    for seed in (101, 202, 303):
        p = ART / f"seed{seed}" / "FULLCAT.json"
        assert p.exists(), f"missing {p}"
        fc = json.loads(p.read_text())
        proof = json.loads((ART / f"seed{seed}" / "FROM_SCRATCH_PROOF.json").read_text())
        init_a = json.loads((ART / f"seed{seed}" / "INIT_AUDIT.json").read_text())
        ckpt = d["ckpts"] / f"final_seed{seed}_best.pt"
        assert ckpt.exists()
        payload = torch.load(ckpt, map_location="cpu")
        assert payload.get("selection") == "fullcat_DEV_NDCG@20"
        assert proof["INIT_MODE"] == "FRESH_RANDOM"
        rows.append(
            {
                "seed": seed,
                "selected_epoch": fc["selected_epoch"],
                "screen3k_NDCG@20": fc["screen3k_NDCG@20"],
                "NDCG@20": fc["NDCG@20"],
                "Recall@20": fc["Recall@20"],
                "MRR": fc["MRR"],
                "delta_vs_0.2719": fc["delta_vs_B0_seed303"],
                "delta_vs_~0.276": fc["delta_vs_B0_mean"],
                "ckpt": str(ckpt),
                "ckpt_sha256": file_sha256(ckpt),
                "init_param_checksum_sha256": init_a["init_param_checksum_sha256"],
                "frozen_name": fc.get("frozen_name"),
                "FROM_SCRATCH": True,
            }
        )

    nd = np.array([r["NDCG@20"] for r in rows], dtype=np.float64)
    rec = np.array([r["Recall@20"] for r in rows], dtype=np.float64)
    mrr = np.array([r["MRR"] for r in rows], dtype=np.float64)

    neg_man = {}
    if (ART / "HARDNEG_GENERATION_MANIFEST.json").exists():
        neg_man = json.loads((ART / "HARDNEG_GENERATION_MANIFEST.json").read_text())
    screen_meta = json.loads((ART / "SCREEN_3K_USERS.json").read_text())
    feat_meta = {}
    if (ART / "features" / "meta.json").exists():
        feat_meta = json.loads((ART / "features" / "meta.json").read_text())

    summary = {
        "per_seed": rows,
        "mean_NDCG@20": float(nd.mean()),
        "std_NDCG@20": float(nd.std(ddof=1)) if len(nd) > 1 else 0.0,
        "mean_Recall@20": float(rec.mean()),
        "std_Recall@20": float(rec.std(ddof=1)) if len(rec) > 1 else 0.0,
        "mean_MRR": float(mrr.mean()),
        "std_MRR": float(mrr.std(ddof=1)) if len(mrr) > 1 else 0.0,
        "delta_mean_vs_0.2719": float(nd.mean() - B0_SEED303["NDCG@20"]),
        "delta_mean_vs_~0.276": float(nd.mean() - B0_MEAN["NDCG@20"]),
        "NO_BEST_SEED_PICK": True,
        "EXTERNAL_SEEN": False,
    }
    write_json(ART / "SUMMARY.json", summary)

    manifest = {
        "name": "FINAL_TRUE_FINAL_HARDNEG_MANIFEST",
        "architecture": ARCH,
        "training": "FROM_SCRATCH",
        "TRAINED_FROM_SCRATCH": True,
        "negative_recipe": "R3",
        "hard_negative_range": [0.05, 0.30],
        "negative_mix_per_pos": {"hard_A11": 2, "popularity": 1, "random": 1},
        "positive_count": neg_man.get("n_pos") or feat_meta.get("n_pos"),
        "loss": "BCE + λ_BPR * BPR",
        "BCE_coefficient": 1.0,
        "BPR_coefficient": 0.5,
        "optimizer": "Adam",
        "LR": 1e-3,
        "WD": 1e-4,
        "batch_size_BCE": 4096,
        "batch_size_BPR": 1024,
        "max_epochs": 20,
        "train_early_stop": False,
        "screen3k_eval_early_stop": {
            "enabled": True,
            "rule": "after min_epoch, stop if last patience consecutive epoch-to-epoch NDCG@20 gains are all < min_delta",
            "min_epoch": 5,
            "patience": 3,
            "min_delta": 0.002,
        },
        "early_stop": False,
        "screen3k_n": screen_meta["n"],
        "screen3k_users_sha256": screen_meta["sha256"],
        "checkpoint_selection_rule": (
            "SCREEN_3K until eval early-stop → FULLCAT on top-2 SCREEN epochs "
            "(rule B; seed101 may retain earlier broader candidate set) → argmax NDCG@20"
        ),
        "per_seed": rows,
        "mean_plus_std": {
            "NDCG@20": f"{summary['mean_NDCG@20']:.6f} ± {summary['std_NDCG@20']:.6f}",
            "Recall@20": f"{summary['mean_Recall@20']:.6f} ± {summary['std_Recall@20']:.6f}",
            "MRR": f"{summary['mean_MRR']:.6f} ± {summary['std_MRR']:.6f}",
        },
        "refs": {"TRUE_FINAL_seed303": 0.2719, "TRUE_FINAL_mean": 0.2764},
        "hardneg_generation_manifest": neg_man,
        "joint_out": str(JOINT_OUT),
        "checkpoints_dir": str(d["ckpts"]),
        "preprocessing": "A5/H3/LEG rematerialized for hardneg rows; scalers fit on hardneg train",
        "code_git_commit": git_commit(),
        "DEV_SELECTION_COMPLETE": True,
        "MODEL_FROZEN": True,
        "EXTERNAL_SEEN": False,
        "TEST_STATUS": "LOCKED_NOT_RUN",
        "requires_to_unseal": "GO_TRUE_FINAL_EXTERNAL=YES",
        "NOTE_GO_EXTERNAL": "Generic GO_EXTERNAL is NOT sufficient for this track",
        "timestamp": utc_now(),
    }
    man_path = ART / "FINAL_TRUE_FINAL_HARDNEG_MANIFEST.json"
    write_json(man_path, manifest)
    write_json(JOINT_OUT / "FINAL_TRUE_FINAL_HARDNEG_MANIFEST.json", manifest)
    write_json(REP / "FINAL_TRUE_FINAL_HARDNEG_MANIFEST.json", manifest)
    man_hash = file_sha256(man_path)

    hold = {
        "external": "LOCKED",
        "EXTERNAL_SEEN": False,
        "requires": "GO_TRUE_FINAL_EXTERNAL=YES",
        "generic_GO_EXTERNAL_insufficient": True,
        "manifest": str(man_path),
        "manifest_sha256": man_hash,
        "message": "STOP. Wait for GO_TRUE_FINAL_EXTERNAL=YES before sealed external.",
        "timestamp": utc_now(),
    }
    write_json(ART / "EXTERNAL_HOLD.json", hold)
    write_json(JOINT_OUT / "EXTERNAL_HOLD.json", hold)
    (ART / "AWAITING_GO_TRUE_FINAL_EXTERNAL.flag").write_text("AWAITING\n")
    (JOINT_OUT / "AWAITING_GO_TRUE_FINAL_EXTERNAL.flag").write_text("AWAITING\n")

    lines = [
        "# FINAL_TRUE_FINAL_HARDNEG_MANIFEST",
        "",
        f"- path: `{man_path}`",
        f"- sha256: `{man_hash}`",
        f"- mean NDCG@20 = **{summary['mean_NDCG@20']:.6f} ± {summary['std_NDCG@20']:.6f}**",
        f"- Δ vs 0.2719 = {summary['delta_mean_vs_0.2719']:+.6f}",
        f"- Δ vs ~0.276 = {summary['delta_mean_vs_~0.276']:+.6f}",
        "- EXTERNAL_SEEN = **NO**",
        "- requires: `GO_TRUE_FINAL_EXTERNAL=YES`",
        "",
        "| seed | epoch | screen3k | fullcat | Δ0.2719 |",
        "|-----:|------:|---------:|--------:|--------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['selected_epoch']} | {r['screen3k_NDCG@20']:.6f} | "
            f"{r['NDCG@20']:.6f} | {r['delta_vs_0.2719']:+.6f} |"
        )
    (REP / "FINAL_TRUE_FINAL_HARDNEG_MANIFEST.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"summary": summary, "manifest_path": str(man_path), "manifest_sha256": man_hash}, indent=2))
    print("[tfhn] STOP — EXTERNAL_SEEN=NO — await GO_TRUE_FINAL_EXTERNAL=YES")


if __name__ == "__main__":
    main()
