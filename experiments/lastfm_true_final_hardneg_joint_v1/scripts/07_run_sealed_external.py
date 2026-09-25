#!/usr/bin/env python3
"""Sealed external for TRUE FINAL hardneg — only after DEV freeze + GO.

Requires:
  GO_TRUE_FINAL_EXTERNAL=YES
  CONFIRM_UNSEAL_LASTFM_TEST=YES
  artifacts/PROTOCOL_COMPLETE.flag
  artifacts/FINAL_TRUE_FINAL_HARDNEG_MANIFEST.json

REPLACE_TRUE_FINAL stays DEFERRED unless user later sets it YES.

Writes under LASTFM_TRUE_FINAL/JOINT_HARDNEG_R3_FROM_SCRATCH_V1/07_EXTERNAL/
without touching JOINT_TRAINING_V1 / easy external artifacts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ART = EXP / "artifacts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "src"))


def main() -> None:
    if os.environ.get("GO_TRUE_FINAL_EXTERNAL") != "YES":
        raise RuntimeError("Set GO_TRUE_FINAL_EXTERNAL=YES")
    if os.environ.get("CONFIRM_UNSEAL_LASTFM_TEST") != "YES":
        raise RuntimeError("Set CONFIRM_UNSEAL_LASTFM_TEST=YES (sealed gate)")
    if not (ART / "PROTOCOL_COMPLETE.flag").exists():
        raise RuntimeError("DEV protocol not complete")
    man_path = ART / "FINAL_TRUE_FINAL_HARDNEG_MANIFEST.json"
    if not man_path.exists():
        raise RuntimeError(f"missing {man_path}")
    go_path = ART / "GO_TRUE_FINAL_EXTERNAL.json"
    go = json.loads(go_path.read_text())
    if go.get("REPLACE_TRUE_FINAL") == "YES":
        raise RuntimeError("REPLACE_TRUE_FINAL=YES not authorized in this path; keep DEFERRED")
    if os.environ.get("TFHN_REPLACE_TRUE_FINAL", "DEFERRED") == "YES":
        raise RuntimeError("TFHN_REPLACE_TRUE_FINAL=YES refused; keep DEFERRED")

    from tfhn.sealed_external import run_hardneg_sealed_external  # noqa: E402

    manifest = json.loads(man_path.read_text())
    payload = run_hardneg_sealed_external(manifest)

    go["STATUS"] = "EXTERNAL_DONE"
    go["EXTERNAL_SEEN"] = True
    go["EXTERNAL_MEAN_NDCG@20"] = payload["block_a"]["true_final_hardneg_mean"]["NDCG@20"]
    go["delta_vs_easy_ext_mean"] = payload["delta_vs_easy_ext_mean"]
    go["REPLACE_TRUE_FINAL"] = "DEFERRED"
    go_path.write_text(json.dumps(go, indent=2) + "\n")

    a = payload["block_a"]["true_final_hardneg_mean"]
    print(
        f"HARDNEG_EXTERNAL = OK mean NDCG@20={a['NDCG@20']:.6f} "
        f"Δ_vs_easy={payload['delta_vs_easy_ext_mean']:+.6f} "
        f"REPLACE_TRUE_FINAL=DEFERRED",
        flush=True,
    )


if __name__ == "__main__":
    main()
