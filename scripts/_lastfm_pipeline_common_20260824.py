"""Shared helpers for numbered Last-FM* pipeline entry scripts (2026-08-24)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
SEEDS = (101, 202, 303)


def venv_python() -> Path:
    if not PY.exists():
        raise SystemExit(f"Missing venv python: {PY}\nRun: python3 -m venv .venv && pip install -r requirements.txt")
    return PY


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    for k in list(env):
        if k.startswith("LASTFM_") or k.startswith("PYTORCH_MPS_"):
            del env[k]
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": str(ROOT / ".mplconfig"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "LASTFM_SKIP_C1_WAIT": "1",
        }
    )
    return env


def run_script(script: str, *, extra_env: dict[str, str] | None = None) -> None:
    py = venv_python()
    cmd = [str(py), "-u", str(ROOT / "scripts" / script)]
    env = base_env()
    if extra_env:
        env.update(extra_env)
    print(f"[pipeline] {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    if rc != 0:
        raise SystemExit(f"[pipeline] {script} failed with exit code {rc}")


def run_module_prepare() -> None:
    py = venv_python()
    code = """
from src.lastfm_lp.config import load_protocol_config
from src.lastfm_lp.pipeline.prepare import prepare_all
prepare_all(load_protocol_config("configs/lastfm_star_race_clean_3.yaml"), verify=False)
"""
    env = base_env()
    print("[pipeline] prepare anti-leakage splits + pair tables", flush=True)
    rc = subprocess.run([str(py), "-c", code], cwd=str(ROOT), env=env).returncode
    if rc != 0:
        raise SystemExit(f"[pipeline] prepare failed with exit code {rc}")
