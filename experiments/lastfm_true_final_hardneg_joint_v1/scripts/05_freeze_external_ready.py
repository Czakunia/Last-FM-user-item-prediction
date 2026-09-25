#!/usr/bin/env python3
"""Compatibility shim: redirect to 04_summarize (FINAL_TRUE_FINAL_HARDNEG_MANIFEST)."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "04_summarize.py"), run_name="__main__")
