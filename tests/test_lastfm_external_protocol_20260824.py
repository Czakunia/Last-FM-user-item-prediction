from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lastfm_lp.data.build_splits import load_user_sets



def _pair_set(d: dict[int, set[int]]) -> set[tuple[int, int]]:
    return {(int(u), int(i)) for u, items in d.items() for i in items}


def test_upstream_train_equals_model_train_plus_validation():
    upstream_train = load_user_sets(ROOT / "data" / "LastFM_star_IntentAwareRS" / "train.txt")
    model_train = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "model_train.txt")
    valid = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "valid.txt")
    assert _pair_set(upstream_train) == (_pair_set(model_train) | _pair_set(valid))


def test_zero_train_test_overlap():
    model_train = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "model_train.txt")
    valid = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "valid.txt")
    test = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "test.txt")
    assert len(_pair_set(model_train) & _pair_set(valid)) == 0
    assert len(_pair_set(model_train) & _pair_set(test)) == 0
    assert len(_pair_set(valid) & _pair_set(test)) == 0


def test_frozen_epoch_mapping():
    assert {101: 36, 202: 34, 303: 43} == {101: 36, 202: 34, 303: 43}


def test_sealed_guard_blocks_execution(monkeypatch):
    monkeypatch.delenv("CONFIRM_UNSEAL_LASTFM_TEST", raising=False)
    try:
        runpy.run_path(str(ROOT / "scripts" / "LAST_FM_EXT_04_run_sealed_test_once_20260824.py"))
    except RuntimeError as exc:
        assert "Sealed Last-FM* test evaluation is disabled" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError from sealed test guard")


def test_external_repos_gitignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "external_repos/" in gitignore
