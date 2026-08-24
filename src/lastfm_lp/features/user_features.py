"""User-level features (placeholder for later stages)."""

from __future__ import annotations


def user_history_length(user_history: dict[int, set[int]], user_id: int) -> int:
    return len(user_history.get(user_id, ()))
