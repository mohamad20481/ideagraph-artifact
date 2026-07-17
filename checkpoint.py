"""
checkpoint.py - Checkpoint/resume system for long pipeline runs.

Saves pipeline state to disk after each stage so runs can be resumed
if interrupted. Also enables incremental progress reporting.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "checkpoints")


class PipelineCheckpoint:
    """Save and restore pipeline state between stages."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.dir = os.path.join(_CHECKPOINT_DIR, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self._state: Dict[str, Any] = {
            "run_id": run_id,
            "created_at": time.time(),
            "current_stage": 0,
            "completed_stages": [],
            "stage_data": {},
            "metadata": {},
        }
        # Try to load existing checkpoint
        self._load()

    def _path(self) -> str:
        return os.path.join(self.dir, "checkpoint.json")

    def _load(self) -> None:
        """Load checkpoint from disk if it exists."""
        path = self._path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # start fresh

    def save(self) -> None:
        """Persist current state to disk."""
        self._state["last_saved"] = time.time()
        with open(self._path(), "w") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False, default=str)

    @property
    def current_stage(self) -> int:
        return self._state.get("current_stage", 0)

    @current_stage.setter
    def current_stage(self, value: int) -> None:
        self._state["current_stage"] = value
        self.save()

    def is_stage_complete(self, stage: int) -> bool:
        """Check if a stage was already completed (for resume)."""
        return stage in self._state.get("completed_stages", [])

    def complete_stage(self, stage: int, data: Dict[str, Any]) -> None:
        """Mark a stage as complete and save its output data."""
        if stage not in self._state["completed_stages"]:
            self._state["completed_stages"].append(stage)
        self._state["stage_data"][str(stage)] = data
        self._state["current_stage"] = stage + 1
        self.save()

    def get_stage_data(self, stage: int) -> Optional[Dict[str, Any]]:
        """Get saved output from a completed stage."""
        return self._state["stage_data"].get(str(stage))

    def set_metadata(self, key: str, value: Any) -> None:
        """Store arbitrary metadata (topic, budget, etc.)."""
        self._state["metadata"][key] = value
        self.save()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._state["metadata"].get(key, default)

    def clear(self) -> None:
        """Remove checkpoint (run completed successfully)."""
        path = self._path()
        if os.path.exists(path):
            os.remove(path)

    def elapsed_since_create(self) -> float:
        """Seconds since checkpoint was created."""
        return time.time() - self._state.get("created_at", time.time())

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._state)


# mtime-keyed cache for list_checkpoints. The resume UI calls this on every
# Streamlit rerun, but the underlying directory rarely changes. We re-scan
# only when a checkpoint file's mtime moves; otherwise we replay the cached
# parsed result. Per-run-id sub-cache avoids re-parsing the same JSON when
# only a sibling checkpoint changed.
_LIST_CACHE: Dict[str, Dict[str, Any]] = {}  # {run_id: {"mtime": float, "row": dict}}
_LIST_CACHE_DIR_MTIME: float = -1.0


def list_checkpoints() -> list:
    """List all available checkpoints (for resume UI). Cached by mtime."""
    global _LIST_CACHE_DIR_MTIME
    if not os.path.exists(_CHECKPOINT_DIR):
        return []

    # Cheap fast-path: if the parent dir hasn't been touched (no add/remove),
    # walk the existing cache entries and verify each file's mtime to detect
    # in-place updates without re-parsing untouched ones.
    try:
        dir_mtime = os.path.getmtime(_CHECKPOINT_DIR)
    except OSError:
        dir_mtime = -1.0

    results: list = []
    seen: set = set()
    for run_id in os.listdir(_CHECKPOINT_DIR):
        cp_path = os.path.join(_CHECKPOINT_DIR, run_id, "checkpoint.json")
        try:
            mtime = os.path.getmtime(cp_path)
        except OSError:
            continue
        seen.add(run_id)
        cached = _LIST_CACHE.get(run_id)
        if cached is not None and cached["mtime"] == mtime:
            results.append(cached["row"])
            continue
        try:
            with open(cp_path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        row = {
            "run_id": run_id,
            "topic": state.get("metadata", {}).get("topic", ""),
            "current_stage": state.get("current_stage", 0),
            "completed_stages": len(state.get("completed_stages", [])),
            "created_at": state.get("created_at", 0),
            "last_saved": state.get("last_saved", 0),
        }
        _LIST_CACHE[run_id] = {"mtime": mtime, "row": row}
        results.append(row)

    # Drop cache entries for run_ids that disappeared from disk.
    if seen != set(_LIST_CACHE.keys()):
        for stale in list(_LIST_CACHE.keys() - seen):
            _LIST_CACHE.pop(stale, None)

    _LIST_CACHE_DIR_MTIME = dir_mtime
    results.sort(key=lambda x: x.get("last_saved", 0), reverse=True)
    return results


def delete_checkpoint(run_id: str) -> bool:
    """Delete a checkpoint directory."""
    import shutil
    path = os.path.join(_CHECKPOINT_DIR, run_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        # Invalidate the corresponding cache entry so list_checkpoints
        # doesn't return stale data after deletion.
        _LIST_CACHE.pop(run_id, None)
        return True
    return False
