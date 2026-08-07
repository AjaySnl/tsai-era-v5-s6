"""
Checkpoint management — save and load complete training state.

A checkpoint without a ledger offset is incomplete.
Model state + optimizer state + RNG state + ledger offset must travel together.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from core.config import CHECKPOINTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    step:             int
    branch_id:        str
    ledger_offset:    int
    batch_id:         str
    tokens_consumed:  int
    curriculum_stage: str
    model_state:      Dict[str, List]   # state_dict with tensors as lists
    optimizer_state:  Dict              # Adam state (scalars + lists)
    torch_rng_state:  List[int]         # torch RNG bytes as list of ints
    numpy_rng_state:  dict              # numpy RandomState serialization
    mixer_lane_counts: Dict[str, int]   # MixtureScheduler._lane_counts (D10 fix)
    checkpoint_hash:  str = ""          # sha256 of this checkpoint's content


def save_checkpoint(
    ckpt: Checkpoint,
    output_dir: Path = CHECKPOINTS_DIR,
) -> Path:
    """Serialize checkpoint to JSON and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"step_{ckpt.step:05d}.json"

    # Compute hash before saving
    content = _serializable(ckpt)
    content.pop("checkpoint_hash", None)
    raw = json.dumps(content, sort_keys=True)
    ckpt.checkpoint_hash = hashlib.sha256(raw.encode()).hexdigest()

    final = _serializable(ckpt)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    logger.info(f"[PASS] checkpoint_saved: step={ckpt.step}, hash={ckpt.checkpoint_hash[:8]}, path={path.name}")
    return path


def load_checkpoint(path: Path) -> Checkpoint:
    """Load checkpoint from JSON and verify its hash."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    stored_hash = data.pop("checkpoint_hash", "")
    raw = json.dumps(data, sort_keys=True)
    computed_hash = hashlib.sha256(raw.encode()).hexdigest()

    if stored_hash and stored_hash != computed_hash:
        raise ValueError(
            f"Checkpoint integrity check failed for {path.name}. "
            f"Expected {stored_hash[:8]}, got {computed_hash[:8]}."
        )

    data["checkpoint_hash"] = stored_hash
    return Checkpoint(**data)


def restore_rng_state(ckpt: Checkpoint) -> None:
    """Restore PyTorch and NumPy RNG states from checkpoint."""
    torch_bytes = bytes(ckpt.torch_rng_state)
    torch.random.set_rng_state(torch.tensor(list(torch_bytes), dtype=torch.uint8))
    np.random.set_state(_numpy_rng_from_dict(ckpt.numpy_rng_state))


def capture_rng_state() -> tuple[List[int], dict]:
    """Capture current PyTorch and NumPy RNG states."""
    torch_state = list(torch.random.get_rng_state().numpy().tolist())
    np_state = _numpy_rng_to_dict(np.random.get_state())
    return torch_state, np_state


# ── Serialization helpers ─────────────────────────────────────────────────────

def _serializable(ckpt: Checkpoint) -> dict:
    """Convert a Checkpoint to a JSON-serializable dict."""
    d = asdict(ckpt)
    return d


def _numpy_rng_to_dict(state: tuple) -> dict:
    """Convert numpy RandomState tuple to a JSON-serializable dict."""
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "name": name,
        "keys": keys.tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _numpy_rng_from_dict(d: dict) -> tuple:
    """Reconstruct numpy RandomState tuple from dict."""
    import numpy as np
    return (
        d["name"],
        np.array(d["keys"], dtype=np.uint32),
        d["pos"],
        d["has_gauss"],
        d["cached_gaussian"],
    )
