"""
Learning Ledger — append-only JSONL file.

Attaches training outcomes (loss, perplexity, gradient norm) back to the
data that caused them. This two-way linkage is the mechanism that lets V5
teach V6 which shards were useful, which were redundant, and which caused
instability.

Per-token perplexity = exp(cross_entropy_loss_per_token).
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from core.config import LEARNING_JSONL

logger = logging.getLogger(__name__)


@dataclass
class LearningEvent:
    step:              int
    batch_id:          str
    loss:              float
    perplexity:        float         # exp(loss)
    gradient_norm:     float
    lane:              str
    shard_ids:         List[str]
    tokens_consumed:   int
    useful_tokens:     int           # loss-bearing tokens
    model_phase:       str           # seed / general / specialisation / anneal
    per_token_loss:    List[float]   # cross-entropy per token position (token-level trace)
    branch_id:         str = ""
    timestamp:         float = 0.0

    @classmethod
    def from_loss(
        cls,
        step: int,
        batch_id: str,
        loss: float,
        gradient_norm: float,
        lane: str,
        shard_ids: List[str],
        tokens_consumed: int,
        useful_tokens: int,
        model_phase: str,
        per_token_loss: List[float],
        branch_id: str = "",
    ) -> "LearningEvent":
        return cls(
            step            = step,
            batch_id        = batch_id,
            loss            = round(loss, 6),
            perplexity      = round(math.exp(min(loss, 20.0)), 4),
            gradient_norm   = round(gradient_norm, 6),
            lane            = lane,
            shard_ids       = shard_ids,
            tokens_consumed = tokens_consumed,
            useful_tokens   = useful_tokens,
            model_phase     = model_phase,
            per_token_loss  = [round(v, 6) for v in per_token_loss],
            branch_id       = branch_id,
            timestamp       = time.time(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class LearningLedger:
    """
    Append-only JSONL-based learning ledger.
    One line per training step, written immediately after each train_step().
    """

    def __init__(self, path: Path = LEARNING_JSONL) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate to start fresh on each run
        path.write_text("")
        self._count = 0
        logger.debug(f"LearningLedger opened at {path}")

    def append(self, event: LearningEvent) -> None:
        """Append one learning event as a JSON line."""
        if event.timestamp == 0.0:
            event.timestamp = time.time()
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
        self._count += 1

    def read_all(self) -> List[LearningEvent]:
        """Read all events from disk (for audit and reporting)."""
        events = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    events.append(LearningEvent(**data))
        return events

    @property
    def count(self) -> int:
        return self._count
