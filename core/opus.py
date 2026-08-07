"""
OPUS data selector — simulates proxy-based batch selection.

OPUS keeps a ghost copy of the current model and scores each candidate batch
by estimating how useful the update would be for the target benchmarks.
Here we simulate this with a deterministic hash-based scoring function
that produces stable, reproducible scores without actual model inference.

Every decision is recorded to an append-only JSONL audit trail.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.config import PROTECTED_LANES, OPUS_JSONL

logger = logging.getLogger(__name__)


class OpusStatus(str, Enum):
    ACCEPTED      = "ACCEPTED"
    REJECTED      = "REJECTED"
    DEFERRED      = "DEFERRED"
    FLOOR_OVERRIDE = "FLOOR_OVERRIDE"   # protected lane — always admitted


@dataclass
class OpusDecision:
    candidate_id:   str
    shard_id:       str
    lane:           str
    score:          float
    status:         str
    reason:         str
    step:           int
    proxy_version:  str = "proxy_v1"
    timestamp:      float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["timestamp"] == 0.0:
            d["timestamp"] = time.time()
        return d


class OpusSelector:
    """
    Deterministic OPUS-style data selector.

    Scoring: SHA-256 of (shard_id + str(step)) → normalized float in [0, 1].
    This is stable across runs for the same inputs (deterministic).

    Acceptance threshold: configurable (default 0.40 — keeps ~40% of candidates,
    matching V4's reported 40% keep rate).
    """

    def __init__(
        self,
        keep_fraction: float = 0.40,
        audit_path: Path = OPUS_JSONL,
        fresh: bool = False,
    ) -> None:
        self._keep_fraction = keep_fraction
        self._audit_path = audit_path
        self._decisions: List[OpusDecision] = []
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        # Only truncate when explicitly requested (e.g., start of a new run).
        # Tests pass a temp path and never set fresh=True, so they never
        # touch the production audit file.
        if fresh:
            self._audit_path.write_text("")

    def score(self, shard_id: str, step: int) -> float:
        """Deterministic proxy score for a candidate batch."""
        raw = hashlib.sha256(f"{shard_id}:{step}".encode()).digest()
        # Use first 4 bytes as unsigned int, normalize to [0, 1]
        value = int.from_bytes(raw[:4], "big") / 0xFFFFFFFF
        return round(value, 6)

    def evaluate(
        self,
        shard_id: str,
        lane: str,
        step: int,
        floor_active: bool = False,
    ) -> OpusDecision:
        """
        Evaluate one candidate batch.
        - Protected floors always produce FLOOR_OVERRIDE (always admitted).
        - Otherwise: ACCEPTED if score >= threshold, REJECTED below.
        - A DEFERRED decision means score is in a borderline range (0.35–0.40).
        """
        candidate_id = f"cand_{step:04d}_{shard_id[:8]}"
        s = self.score(shard_id, step)

        if floor_active or lane in PROTECTED_LANES:
            status = OpusStatus.FLOOR_OVERRIDE
            reason = f"Protected floor lane '{lane}' — bypass selector."
        elif s >= self._keep_fraction:
            status = OpusStatus.ACCEPTED
            reason = f"Score {s:.4f} >= threshold {self._keep_fraction}."
        elif s >= self._keep_fraction * 0.875:  # borderline range
            status = OpusStatus.DEFERRED
            reason = f"Score {s:.4f} in borderline range — deferred."
        else:
            status = OpusStatus.REJECTED
            reason = f"Score {s:.4f} < threshold {self._keep_fraction}."

        decision = OpusDecision(
            candidate_id  = candidate_id,
            shard_id      = shard_id,
            lane          = lane,
            score         = s,
            status        = status.value,
            reason        = reason,
            step          = step,
            timestamp     = time.time(),
        )
        self._record(decision)
        return decision

    def is_admitted(self, decision: OpusDecision) -> bool:
        """Return True if the decision admits the batch to training."""
        return decision.status in (
            OpusStatus.ACCEPTED.value,
            OpusStatus.FLOOR_OVERRIDE.value,
        )
        # Note: DEFERRED is distinct from REJECTED — deferred batches are
        # candidates for a future training phase (e.g., annealing reserve).
        # They are not admitted to the current step but are preserved in the
        # audit trail for later scheduling. REJECTED batches are low-value
        # and discarded entirely.

    def is_deferred(self, decision: OpusDecision) -> bool:
        """Return True if the batch was deferred (not rejected, not admitted)."""
        return decision.status == OpusStatus.DEFERRED.value

    def _record(self, decision: OpusDecision) -> None:
        """Append decision to in-memory list and JSONL audit file."""
        self._decisions.append(decision)
        with open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")

    @property
    def decisions(self) -> List[OpusDecision]:
        return list(self._decisions)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self._decisions:
            counts[d.status] = counts.get(d.status, 0) + 1
        return counts
