"""
Mixture schedule and curriculum stage management.

Converts the human-readable curriculum design from Session 5 into
per-step lane quotas that the training loop can execute.
Also tracks planned vs actual token consumption per lane.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.config import CURRICULUM, CurriculumStageDef, LANES, PROTECTED_LANES, SEQ_LEN

logger = logging.getLogger(__name__)


# ── Mixture accounting ────────────────────────────────────────────────────────

@dataclass
class MixtureAccounting:
    """Tracks planned vs actual token consumption per capability lane."""
    planned: Dict[str, int] = field(default_factory=lambda: {lane: 0 for lane in LANES})
    actual:  Dict[str, int] = field(default_factory=lambda: {lane: 0 for lane in LANES})

    def record_planned(self, lane: str, tokens: int) -> None:
        self.planned[lane] = self.planned.get(lane, 0) + tokens

    def record_actual(self, lane: str, tokens: int) -> None:
        self.actual[lane] = self.actual.get(lane, 0) + tokens

    def compliance_report(self) -> Dict[str, dict]:
        report = {}
        for lane in LANES:
            planned = self.planned.get(lane, 0)
            actual  = self.actual.get(lane, 0)
            pct = round(100 * actual / max(planned, 1), 1)
            report[lane] = {
                "planned_tokens": planned,
                "actual_tokens": actual,
                "compliance_pct": pct,
                "status": "OK" if pct >= 80 else "UNDER",
            }
        return report


# ── Mixture scheduler ─────────────────────────────────────────────────────────

class MixtureScheduler:
    """
    Given the current token count, returns the active curriculum stage
    and the lane that should be selected for the next batch.

    Protected floors are always enforced: if a protected lane (Indic, Agentic)
    has fallen below its floor, it is selected regardless of other logic.
    """

    def __init__(self) -> None:
        self._stages: List[CurriculumStageDef] = CURRICULUM
        self._accounting = MixtureAccounting()
        self._lane_counts: Dict[str, int] = {lane: 0 for lane in LANES}
        self._total_tokens = 0

    def get_stage(self, tokens_consumed: int) -> CurriculumStageDef:
        """Return the active CurriculumStageDef for the current token count."""
        for stage in self._stages:
            if stage.token_start <= tokens_consumed < stage.token_end:
                return stage
        return self._stages[-1]

    def select_lane(self, tokens_consumed: int) -> str:
        """
        Select the next capability lane to sample from.
        Protected floors take priority; otherwise proportional selection.
        Also records planned tokens for compliance accounting.
        """
        stage = self.get_stage(tokens_consumed)

        # Record planned allocation for this step based on stage weights
        for lane, weight in stage.mixture_weights.items():
            planned = int(weight * SEQ_LEN)   # proportional planned tokens per step
            self._accounting.record_planned(lane, planned)

        # Check protected floors first
        total = max(sum(self._lane_counts.values()), 1)
        for lane in PROTECTED_LANES:
            floor = stage.protected_floors.get(lane, 0.0)
            current_share = self._lane_counts.get(lane, 0) / total
            if current_share < floor * 0.9:  # 10% tolerance
                self._lane_counts[lane] = self._lane_counts.get(lane, 0) + 1
                return lane

        # Proportional selection based on mixture weights
        weights = stage.mixture_weights
        active_lanes = [l for l in LANES if weights.get(l, 0) > 0]
        if not active_lanes:
            active_lanes = ["web"]

        # Find the lane most under-served relative to its target share
        best_lane = active_lanes[0]
        best_deficit = -1.0
        for lane in active_lanes:
            target = weights.get(lane, 0.0)
            current = self._lane_counts.get(lane, 0) / total
            deficit = target - current
            if deficit > best_deficit:
                best_deficit = deficit
                best_lane = lane

        self._lane_counts[best_lane] = self._lane_counts.get(best_lane, 0) + 1
        return best_lane

    def record_consumption(self, lane: str, tokens: int) -> None:
        """Record actual token consumption for accounting."""
        self._accounting.record_actual(lane, tokens)
        self._total_tokens += tokens

    def compliance_report(self) -> Dict[str, dict]:
        return self._accounting.compliance_report()

    def get_lane_counts(self) -> Dict[str, int]:
        """Return a copy of lane counts for checkpoint serialisation."""
        return dict(self._lane_counts)

    def restore_lane_counts(self, counts: Dict[str, int]) -> None:
        """Restore lane counts from a checkpoint (D10 fix)."""
        self._lane_counts = dict(counts)

    @property
    def accounting(self) -> MixtureAccounting:
        return self._accounting
