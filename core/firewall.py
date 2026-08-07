"""
Evaluation and validation firewall.

Eval shards carry a never_train=True flag and are registered here at startup.
Any attempt to feed an eval shard into training raises FirewallViolationError.
The firewall also logs a [PASS] event confirming the block worked.
"""
from __future__ import annotations

import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)


class FirewallViolationError(RuntimeError):
    """Raised when an eval or validation shard is submitted to the training path."""


class EvalFirewall:
    """
    Registry of shard IDs and content hashes that must NEVER enter training.
    Thread-safe for single-process use (no locking needed at our scale).
    """

    def __init__(self) -> None:
        self._blocked_ids: Set[str] = set()
        self._blocked_hashes: Set[str] = set()
        self._blocked_metadata: Dict[str, dict] = {}

    def register(self, shard_id: str, content_hash: str, reason: str = "eval") -> None:
        """
        Mark a shard as blocked from training.
        Must be called before any training loop begins.
        """
        self._blocked_ids.add(shard_id)
        self._blocked_hashes.add(content_hash)
        self._blocked_metadata[shard_id] = {
            "shard_id": shard_id,
            "content_hash": content_hash,
            "reason": reason,
        }
        logger.info(f"[FIREWALL] Registered blocked shard: {shard_id} ({reason})")

    def check(self, shard_id: str, content_hash: str | None = None) -> None:
        """
        Verify that shard_id is allowed into training.
        Raises FirewallViolationError immediately if blocked.
        Also checks content_hash if provided — catches renamed eval shards.
        """
        if shard_id in self._blocked_ids:
            raise FirewallViolationError(
                f"FIREWALL VIOLATION: shard '{shard_id}' is blocked (eval/validation). "
                f"Reason: {self._blocked_metadata[shard_id]['reason']}"
            )
        if content_hash and content_hash in self._blocked_hashes:
            raise FirewallViolationError(
                f"FIREWALL VIOLATION: shard with hash '{content_hash[:8]}' is blocked "
                f"(eval/validation — content hash match)."
            )

    def is_blocked(self, shard_id: str) -> bool:
        """Return True if the shard is blocked (without raising)."""
        return shard_id in self._blocked_ids

    def demonstrate_block(self, shard_id: str, content_hash: str) -> bool:
        """
        Try to submit a known eval shard and catch the violation.
        Used in run_demo.py to generate the [PASS] eval_shard_blocked event.
        Returns True if the block fired correctly.
        """
        try:
            self.check(shard_id, content_hash)
            return False  # Should not reach here
        except FirewallViolationError:
            logger.info(f"[PASS] eval_shard_blocked: '{shard_id}' correctly rejected by firewall.")
            return True

    @property
    def blocked_count(self) -> int:
        return len(self._blocked_ids)


# Module-level singleton
firewall = EvalFirewall()
