"""
Audit subsystem — verifies all invariants and generates the evidence bundle.

Checks:
  1. Tokenizer hash stability
  2. Manifest integrity (content hash, tokenizer hash)
  3. Shard immutability
  4. Eval firewall (no eval shard in ledger)
  5. Consumption ledger append-only (no duplicates)
  6. Checkpoint integrity (hash match)
  7. Batch hash reproducibility
  8. Mixture compliance (planned vs actual)
  9. Replay hash match

Generates:
  submission_artifacts/evidence.json
  submission_artifacts/evidence.md
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from core.config import EVIDENCE_JSON, EVIDENCE_MD, MANIFESTS_DIR, CHECKPOINTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class RequirementResult:
    name:        str
    result:      str    # PASS / FAIL
    evidence:    str    # path or description of supporting artifact
    detail:      str = ""


@dataclass
class AuditReport:
    requirements: List[RequirementResult]
    overall:      str   # PASS if all pass, FAIL otherwise

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "requirements": [asdict(r) for r in self.requirements],
        }


class Auditor:
    """Collects audit results and generates the evidence bundle."""

    def __init__(self) -> None:
        self._results: List[RequirementResult] = []

    def record(
        self,
        name: str,
        passed: bool,
        evidence: str,
        detail: str = "",
    ) -> None:
        result = "PASS" if passed else "FAIL"
        self._results.append(RequirementResult(name, result, evidence, detail))
        icon = "[OK]" if passed else "[!]"
        logger.info(f"[AUDIT] {icon} {name}: {result} -- {evidence}")

    def build_report(self) -> AuditReport:
        overall = "PASS" if all(r.result == "PASS" for r in self._results) else "FAIL"
        return AuditReport(requirements=list(self._results), overall=overall)

    def save_evidence(self) -> None:
        """Write evidence.json and evidence.md to submission_artifacts/."""
        report = self.build_report()

        # JSON
        EVIDENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(EVIDENCE_JSON, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Markdown table
        lines = [
            "# Evidence Report\n",
            f"**Overall result: {report.overall}**\n",
            "",
            "| Requirement | Result | Evidence |",
            "|---|---|---|",
        ]
        for r in report.requirements:
            badge = "PASS" if r.result == "PASS" else "FAIL"
            lines.append(f"| {r.name} | {badge} | {r.evidence} |")

        if any(r.detail for r in report.requirements):
            lines += ["", "## Details", ""]
            for r in report.requirements:
                if r.detail:
                    lines.append(f"- **{r.name}**: {r.detail}")

        with open(EVIDENCE_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Evidence saved → {EVIDENCE_JSON.name}, {EVIDENCE_MD.name}")


# ── Standalone verification functions ────────────────────────────────────────

def verify_tokenizer_hash(tokenizer_hash: str, expected_hash: str) -> bool:
    return tokenizer_hash == expected_hash


def verify_manifest_integrity(shard, manifest) -> bool:
    """Re-check content hash and tokenizer hash match between shard and manifest."""
    return (
        shard.content_hash  == manifest.content_hash  and
        shard.tokenizer_hash == manifest.tokenizer_hash and
        shard.token_count    == manifest.token_count
    )


def verify_no_eval_in_ledger(ledger, blocked_shard_ids: set) -> tuple[bool, List[int]]:
    """Scan ledger for any eval shard IDs. Returns (clean, violating_steps)."""
    events = ledger.get_events_in_range(0, 999_999)
    violations = []
    for event in events:
        for sid in event.shard_ids:
            if sid in blocked_shard_ids:
                violations.append(event.global_step)
    return len(violations) == 0, violations


def verify_checkpoint_hash(path: Path) -> bool:
    """Load a checkpoint and verify its stored hash matches recomputed hash."""
    import json, hashlib
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    stored = data.pop("checkpoint_hash", "")
    raw = json.dumps(data, sort_keys=True)
    computed = hashlib.sha256(raw.encode()).hexdigest()
    return stored == computed


def verify_replay_hashes(
    original_events,
    replayed_events,
) -> tuple[bool, List[str]]:
    """
    Compare original and replayed consumption events.
    Returns (all_match, list_of_mismatches).
    """
    mismatches = []
    for orig, repl in zip(original_events, replayed_events):
        if orig.batch_hash != repl.batch_hash:
            mismatches.append(
                f"step={orig.global_step}: orig={orig.batch_hash[:8]} vs repl={repl.batch_hash[:8]}"
            )
    return len(mismatches) == 0, mismatches


def count_manifests(manifests_dir: Path = MANIFESTS_DIR) -> int:
    return len(list(manifests_dir.glob("*.json")))


def count_checkpoints(checkpoints_dir: Path = CHECKPOINTS_DIR) -> int:
    return len(list(checkpoints_dir.glob("step_*.json")))
