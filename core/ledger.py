"""
Training Consumption Ledger — append-only SQLite database.

Records every batch consumed during training. Append-only is enforced
at the database level via BEFORE UPDATE and BEFORE DELETE triggers that
raise an error, making it impossible to silently modify history.

This ledger is the run's memory — it enables crash recovery, replay,
and audit of any training interval.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from core.config import CONSUMPTION_DB

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS consumption_events (
    ledger_offset    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT    NOT NULL,
    branch_id        TEXT    NOT NULL,
    global_step      INTEGER NOT NULL,
    checkpoint_id    TEXT,
    rank             INTEGER NOT NULL DEFAULT 0,
    microbatch_id    TEXT    NOT NULL,
    packed_sample_ids TEXT   NOT NULL,   -- JSON array
    shard_ids        TEXT    NOT NULL,   -- JSON array
    token_span_start INTEGER NOT NULL,
    token_span_end   INTEGER NOT NULL,
    loss_mask_hash   TEXT    NOT NULL,
    mixture_lane     TEXT    NOT NULL,
    curriculum_stage TEXT    NOT NULL,
    tokenizer_version TEXT   NOT NULL,
    batch_hash       TEXT    NOT NULL,
    tokens_consumed  INTEGER NOT NULL,
    timestamp        REAL    NOT NULL
);
"""

_CREATE_IMMUTABLE_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS prevent_update
BEFORE UPDATE ON consumption_events
BEGIN
    SELECT RAISE(FAIL, 'consumption_events is append-only — UPDATE not permitted.');
END;

CREATE TRIGGER IF NOT EXISTS prevent_delete
BEFORE DELETE ON consumption_events
BEGIN
    SELECT RAISE(FAIL, 'consumption_events is append-only — DELETE not permitted.');
END;
"""


@dataclass
class ConsumptionEvent:
    run_id:           str
    branch_id:        str
    global_step:      int
    microbatch_id:    str
    packed_sample_ids: List[str]
    shard_ids:        List[str]
    token_span_start: int
    token_span_end:   int
    loss_mask_hash:   str
    mixture_lane:     str
    curriculum_stage: str
    tokenizer_version: str
    batch_hash:       str
    tokens_consumed:  int
    checkpoint_id:    str = ""
    rank:             int = 0
    ledger_offset:    int = -1          # set after INSERT
    timestamp:        float = 0.0


class ConsumptionLedger:
    """
    Append-only SQLite-backed consumption ledger.
    Each call to append() inserts one row; updates and deletes are blocked.
    """

    def __init__(self, db_path: Path = CONSUMPTION_DB) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove stale DB from previous runs to ensure fresh start
        if db_path.exists():
            db_path.unlink()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_CREATE_TABLE + _CREATE_IMMUTABLE_TRIGGERS)
        self._conn.commit()
        logger.debug(f"ConsumptionLedger opened at {db_path}")

    def append(self, event: ConsumptionEvent) -> int:
        """Insert one event and return its ledger_offset (row id)."""
        if event.timestamp == 0.0:
            event.timestamp = time.time()
        cur = self._conn.execute(
            """
            INSERT INTO consumption_events (
                run_id, branch_id, global_step, checkpoint_id, rank,
                microbatch_id, packed_sample_ids, shard_ids,
                token_span_start, token_span_end,
                loss_mask_hash, mixture_lane, curriculum_stage,
                tokenizer_version, batch_hash, tokens_consumed, timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event.run_id, event.branch_id, event.global_step,
                event.checkpoint_id, event.rank,
                event.microbatch_id,
                json.dumps(event.packed_sample_ids),
                json.dumps(event.shard_ids),
                event.token_span_start, event.token_span_end,
                event.loss_mask_hash, event.mixture_lane,
                event.curriculum_stage, event.tokenizer_version,
                event.batch_hash, event.tokens_consumed, event.timestamp,
            ),
        )
        self._conn.commit()
        offset = cur.lastrowid
        event.ledger_offset = offset
        return offset

    def get_offset(self) -> int:
        """Return the current maximum ledger offset (0 if empty)."""
        row = self._conn.execute(
            "SELECT MAX(ledger_offset) FROM consumption_events"
        ).fetchone()
        return row[0] or 0

    def get_event_at_step(self, global_step: int, branch_id: str | None = None) -> Optional[ConsumptionEvent]:
        """Retrieve the consumption event for a specific global step."""
        if branch_id:
            row = self._conn.execute(
                "SELECT * FROM consumption_events WHERE global_step=? AND branch_id=? LIMIT 1",
                (global_step, branch_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM consumption_events WHERE global_step=? LIMIT 1",
                (global_step,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def get_events_in_range(
        self, step_start: int, step_end: int, branch_id: str | None = None
    ) -> List[ConsumptionEvent]:
        """Return all events between step_start and step_end (inclusive)."""
        if branch_id:
            rows = self._conn.execute(
                "SELECT * FROM consumption_events WHERE global_step BETWEEN ? AND ? AND branch_id=? ORDER BY global_step",
                (step_start, step_end, branch_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM consumption_events WHERE global_step BETWEEN ? AND ? ORDER BY global_step",
                (step_start, step_end),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM consumption_events").fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_event(row: tuple) -> ConsumptionEvent:
        (offset, run_id, branch_id, step, ckpt_id, rank,
         mb_id, sample_ids_json, shard_ids_json,
         span_start, span_end, lm_hash, lane, stage,
         tok_ver, batch_hash, tokens, ts) = row
        return ConsumptionEvent(
            run_id            = run_id,
            branch_id         = branch_id,
            global_step       = step,
            microbatch_id     = mb_id,
            packed_sample_ids = json.loads(sample_ids_json),
            shard_ids         = json.loads(shard_ids_json),
            token_span_start  = span_start,
            token_span_end    = span_end,
            loss_mask_hash    = lm_hash,
            mixture_lane      = lane,
            curriculum_stage  = stage,
            tokenizer_version = tok_ver,
            batch_hash        = batch_hash,
            tokens_consumed   = tokens,
            checkpoint_id     = ckpt_id or "",
            rank              = rank,
            ledger_offset     = offset,
            timestamp         = ts,
        )
