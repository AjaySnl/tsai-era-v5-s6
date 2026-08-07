"""Tests: consumption ledger append-only and no duplicates."""
import tempfile
import unittest
from pathlib import Path
from core.ledger import ConsumptionLedger, ConsumptionEvent


def _event(step: int, branch: str = "main") -> ConsumptionEvent:
    return ConsumptionEvent(
        run_id="test_run",
        branch_id=branch,
        global_step=step,
        microbatch_id=f"mb_{step}",
        packed_sample_ids=[f"sample_{step}"],
        shard_ids=["shard_001"],
        token_span_start=step * 64,
        token_span_end=(step + 1) * 64,
        loss_mask_hash="deadbeef",
        mixture_lane="web",
        curriculum_stage="seed",
        tokenizer_version="v1_abc123",
        batch_hash=f"hash_{step:04x}",
        tokens_consumed=64,
    )


class TestLedger(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "test.db"
        self.ledger = ConsumptionLedger(db_path)

    def tearDown(self):
        self.ledger.close()
        self._tmp.cleanup()

    def test_append_increments_offset(self):
        self.ledger.append(_event(1))
        self.ledger.append(_event(2))
        self.assertEqual(self.ledger.get_offset(), 2)

    def test_count_matches_appended(self):
        for i in range(5):
            self.ledger.append(_event(i))
        self.assertEqual(self.ledger.count(), 5)

    def test_get_event_at_step(self):
        self.ledger.append(_event(7))
        ev = self.ledger.get_event_at_step(7)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.global_step, 7)
        self.assertEqual(ev.batch_hash, "hash_0007")

    def test_update_blocked(self):
        self.ledger.append(_event(1))
        import sqlite3
        with self.assertRaises((sqlite3.OperationalError, sqlite3.IntegrityError)):
            self.ledger._conn.execute(
                "UPDATE consumption_events SET mixture_lane='hacked' WHERE global_step=1"
            )

    def test_delete_blocked(self):
        self.ledger.append(_event(1))
        import sqlite3
        with self.assertRaises((sqlite3.OperationalError, sqlite3.IntegrityError)):
            self.ledger._conn.execute(
                "DELETE FROM consumption_events WHERE global_step=1"
            )

    def test_range_query(self):
        for i in range(10):
            self.ledger.append(_event(i))
        events = self.ledger.get_events_in_range(3, 6)
        self.assertEqual(len(events), 4)
        self.assertEqual([e.global_step for e in events], [3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
