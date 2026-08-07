"""Tests: replay produces identical batch hashes to original run."""
import unittest
from core.packing import PackingPolicy, SequenceRecord, pack, TrainingMode


def _make_seq(seed_text: str, shard_id: str = "s1") -> SequenceRecord:
    return SequenceRecord(
        doc_id=f"doc_{hash(seed_text) % 1000:03d}",
        shard_id=shard_id,
        token_ids=[ord(c) % 95 + 5 for c in seed_text],
        mode=TrainingMode.PRETRAIN,
    )


class TestReplay(unittest.TestCase):

    def test_same_inputs_produce_same_batch_hash(self):
        """
        Core replay invariant: given the same token sequence and packing policy,
        the batch hash must be identical on every run.
        """
        seqs = [_make_seq(f"Document number {i} with stable content.") for i in range(4)]
        seq_len = 64

        original = pack(seqs, PackingPolicy.GREEDY, seq_len, "batch")
        replayed = pack(seqs, PackingPolicy.GREEDY, seq_len, "batch")

        self.assertEqual(len(original), len(replayed))
        for orig, repl in zip(original, replayed):
            self.assertEqual(
                orig.batch_hash, repl.batch_hash,
                f"Batch hash mismatch for batch {orig.batch_id} vs {repl.batch_id}"
            )

    def test_different_inputs_produce_different_hash(self):
        seqs1 = [_make_seq("Version one of the document.")]
        seqs2 = [_make_seq("Version two of the document.")]
        b1 = pack(seqs1, PackingPolicy.PAD_ONLY, 32, "b")[0]
        b2 = pack(seqs2, PackingPolicy.PAD_ONLY, 32, "b")[0]
        self.assertNotEqual(b1.batch_hash, b2.batch_hash)

    def test_all_packing_policies_are_deterministic(self):
        seqs = [_make_seq(f"Text {i}") for i in range(5)]
        for policy in PackingPolicy:
            b1 = pack(seqs, policy, 32, "run1")
            b2 = pack(seqs, policy, 32, "run1")
            hashes1 = [b.batch_hash for b in b1]
            hashes2 = [b.batch_hash for b in b2]
            self.assertEqual(hashes1, hashes2, f"Policy {policy} not deterministic")


if __name__ == "__main__":
    unittest.main()
