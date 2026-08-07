"""Tests: packing correctness, mask correctness, position IDs."""
import unittest
from core.packing import (
    PackingPolicy, TrainingMode, SequenceRecord,
    pack, packing_utilization, _hash_batch,
)
from core.tokenizer import PAD_ID, EOS_ID


def _seq(text: str, doc_id: str = "d1", shard_id: str = "s1",
         mode: TrainingMode = TrainingMode.PRETRAIN) -> SequenceRecord:
    return SequenceRecord(
        doc_id=doc_id,
        shard_id=shard_id,
        token_ids=[ord(c) % 95 + 5 for c in text],  # arbitrary token ids
        mode=mode,
    )


class TestPacking(unittest.TestCase):

    SEQ_LEN = 32

    def test_all_policies_return_batches(self):
        seqs = [_seq("Hello world", f"d{i}") for i in range(4)]
        for policy in PackingPolicy:
            batches = pack(seqs, policy, self.SEQ_LEN)
            self.assertGreater(len(batches), 0, f"No batches for policy {policy}")

    def test_all_batches_have_correct_seq_len(self):
        seqs = [_seq("Short text.", f"d{i}") for i in range(6)]
        for policy in PackingPolicy:
            batches = pack(seqs, policy, self.SEQ_LEN)
            for b in batches:
                self.assertEqual(len(b.token_ids), self.SEQ_LEN)
                self.assertEqual(len(b.loss_mask), self.SEQ_LEN)
                self.assertEqual(len(b.attention_mask), self.SEQ_LEN)
                self.assertEqual(len(b.position_ids), self.SEQ_LEN)

    def test_pretrain_loss_mask_has_no_loss_on_pad(self):
        seqs = [_seq("AB", "d1")]
        batches = pack(seqs, PackingPolicy.PAD_ONLY, self.SEQ_LEN)
        b = batches[0]
        for i, (tid, mask) in enumerate(zip(b.token_ids, b.loss_mask)):
            if tid == PAD_ID:
                self.assertEqual(mask, 0, f"PAD at position {i} should have loss_mask=0")

    def test_attention_mask_zero_on_pad(self):
        seqs = [_seq("Hi", "d1")]
        batches = pack(seqs, PackingPolicy.PAD_ONLY, self.SEQ_LEN)
        b = batches[0]
        for tid, amask in zip(b.token_ids, b.attention_mask):
            if tid == PAD_ID:
                self.assertEqual(amask, 0)
            else:
                self.assertEqual(amask, 1)

    def test_position_ids_start_at_zero(self):
        seqs = [_seq("Test", "d1")]
        batches = pack(seqs, PackingPolicy.PAD_ONLY, self.SEQ_LEN)
        self.assertEqual(batches[0].position_ids[0], 0)

    def test_batch_hash_is_stable(self):
        seqs = [_seq("Deterministic", "d1")]
        b1 = pack(seqs, PackingPolicy.PAD_ONLY, self.SEQ_LEN, "b")[0]
        b2 = pack(seqs, PackingPolicy.PAD_ONLY, self.SEQ_LEN, "b")[0]
        self.assertEqual(b1.batch_hash, b2.batch_hash)

    def test_utilization_in_range(self):
        seqs = [_seq("Some text here.", f"d{i}") for i in range(3)]
        batches = pack(seqs, PackingPolicy.GREEDY, self.SEQ_LEN)
        u = packing_utilization(batches)
        self.assertGreaterEqual(u, 0.0)
        self.assertLessEqual(u, 1.0)

    def test_best_fit_utilization_ge_pad_only(self):
        seqs = [_seq("Short.", f"d{i}") for i in range(8)]
        pad_batches  = pack(seqs, PackingPolicy.PAD_ONLY,  self.SEQ_LEN)
        best_batches = pack(seqs, PackingPolicy.BEST_FIT,  self.SEQ_LEN)
        u_pad  = packing_utilization(pad_batches)
        u_best = packing_utilization(best_batches)
        self.assertGreaterEqual(u_best, u_pad)


if __name__ == "__main__":
    unittest.main()
