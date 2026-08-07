"""Tests: eval firewall never admits eval shards to training."""
import unittest
from core.firewall import EvalFirewall, FirewallViolationError


class TestFirewall(unittest.TestCase):

    def setUp(self):
        self.fw = EvalFirewall()
        self.fw.register("eval_shard_001", "hash_abc123", reason="eval")
        self.fw.register("val_shard_001", "hash_def456", reason="validation")

    def test_eval_shard_blocked_by_id(self):
        with self.assertRaises(FirewallViolationError):
            self.fw.check("eval_shard_001")

    def test_eval_shard_blocked_by_hash(self):
        with self.assertRaises(FirewallViolationError):
            self.fw.check("unknown_id", content_hash="hash_abc123")

    def test_train_shard_allowed(self):
        # Should not raise
        self.fw.check("train_shard_001", "train_hash_xyz")

    def test_is_blocked_returns_true_for_eval(self):
        self.assertTrue(self.fw.is_blocked("eval_shard_001"))

    def test_is_blocked_returns_false_for_train(self):
        self.assertFalse(self.fw.is_blocked("train_shard_999"))

    def test_demonstrate_block_returns_true(self):
        result = self.fw.demonstrate_block("eval_shard_001", "hash_abc123")
        self.assertTrue(result)

    def test_blocked_count(self):
        self.assertEqual(self.fw.blocked_count, 2)


if __name__ == "__main__":
    unittest.main()
