"""Tests: shard immutability and manifest validation."""
import unittest
from core.tokenizer import CharTokenizer
from core.shard import Shard, build_manifest, validate_manifest


def _make_shard() -> Shard:
    tok = CharTokenizer()
    tok.freeze()
    tokens = tok.encode("Hello shard test.")
    return Shard(
        shard_id="test_shard_001",
        doc_ids=["doc_0001"],
        token_ids=tokens,
        lane="web",
        tokenizer_hash=tok.tokenizer_hash,
        source="test_corpus",
    )


class TestShard(unittest.TestCase):

    def test_shard_is_immutable(self):
        shard = _make_shard()
        with self.assertRaises(AttributeError):
            shard.lane = "code"

    def test_content_hash_is_stable(self):
        shard1 = _make_shard()
        shard2 = _make_shard()
        self.assertEqual(shard1.content_hash, shard2.content_hash)

    def test_token_count_matches(self):
        shard = _make_shard()
        self.assertEqual(shard.token_count, len(shard.token_ids))

    def test_manifest_validation_passes(self):
        shard = _make_shard()
        manifest = build_manifest(shard)
        self.assertTrue(validate_manifest(shard, manifest))

    def test_manifest_detects_hash_tampering(self):
        shard = _make_shard()
        manifest = build_manifest(shard)
        # Tamper with manifest hash
        object.__setattr__(manifest, "content_hash", "deadbeef" * 8)
        with self.assertRaises(ValueError):
            validate_manifest(shard, manifest)

    def test_manifest_detects_token_count_tampering(self):
        shard = _make_shard()
        manifest = build_manifest(shard)
        object.__setattr__(manifest, "token_count", 9999)
        with self.assertRaises(ValueError):
            validate_manifest(shard, manifest)


if __name__ == "__main__":
    unittest.main()
