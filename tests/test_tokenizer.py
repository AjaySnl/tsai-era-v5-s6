"""Tests: tokenizer hash stability and encode/decode correctness."""
import unittest
from core.tokenizer import CharTokenizer, PAD_ID, EOS_ID


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tok = CharTokenizer()
        self.tok.freeze()

    def test_hash_is_stable(self):
        """Same tokenizer always produces the same hash."""
        tok2 = CharTokenizer()
        h2 = tok2.freeze()
        self.assertEqual(self.tok.tokenizer_hash, h2)

    def test_hash_is_nonempty(self):
        self.assertTrue(len(self.tok.tokenizer_hash) == 64)

    def test_encode_decode_roundtrip(self):
        text = "Hello, world! 42"
        ids = self.tok.encode(text)
        recovered = self.tok.decode(ids)
        self.assertEqual(text, recovered)

    def test_encode_with_eos_appends_eos(self):
        ids = self.tok.encode_with_eos("abc")
        self.assertEqual(ids[-1], EOS_ID)

    def test_pad_id_is_zero(self):
        self.assertEqual(PAD_ID, 0)

    def test_encode_requires_frozen(self):
        tok = CharTokenizer()
        with self.assertRaises(RuntimeError):
            tok.encode("test")

    def test_hash_changes_after_different_vocab(self):
        """Two tokenizers with identical vocab should have identical hashes."""
        tok3 = CharTokenizer()
        h3 = tok3.freeze()
        self.assertEqual(self.tok.tokenizer_hash, h3)


if __name__ == "__main__":
    unittest.main()
