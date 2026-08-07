"""
Tiny character-level tokenizer.
Vocabulary: printable ASCII (95 chars) + 3 special tokens.
Must be frozen before shards are created — tokenizer_hash ties every shard
to the exact vocabulary that produced its token IDs.
"""
from __future__ import annotations

import hashlib
import json
import string
from typing import List


# Special tokens
EOS_TOKEN  = "<EOS>"
PAD_TOKEN  = "<PAD>"
MASK_TOKEN = "<MASK>"
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN, MASK_TOKEN]

PAD_ID  = 0
EOS_ID  = 1
MASK_ID = 2


class CharTokenizer:
    """
    Deterministic char-level tokenizer.
    Call freeze() once before any encoding — this locks the vocabulary and
    computes the tokenizer_hash that all shard manifests will record.
    """

    def __init__(self) -> None:
        # Build vocab: specials first, then printable ASCII in sorted order
        chars = sorted(string.printable)
        self._vocab: List[str] = SPECIAL_TOKENS + chars
        self._token_to_id: dict[str, int] = {t: i for i, t in enumerate(self._vocab)}
        self._id_to_token: dict[int, str] = {i: t for i, t in enumerate(self._vocab)}
        self._frozen: bool = False
        self.tokenizer_hash: str = ""
        self.vocab_size: int = len(self._vocab)

    # ── Public API ─────────────────────────────────────────────────────────────

    def freeze(self) -> str:
        """
        Lock the vocabulary and compute a stable hash.
        Returns the tokenizer_hash string.
        Must be called before any encode/decode calls.
        """
        vocab_repr = json.dumps(self._vocab, ensure_ascii=True, sort_keys=False)
        self.tokenizer_hash = hashlib.sha256(vocab_repr.encode()).hexdigest()
        self._frozen = True
        return self.tokenizer_hash

    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token IDs. Unknown chars become MASK_ID."""
        self._require_frozen()
        ids = []
        for ch in text:
            ids.append(self._token_to_id.get(ch, MASK_ID))
        return ids

    def encode_with_eos(self, text: str) -> List[int]:
        """Encode text and append an EOS token."""
        return self.encode(text) + [EOS_ID]

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text, skipping special tokens."""
        return "".join(
            self._id_to_token.get(i, "")
            for i in ids
            if i not in (PAD_ID, EOS_ID, MASK_ID)
        )

    def vocab_repr(self) -> List[str]:
        """Return the full ordered vocabulary list."""
        return list(self._vocab)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _require_frozen(self) -> None:
        if not self._frozen:
            raise RuntimeError("Tokenizer must be frozen before encoding. Call freeze() first.")


# Module-level singleton — import and use everywhere
tokenizer = CharTokenizer()
