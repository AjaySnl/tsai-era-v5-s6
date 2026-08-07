"""
Sequence packing with five policies and correct mask generation.

Five packing policies:
  PAD_ONLY           - pad each sample to seq_len; preserves structure; wastes compute
  CONCAT_CHOP        - concatenate all tokens with EOS, then cut fixed windows
  GREEDY             - place each sequence in the first available slot
  BEST_FIT           - sort by length, find tightest fit (minimises wasted positions)
  STRUCTURE_PRESERVING - like GREEDY but never mixes documents within same window

Loss mask rules (set by training mode):
  pretrain  - all real (non-PAD) tokens = 1
  sft       - response tokens = 1, prompt tokens = 0, PAD = 0
  agentic   - model output tokens = 1, observations/context = 0, PAD = 0
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.tokenizer import PAD_ID, EOS_ID


class PackingPolicy(str, Enum):
    PAD_ONLY              = "pad_only"
    CONCAT_CHOP           = "concat_chop"
    GREEDY                = "greedy"
    BEST_FIT              = "best_fit"
    STRUCTURE_PRESERVING  = "structure_preserving"


class TrainingMode(str, Enum):
    PRETRAIN = "pretrain"
    SFT      = "sft"
    AGENTIC  = "agentic"


@dataclass
class SequenceRecord:
    """One tokenized document ready for packing."""
    doc_id:    str
    shard_id:  str
    token_ids: List[int]
    mode:      TrainingMode = TrainingMode.PRETRAIN
    # For SFT: index where the response starts
    response_start: int = 0
    # For agentic: list of (start, end) spans that are model output
    model_output_spans: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class PackedBatch:
    """A fixed-length packed sequence ready for the training loop."""
    batch_id:       str
    token_ids:      List[int]
    loss_mask:      List[int]        # 1 = calculate loss, 0 = context only
    attention_mask: List[int]        # 1 = real token, 0 = padding
    position_ids:   List[int]        # 0-based position within the window
    doc_boundaries: List[int]        # token indices where a new doc begins
    shard_ids:      List[str]        # which shards contributed to this batch
    packing_policy: str
    seq_len:        int
    useful_tokens:  int              # tokens where loss_mask == 1
    batch_hash:     str              # sha256(token_ids + loss_mask)

    @property
    def utilization(self) -> float:
        """Fraction of positions carrying useful loss-bearing tokens."""
        return self.useful_tokens / max(self.seq_len, 1)


# ── Public API ────────────────────────────────────────────────────────────────

def pack(
    sequences: List[SequenceRecord],
    policy: PackingPolicy,
    seq_len: int,
    batch_id_prefix: str = "batch",
) -> List[PackedBatch]:
    """
    Pack a list of SequenceRecords into PackedBatch objects.
    Returns one PackedBatch per packed window.
    """
    if policy == PackingPolicy.PAD_ONLY:
        windows = _pack_pad_only(sequences, seq_len)
    elif policy == PackingPolicy.CONCAT_CHOP:
        windows = _pack_concat_chop(sequences, seq_len)
    elif policy == PackingPolicy.GREEDY:
        windows = _pack_greedy(sequences, seq_len)
    elif policy == PackingPolicy.BEST_FIT:
        windows = _pack_best_fit(sequences, seq_len)
    elif policy == PackingPolicy.STRUCTURE_PRESERVING:
        windows = _pack_structure_preserving(sequences, seq_len)
    else:
        raise ValueError(f"Unknown packing policy: {policy}")

    batches = []
    for i, (window_tokens, window_seqs, boundaries) in enumerate(windows):
        bid = f"{batch_id_prefix}_{i:04d}"
        loss_mask  = _build_loss_mask(window_tokens, window_seqs, boundaries, seq_len)
        attn_mask  = [1 if t != PAD_ID else 0 for t in window_tokens]
        pos_ids    = _build_position_ids(window_tokens, boundaries, seq_len)
        useful     = sum(loss_mask)
        batch_hash = _hash_batch(window_tokens, loss_mask)
        shard_ids  = list({seq.shard_id for seq in window_seqs})

        batches.append(PackedBatch(
            batch_id       = bid,
            token_ids      = window_tokens,
            loss_mask      = loss_mask,
            attention_mask = attn_mask,
            position_ids   = pos_ids,
            doc_boundaries = boundaries,
            shard_ids      = shard_ids,
            packing_policy = policy.value,
            seq_len        = seq_len,
            useful_tokens  = useful,
            batch_hash     = batch_hash,
        ))
    return batches


# ── Packing implementations ───────────────────────────────────────────────────

_Window = Tuple[List[int], List[SequenceRecord], List[int]]  # tokens, seqs, boundaries


def _pad_window(tokens: List[int], seq_len: int) -> List[int]:
    """Pad or truncate to exact seq_len."""
    if len(tokens) >= seq_len:
        return tokens[:seq_len]
    return tokens + [PAD_ID] * (seq_len - len(tokens))


def _pack_pad_only(sequences: List[SequenceRecord], seq_len: int) -> List[_Window]:
    windows = []
    for seq in sequences:
        tokens = _pad_window(seq.token_ids[:seq_len], seq_len)
        windows.append((tokens, [seq], [0]))
    return windows


def _pack_concat_chop(sequences: List[SequenceRecord], seq_len: int) -> List[_Window]:
    """Concatenate with EOS markers, then cut fixed windows."""
    flat: List[int] = []
    for seq in sequences:
        flat.extend(seq.token_ids)
        flat.append(EOS_ID)

    windows = []
    for start in range(0, len(flat), seq_len):
        chunk = flat[start: start + seq_len]
        if not chunk:
            break
        chunk = _pad_window(chunk, seq_len)
        windows.append((chunk, sequences, [0]))
    return windows


def _pack_greedy(sequences: List[SequenceRecord], seq_len: int) -> List[_Window]:
    """Place each sequence in the first window with enough remaining space."""
    slots: List[List[int]] = []
    slot_seqs: List[List[SequenceRecord]] = []
    slot_bounds: List[List[int]] = []

    for seq in sequences:
        tokens = seq.token_ids[:seq_len]
        placed = False
        for i, slot in enumerate(slots):
            remaining = seq_len - len(slot)
            if remaining >= len(tokens) + 1:  # +1 for EOS
                if slot:
                    slot_bounds[i].append(len(slot))
                slot.extend(tokens)
                slot.append(EOS_ID)
                slot_seqs[i].append(seq)
                placed = True
                break
        if not placed:
            slots.append(list(tokens) + [EOS_ID])
            slot_seqs.append([seq])
            slot_bounds.append([0])

    windows = []
    for tokens, seqs, bounds in zip(slots, slot_seqs, slot_bounds):
        windows.append((_pad_window(tokens, seq_len), seqs, bounds))
    return windows


def _pack_best_fit(sequences: List[SequenceRecord], seq_len: int) -> List[_Window]:
    """Sort by length descending, place into tightest fitting slot."""
    sorted_seqs = sorted(sequences, key=lambda s: len(s.token_ids), reverse=True)
    slots: List[List[int]] = []
    slot_seqs: List[List[SequenceRecord]] = []
    slot_bounds: List[List[int]] = []
    slot_remaining: List[int] = []

    for seq in sorted_seqs:
        tokens = seq.token_ids[:seq_len]
        needed = len(tokens) + 1  # +1 for EOS

        # Find slot with least remaining space that still fits (best-fit heuristic)
        best_idx = -1
        best_rem = seq_len + 1
        for i, rem in enumerate(slot_remaining):
            if rem >= needed and rem < best_rem:
                best_rem = rem
                best_idx = i

        if best_idx >= 0:
            slot = slots[best_idx]
            if slot:
                slot_bounds[best_idx].append(len(slot))
            slot.extend(tokens)
            slot.append(EOS_ID)
            slot_seqs[best_idx].append(seq)
            slot_remaining[best_idx] -= needed
        else:
            slots.append(list(tokens) + [EOS_ID])
            slot_seqs.append([seq])
            slot_bounds.append([0])
            slot_remaining.append(seq_len - needed)

    windows = []
    for tokens, seqs, bounds in zip(slots, slot_seqs, slot_bounds):
        windows.append((_pad_window(tokens, seq_len), seqs, bounds))
    return windows


def _pack_structure_preserving(sequences: List[SequenceRecord], seq_len: int) -> List[_Window]:
    """
    Each window contains exactly one document (or one document padded).
    Unrelated documents NEVER share attention context.
    Required for SFT and agentic traces.
    """
    windows = []
    for seq in sequences:
        tokens = _pad_window(seq.token_ids[:seq_len], seq_len)
        windows.append((tokens, [seq], [0]))
    return windows


# ── Mask builders ─────────────────────────────────────────────────────────────

def _build_loss_mask(
    tokens: List[int],
    seqs: List[SequenceRecord],
    boundaries: List[int],
    seq_len: int,
) -> List[int]:
    """Build loss mask based on the training mode of sequences in this window."""
    mask = [0] * seq_len

    # Determine dominant mode
    modes = [s.mode for s in seqs]
    mode = modes[0] if modes else TrainingMode.PRETRAIN

    if mode == TrainingMode.PRETRAIN:
        # All real (non-PAD) tokens contribute to loss
        for i, t in enumerate(tokens):
            mask[i] = 1 if t != PAD_ID else 0

    elif mode == TrainingMode.SFT:
        # Only response tokens (after response_start) get loss
        for seq in seqs:
            start = seq.response_start
            for i in range(start, len(seq.token_ids)):
                if i < seq_len and tokens[i] != PAD_ID:
                    mask[i] = 1

    elif mode == TrainingMode.AGENTIC:
        # Only model output spans get loss; observations are context-only
        for seq in seqs:
            for (span_start, span_end) in seq.model_output_spans:
                for i in range(span_start, min(span_end, seq_len)):
                    if tokens[i] != PAD_ID:
                        mask[i] = 1

    return mask


def _build_position_ids(
    tokens: List[int],
    boundaries: List[int],
    seq_len: int,
) -> List[int]:
    """
    Assign position IDs.
    Position resets to 0 at each document boundary (EOS token).
    """
    pos_ids = []
    pos = 0
    boundary_set = set(boundaries)
    for i, t in enumerate(tokens):
        if i in boundary_set and i > 0:
            pos = 0  # reset at new document
        pos_ids.append(pos)
        pos += 1
    return pos_ids


def _hash_batch(token_ids: List[int], loss_mask: List[int]) -> str:
    """Stable SHA-256 of token_ids concatenated with loss_mask."""
    raw = ",".join(str(t) for t in token_ids)
    raw += "|" + ",".join(str(m) for m in loss_mask)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Utilization helper ────────────────────────────────────────────────────────

def packing_utilization(batches: List[PackedBatch]) -> float:
    """Overall utilization across a list of batches."""
    total_positions = sum(b.seq_len for b in batches)
    useful_positions = sum(b.useful_tokens for b in batches)
    return useful_positions / max(total_positions, 1)
