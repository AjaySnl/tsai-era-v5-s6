# ERA V5 Session 6 — Training Data Execution System

A complete, production-quality miniature training data execution system demonstrating the full lifecycle from raw documents to auditable training runs, crash recovery, replay, and fork.

## Quick Start

```bash
pip install -r requirements.txt
python run_demo.py
```

All evidence is generated automatically in `submission_artifacts/`.

## Architecture

```
core/
├── config.py           Shared constants, seeds, curriculum stage definitions
├── corpus.py           Synthetic document generator (deterministic, seed=42)
├── tokenizer.py        Char-level tokenizer with freeze() + tokenizer_hash
├── shard.py            Immutable tokenized shards + manifest validation
├── firewall.py         Eval/validation firewall — blocks shards from training
├── mixture.py          Curriculum stages, lane weights, protected floors
├── packing.py          5 packing policies, loss/attention/position masks
├── opus.py             OPUS-style data selector with audit trail (JSONL)
├── ledger.py           Consumption ledger (SQLite, append-only via triggers)
├── learning_ledger.py  Learning ledger (JSONL) — attaches outcomes to data
├── checkpoint.py       Checkpoint save/load with hash integrity verification
├── trainer.py          TinyModel (PyTorch CPU) + train_step + crash simulation
└── audit.py            Audit subsystem — generates evidence bundle

tests/                  67 unit tests covering all critical invariants
run_demo.py             Single entry point — runs the complete demonstration
```

## Pipeline

```
documents (56 train + 3 eval + 3 validation)
  → tokenizer frozen → tokenizer_hash computed
  → tokenized shards (one per capability lane)
  → manifests (content_hash, tokenizer_hash, cleaning_pipeline_hash)
  → eval shards registered in firewall (never_train=True)
  → mixture schedule compiled (3 curriculum stages)
  → training loop (steps 1–50):
      - mixture scheduler selects lane
      - firewall checks shard
      - OPUS scores batch (40% keep rate) — protected floors bypass
      - packing: tokens → PackedBatch (loss mask, attention mask, position IDs)
      - train_step() → real loss, gradient norm
      - consumption ledger (SQLite) appended
      - learning ledger (JSONL) appended
      - checkpoint every 10 steps
  → crash at step 25 (SimulatedCrashError)
  → resume from step-20 checkpoint
      - restore model + RNG state
      - verify next batch hash matches original ledger entry [PASS]
  → continue to step 50
  → replay steps 10–20 → verify all batch hashes match [PASS]
  → fork from step-30 checkpoint → new branch_id
  → audit → evidence bundle generated
  → 67 unit tests run
```

## Key Design Decisions

### Immutable Shards
`Shard.__setattr__` raises `AttributeError` after creation. Any tampering is detectable via `content_hash` (SHA-256 of token IDs).

### Append-Only Consumption Ledger
SQLite with `BEFORE UPDATE` and `BEFORE DELETE` triggers that raise errors. History cannot be silently modified.

### OPUS Data Selection
Deterministic proxy scoring via SHA-256 of `shard_id:step`. Protected lanes (Indic, Agentic) bypass the selector with `FLOOR_OVERRIDE` status. All decisions logged to `opus_decisions.jsonl`.

### Crash Recovery
`SimulatedCrashError` is a real Python exception raised at `CRASH_STEP=25`. Recovery loads the step-20 checkpoint, restores model + optimizer + RNG state + mixer lane counts, then proves the next batch is the expected batch: an independent probe mixer (restored to the same checkpoint state) selects the lane and regenerates the batch. The probe result must match the actual restored mixer's result — proving state was faithfully restored.

### Mixer State in Checkpoints
`Checkpoint` captures `mixer_lane_counts` so that on resume, the lane-selection history is fully restored. This ensures the resumed run selects the same lane at step 25 as an uncrashed run would have.

### Replay
Historical batches are reconstructed from ledger events by loading the original shard (identified by `shard_id` in the ledger) and repacking with the same policy and sequence length. Hashes are compared step-by-step.

### Fork
A new `branch_id` (UUID4) is assigned. The fork runs independently from the same model checkpoint but writes to a separate ledger branch, making the divergence point explicit and auditable.

### Loss Masks
- **Pre-training**: all non-PAD tokens = 1
- **SFT**: only response tokens (after `response_start` index) = 1
- **Agentic**: only model output spans = 1; tool observations = 0

### Packing Policies
1. **PAD_ONLY** — simple, preserves structure, wastes compute
2. **CONCAT_CHOP** — efficient for plain text, cuts at fixed windows
3. **GREEDY** — place each doc in first available slot (speed-optimised)
4. **BEST_FIT** — sort by length, find tightest fit (utilisation-optimised)
5. **STRUCTURE_PRESERVING** — one doc per window (required for SFT/agentic)

## Generated Artifacts

```
submission_artifacts/
├── run.log             Complete execution log with [PASS] events
├── evidence.json       Machine-readable requirement results
├── evidence.md         Human-readable evidence table
├── manifests/          One JSON manifest per shard (13 total)
├── ledgers/
│   ├── consumption.db  SQLite consumption ledger (append-only)
│   ├── learning.jsonl  Learning ledger (one line per training step)
│   └── opus_decisions.jsonl  OPUS accept/reject/defer audit trail
├── checkpoints/        step_00010.json through step_00040.json
└── performance.json    Throughput and packing efficiency metrics
```

## Running Tests

```bash
python -m unittest discover tests -v
```

All 67 tests cover: tokenizer hash stability, shard immutability, manifest validation, packing correctness (loss/attention/position masks), SFT and agentic mask verification, firewall enforcement (eval + validation), ledger append-only integrity, checkpoint hash verification, optimizer state restore, mixer state restore, resume correctness, replay determinism, no-skip/no-repeat invariants, DEFERRED vs REJECTED distinction, per-token loss storage, planned token accounting, and lane/shard consistency.

## Expected Output

```
[PASS] tokenizer_hash_verified
[PASS] eval_shard_blocked
[PASS] checkpoint_saved: step=10
[PASS] checkpoint_saved: step=20
[CRASH] Simulated crash at step 25
[PASS] resume_next_batch_matched
[PASS] replay_hash_matched
Overall: PASS
```

---

*ERA V5 · Session 6 Assignment · The School of AI · August 2026*
