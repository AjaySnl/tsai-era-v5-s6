# Evidence Report

**Overall result: PASS**


| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | Tokenizer re-hash matches on second init |
| Shards and manifest integrity | PASS | manifests/shard_web_v1.json |
| Evaluation firewall | PASS | Eval shard 'shard_eval_doc_0056_eval_eval' and val shard 'shard_val_doc_0059_validation_validation' both blocked |
| SFT and agentic loss masks | PASS | Verified in Phase 5b: SFT prompt=0/response=1, agentic observation=0/output=1 |
| Packing correctness (attention_mask + position_ids) | PASS | Phase 5c: attention_mask and position_ids verified on multi-doc GREEDY batch |
| Crash simulation | PASS | SimulatedCrashError raised at step 25 |
| Crash recovery | PASS | checkpoints/step_00020.json |
| Replay | PASS | ledgers/consumption.db |
| Fork | PASS | ledgers/consumption.db (branch_id=fork_1d5c62e0) |
| Mixture schedule, protected floors and OPUS | PASS | ledgers/opus_decisions.jsonl + mixture accounting |
| Consumption and learning ledgers | PASS | ledgers/learning.jsonl + consumption.db |
| Throughput and packing efficiency | PASS | performance.json |
| Checkpoint, crash, resume, replay and fork | PASS | 4 checkpoints verified |
| Tests, evidence quality and documentation | PASS | tests/test_*.py |

## Details

- **Shards and manifest integrity**: 13 manifests validated
- **Evaluation firewall**: FirewallViolationError raised and caught
- **SFT and agentic loss masks**: SFT response tokens with loss=1: 27, agentic model output tokens with loss=1: 16
- **Packing correctness (attention_mask + position_ids)**: attention_mask: 32 real tokens=1, 32 pad tokens=0 | position_ids: starts_at_zero=True, doc_boundary_resets=[]
- **Crash simulation**: Last checkpoint: step_00020.json
- **Crash recovery**: Resume step 25 batch is deterministic and was not in pre-crash ledger
- **Replay**: Replayed steps 10-20: hash OK=True, spans OK=True, batch_ids OK=True
- **Fork**: Fork ran 5 independent steps
- **Mixture schedule, protected floors and OPUS**: OPUS decisions: {'FLOOR_OVERRIDE': 5, 'ACCEPTED': 29, 'REJECTED': 14, 'DEFERRED': 3}, planned_nonzero=True, deferred_queue_len=0
- **Consumption and learning ledgers**: 50 events, token-level loss stored, no_duplicates=True, no_skips=True
- **Throughput and packing efficiency**: PAD util=68.8% vs GREEDY util=70.3% — useful-loss-bearing < raw when padding present
- **Checkpoint, crash, resume, replay and fork**: Crash at 25, resumed from step 20
- **Tests, evidence quality and documentation**: 67 tests, 0 failures
