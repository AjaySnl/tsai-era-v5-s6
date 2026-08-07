"""
Shared configuration for the V5 Training Data Execution System.
All constants are centralised here so every module reads from one source.
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED: int = 42

# ── Sequence / batch geometry ─────────────────────────────────────────────────
SEQ_LEN: int = 64          # tokens per packed sequence
VOCAB_SIZE: int = 100      # char-level tokenizer (set after freeze)
CRASH_STEP: int = 25       # step at which the simulated crash fires
CHECKPOINT_EVERY: int = 10 # save a checkpoint every N steps
TOTAL_STEPS: int = 50      # total training steps in the main run

# ── Output paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "submission_artifacts"
MANIFESTS_DIR  = ARTIFACTS_DIR / "manifests"
LEDGERS_DIR    = ARTIFACTS_DIR / "ledgers"
CHECKPOINTS_DIR= ARTIFACTS_DIR / "checkpoints"

LOG_PATH         = ARTIFACTS_DIR / "run.log"
EVIDENCE_JSON    = ARTIFACTS_DIR / "evidence.json"
EVIDENCE_MD      = ARTIFACTS_DIR / "evidence.md"
PERFORMANCE_JSON = ARTIFACTS_DIR / "performance.json"
CONSUMPTION_DB   = LEDGERS_DIR  / "consumption.db"
LEARNING_JSONL   = LEDGERS_DIR  / "learning.jsonl"
OPUS_JSONL       = LEDGERS_DIR  / "opus_decisions.jsonl"


# ── Capability lanes ─────────────────────────────────────────────────────────
LANES: List[str] = ["web", "code", "indic", "stem", "agentic", "reasoning", "longctx"]
PROTECTED_LANES: List[str] = ["indic", "agentic"]   # always-on floors


# ── Curriculum stages ─────────────────────────────────────────────────────────
@dataclass
class CurriculumStageDef:
    stage_id: str
    token_start: int
    token_end: int
    mixture_weights: Dict[str, float]
    protected_floors: Dict[str, float]
    warmup_tokens: int = 1000

CURRICULUM: List[CurriculumStageDef] = [
    CurriculumStageDef(
        stage_id="seed",
        token_start=0,
        token_end=1_000,
        mixture_weights={
            "web": 0.67, "code": 0.15, "indic": 0.08,
            "stem": 0.10, "agentic": 0.00, "reasoning": 0.00, "longctx": 0.00,
        },
        protected_floors={"indic": 0.06, "agentic": 0.02},
        warmup_tokens=100,
    ),
    CurriculumStageDef(
        stage_id="general",
        token_start=1_000,
        token_end=3_000,
        mixture_weights={
            "web": 0.50, "code": 0.20, "indic": 0.10,
            "stem": 0.12, "agentic": 0.00, "reasoning": 0.08, "longctx": 0.00,
        },
        protected_floors={"indic": 0.06, "agentic": 0.02},
        warmup_tokens=200,
    ),
    CurriculumStageDef(
        stage_id="specialisation",
        token_start=3_000,
        token_end=999_999,
        mixture_weights={
            "web": 0.25, "code": 0.25, "indic": 0.12,
            "stem": 0.18, "agentic": 0.03, "reasoning": 0.12, "longctx": 0.05,
        },
        protected_floors={"indic": 0.06, "agentic": 0.02},
        warmup_tokens=300,
    ),
]

# ── Document split tags ───────────────────────────────────────────────────────
SPLIT_TRAIN      = "train"
SPLIT_EVAL       = "eval"
SPLIT_VALIDATION = "validation"
