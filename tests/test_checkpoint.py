"""Tests: checkpoint save/load and hash integrity."""
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np

from core.checkpoint import (
    save_checkpoint, load_checkpoint, capture_rng_state,
    Checkpoint,
)
from core.trainer import build_model, build_optimizer, capture_optimizer_state


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._out = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_checkpoint(self, step: int = 10) -> Checkpoint:
        model = build_model(vocab_size=98)
        optimizer = build_optimizer(model)
        torch_rng, np_rng = capture_rng_state()
        return Checkpoint(
            step              = step,
            branch_id         = "main",
            ledger_offset     = 10,
            batch_id          = f"batch_{step:04d}_0000",
            tokens_consumed   = step * 64,
            curriculum_stage  = "seed",
            model_state       = {k: v.tolist() for k, v in model.state_dict().items()},
            optimizer_state   = capture_optimizer_state(optimizer),
            torch_rng_state   = torch_rng,
            numpy_rng_state   = np_rng,
            mixer_lane_counts = {"web": step, "code": 0, "indic": 0, "stem": 0,
                                 "agentic": 0, "reasoning": 0, "longctx": 0},
        )

    def test_save_and_load_roundtrip(self):
        ckpt = self._make_checkpoint(step=10)
        path = save_checkpoint(ckpt, output_dir=self._out)
        loaded = load_checkpoint(path)
        self.assertEqual(loaded.step, 10)
        self.assertEqual(loaded.branch_id, "main")
        self.assertEqual(loaded.ledger_offset, 10)

    def test_hash_is_verified_on_load(self):
        ckpt = self._make_checkpoint(step=5)
        path = save_checkpoint(ckpt, output_dir=self._out)
        # Tamper with file
        import json
        with open(path) as f:
            data = json.load(f)
        data["step"] = 999  # tamper
        with open(path, "w") as f:
            json.dump(data, f)
        with self.assertRaises(ValueError):
            load_checkpoint(path)

    def test_checkpoint_hash_changes_with_content(self):
        ckpt1 = self._make_checkpoint(step=1)
        ckpt2 = self._make_checkpoint(step=2)
        path1 = save_checkpoint(ckpt1, output_dir=self._out)
        path2 = save_checkpoint(ckpt2, output_dir=self._out)
        loaded1 = load_checkpoint(path1)
        loaded2 = load_checkpoint(path2)
        self.assertNotEqual(loaded1.checkpoint_hash, loaded2.checkpoint_hash)


if __name__ == "__main__":
    unittest.main()
