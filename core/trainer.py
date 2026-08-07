"""
Tiny model and training loop.

TinyModel: a 3-layer embedding + linear MLP that produces real gradients
and real loss values, running entirely on CPU.

SimulatedCrashError is a real Python exception raised inside the training
loop at CRASH_STEP — it is caught at the run_demo.py level, not faked.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import CRASH_STEP, SEED
from core.packing import PackedBatch

logger = logging.getLogger(__name__)


class SimulatedCrashError(RuntimeError):
    """Raised deliberately at CRASH_STEP to simulate a training crash."""


# Controls whether the crash fires. Set to False after crash is handled.
_crash_armed: bool = True


def arm_crash() -> None:
    """Re-arm the crash trigger (call at start of fresh run)."""
    global _crash_armed
    _crash_armed = True


def disarm_crash() -> None:
    """Disarm the crash trigger (call after crash is handled, before resume)."""
    global _crash_armed
    _crash_armed = False


class TinyModel(nn.Module):
    """
    Minimal character-level language model.
    Architecture: Embedding → Linear → ReLU → Linear → logits
    Intentionally tiny (~10K parameters) — correctness, not scale.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1        = nn.Linear(embed_dim, hidden_dim)
        self.fc2        = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (seq_len,) int tensor
        Returns logits: (seq_len, vocab_size)
        """
        x = self.embedding(token_ids)       # (seq_len, embed_dim)
        x = F.relu(self.fc1(x))             # (seq_len, hidden_dim)
        return self.fc2(x)                  # (seq_len, vocab_size)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(vocab_size: int, seed: int = SEED) -> TinyModel:
    torch.manual_seed(seed)
    model = TinyModel(vocab_size)
    model.train()
    return model


def build_optimizer(model: TinyModel) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def capture_optimizer_state(optimizer: torch.optim.Adam) -> dict:
    """Serialize optimizer state to a JSON-safe dict."""
    state = optimizer.state_dict()
    serialized: dict = {"param_groups": [], "state": {}}
    for pg in state["param_groups"]:
        serialized["param_groups"].append({k: v for k, v in pg.items() if k != "params"})
    for k, v in state["state"].items():
        entry = {}
        for name, val in v.items():
            if isinstance(val, torch.Tensor):
                entry[name] = val.tolist()
            else:
                entry[name] = val
        serialized["state"][str(k)] = entry
    return serialized


def restore_optimizer_state(
    optimizer: torch.optim.Adam,
    state_dict: dict,
    model: "TinyModel",
) -> None:
    """
    Restore Adam optimizer state from a serialised state dict.
    Reconstructs step count, exp_avg, and exp_avg_sq for each parameter.
    Called by run_demo._restore_optimizer_from_checkpoint (D16 fix — was a pass stub).
    """
    if not state_dict or not state_dict.get("state"):
        return
    try:
        params = list(model.parameters())
        for str_key, saved in state_dict["state"].items():
            idx = int(str_key)
            if idx >= len(params):
                continue
            p = params[idx]
            state = optimizer.state[p]
            state["step"] = torch.tensor(float(saved.get("step", 0)))
            if "exp_avg" in saved:
                state["exp_avg"] = torch.tensor(saved["exp_avg"])
            if "exp_avg_sq" in saved:
                state["exp_avg_sq"] = torch.tensor(saved["exp_avg_sq"])
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"restore_optimizer_state: partial restore ({e})"
        )


def train_step(
    batch: PackedBatch,
    model: TinyModel,
    optimizer: torch.optim.Adam,
    step: int,
) -> Tuple[float, float, List[float]]:
    """
    Execute one training step.

    Returns:
        loss (float)           — mean cross-entropy over loss-masked tokens
        gradient_norm (float)  — L2 norm of all gradients
        per_token_loss (List)  — cross-entropy per token position

    Raises SimulatedCrashError at CRASH_STEP.
    """
    if step == CRASH_STEP and _crash_armed:
        raise SimulatedCrashError(
            f"Simulated crash at step {step} — checkpoint at step "
            f"{step - (step % 10)} should be used for recovery."
        )

    optimizer.zero_grad()

    token_tensor = torch.tensor(batch.token_ids, dtype=torch.long)
    loss_mask    = torch.tensor(batch.loss_mask,  dtype=torch.float)

    # Shift for next-token prediction
    inputs  = token_tensor[:-1]
    targets = token_tensor[1:]
    mask    = loss_mask[1:]

    if inputs.numel() == 0:
        return 0.0, 0.0, []

    logits = model(inputs)   # (seq-1, vocab_size)

    # Per-token cross-entropy
    ce_all = F.cross_entropy(logits, targets, reduction="none")
    per_token_loss = ce_all.detach().tolist()

    # Masked mean
    masked_loss = (ce_all * mask).sum() / (mask.sum() + 1e-8)

    masked_loss.backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()

    return masked_loss.item(), grad_norm, per_token_loss


def measure_throughput(
    model: TinyModel,
    batch: PackedBatch,
    n_warmup: int = 2,
    n_measure: int = 5,
) -> dict:
    """Measure tokens/sec and useful-loss-bearing tokens/sec."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(n_warmup):
        try:
            train_step(batch, model, optimizer, step=999)
        except Exception:
            pass

    # Measurement
    t0 = time.perf_counter()
    for _ in range(n_measure):
        try:
            train_step(batch, model, optimizer, step=999)
        except Exception:
            pass
    elapsed = time.perf_counter() - t0

    total_tokens  = batch.seq_len * n_measure
    useful_tokens = batch.useful_tokens * n_measure
    tokens_per_sec        = total_tokens / max(elapsed, 1e-9)
    useful_tokens_per_sec = useful_tokens / max(elapsed, 1e-9)

    return {
        "tokens_per_sec":              round(tokens_per_sec, 1),
        "useful_loss_bearing_per_sec": round(useful_tokens_per_sec, 1),
        "packing_utilization":         round(batch.utilization, 4),
        "seq_len":                     batch.seq_len,
        "useful_tokens":               batch.useful_tokens,
        "elapsed_sec":                 round(elapsed, 4),
    }
