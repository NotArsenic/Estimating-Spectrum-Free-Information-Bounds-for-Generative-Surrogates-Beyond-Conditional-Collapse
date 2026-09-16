"""Fixed seed offsets, so evaluating or dumping a checkpoint is repeatable."""

import torch

EVAL_SINGLE = 1000  # single-sample draws (KL, W1)
EVAL_MULTI = 2000  # multi-sample CRPS ensemble
EVAL_HDR = 3000  # HDR-rank sampling
DUMP = 4000  # per-event dump sampling


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
