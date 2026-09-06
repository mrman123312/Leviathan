"""Local training objectives and disjoint evaluation manifests."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def sample_depth(minimum: int, maximum: int, generator: torch.Generator | None = None) -> int:
    if minimum < 1 or minimum > maximum:
        raise ValueError("Invalid depth range")
    return int(torch.randint(minimum, maximum + 1, (), generator=generator))


def shortcut_consistency(short: Tensor, long: Tensor) -> Tensor:
    return F.mse_loss(short.float(), long.detach().float())


def donor_distillation(student_logits: Tensor, donor_logits: Tensor) -> Tensor:
    return F.kl_div(F.log_softmax(student_logits.float(), -1),
                    F.softmax(donor_logits.detach().float(), -1), reduction="batchmean")


def halting_supervision(probabilities: Tensor, preferred_depth: Tensor) -> Tensor:
    if probabilities.ndim != 2 or preferred_depth.shape != (len(probabilities),):
        raise ValueError("Expected [batch, depth] probabilities and [batch] depth targets")
    if (preferred_depth < 1).any() or (preferred_depth > probabilities.shape[1]).any():
        raise ValueError("Halting target outside evaluated depths")
    steps = torch.arange(1, probabilities.shape[1] + 1, device=probabilities.device)
    targets = (steps[None] >= preferred_depth[:, None]).to(probabilities.dtype)
    return F.binary_cross_entropy(probabilities, targets)


def content_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvaluationSplit:
    training_hashes: frozenset[str]
    heldout_hashes: frozenset[str]

    def __post_init__(self):
        if not self.heldout_hashes or self.training_hashes & self.heldout_hashes:
            raise ValueError("Heldout set missing or overlaps training/replay data")

    def require_heldout(self, text: str):
        if content_hash(text) not in self.heldout_hashes:
            raise ValueError("Example not in frozen heldout manifest")


@dataclass(frozen=True)
class RetentionGate:
    max_relative_loss_increase: float = 0.02
    accuracy_tolerance: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in
               (self.max_relative_loss_increase, self.accuracy_tolerance)):
            raise ValueError("Finite nonnegative tolerances required")

    def evaluate(self, baseline: dict, candidate: dict) -> tuple[bool, list[str]]:
        reasons = []
        if not baseline:
            return False, ["No baseline metrics; absence of evidence is not a pass"]
        for name, original in baseline.items():
            if not isinstance(original, (int, float)) or not math.isfinite(original):
                reasons.append(f"invalid baseline metric: {name}")
                continue
            if name not in candidate:
                reasons.append(f"missing metric: {name}")
                continue
            value = candidate[name]
            if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value)):
                reasons.append(f"invalid metric: {name}")
            elif name.endswith("loss"):
                if value > original * (1 + self.max_relative_loss_increase):
                    reasons.append(f"retention regression: {name}")
            elif name.endswith("accuracy") and value < original - self.accuracy_tolerance:
                reasons.append(f"accuracy regression: {name}")
            elif name.endswith("seconds") and value >= original:
                reasons.append(f"no measured speed improvement: {name}")
        return not reasons, reasons
