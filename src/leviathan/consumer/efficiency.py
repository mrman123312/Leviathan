"""Falsifiable cache/selection optimizations with explicit validity boundaries."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor


@dataclass(frozen=True)
class CacheScope:
    model_revision: str
    weight_epoch: int
    request_id: str
    precision: str
    operator: str
    state_epoch: int = 0


class ExactDeltaCache:
    """Exact row-local SDR reference, for pure deterministic row-local operators.

    All inputs/dependencies must be represented in x or scope. This is NOT valid
    for causal attention, DeltaNet, training, or mutable operators without epochs.
    Equality checks and gathers may cost more than recomputation: benchmark it.
    """
    def __init__(self):
        self._x = self._y = self._scope = None
        self.last_reused_rows = 0

    def clear(self):
        self._x = self._y = self._scope = None

    def run(self, x: Tensor, op: Callable[[Tensor], Tensor], scope: CacheScope) -> Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("Exact cache is inference-only; do not sever training gradients")
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise ValueError("Finite row-local inputs required")
        compatible = (self._scope == scope and self._x is not None and
                      self._x.shape == x.shape and self._x.device == x.device and self._x.dtype == x.dtype)
        if not compatible:
            y = op(x)
            self.last_reused_rows = 0
        else:
            changed = (x != self._x).any(-1)
            rows = changed.nonzero(as_tuple=True)[0]
            self.last_reused_rows = len(x) - len(rows)
            y = self._y.clone()
            if len(rows):
                y = y.index_copy(0, rows, op(x[rows]))
        self._x, self._y, self._scope = x.clone(), y.clone(), scope
        return y


class ByteLRU:
    """Bounded hot-tile/prefix storage. Caller keys include revision/state/format."""
    def __init__(self, max_bytes: int):
        if max_bytes < 1:
            raise ValueError("Positive byte budget required")
        self.max_bytes = max_bytes
        self.bytes = 0
        self.items = OrderedDict()

    def put(self, key: tuple, value: Tensor) -> bool:
        if value.requires_grad:
            raise ValueError("Cache detached inference tensors only")
        size = value.numel() * value.element_size()
        if size > self.max_bytes:
            return False
        if key in self.items:
            old = self.items.pop(key)
            self.bytes -= old.numel() * old.element_size()
        while self.bytes + size > self.max_bytes:
            _, old = self.items.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
        self.items[key] = value.clone()
        self.bytes += size
        return True

    def get(self, key: tuple) -> Tensor | None:
        value = self.items.pop(key, None)
        if value is not None:
            self.items[key] = value
            return value.clone()
        return None


def certified_topk_reuse(old_query: Tensor, new_query: Tensor, keys: Tensor,
                         old_scores: Tensor, ids: Tensor) -> bool:
    """Sufficient top-k set certificate for EXACT linear dot-product scores.

    Every score moves by at most eps=||dq||_2 max_j ||k_j||_2.
    A selected/unselected gap > 2 eps preserves the set, not its probabilities.
    Caller must guarantee identical keys/precision and recompute selected scores.
    Floating-point guard makes this a conservative reference, not interval arithmetic.
    """
    if old_query.ndim != 1 or new_query.shape != old_query.shape or keys.ndim != 2:
        raise ValueError("Expected vector queries and key matrix")
    if not all(torch.isfinite(t).all() for t in (old_query, new_query, keys, old_scores)):
        return False
    if len(ids) == 0 or len(torch.unique(ids)) != len(ids):
        raise ValueError("Nonempty unique IDs required")
    if len(ids) == len(keys):
        return True
    chosen = torch.zeros(len(keys), dtype=torch.bool, device=keys.device)
    chosen[ids] = True
    gap = old_scores[chosen].min() - old_scores[~chosen].max()
    eps = (new_query.double() - old_query.double()).norm() * keys.double().norm(dim=-1).max()
    guard = 16 * torch.finfo(old_scores.dtype).eps * (old_scores.abs().max() + 1)
    return bool(gap.double() > 2 * eps + guard)


def coefficient_delta(old_output: Tensor, bodies: Tensor, old_weights: Tensor,
                      new_weights: Tensor) -> Tensor:
    """Stable-body coefficient delta: algebraically exact, floating-point approximate.

    Only valid when bodies are unchanged. A refresh is needed for numerical drift.
    This is never a certificate for skipping changed nonlinear cell computation.
    """
    return old_output + ((new_weights - old_weights)[..., None] * bodies).sum(-2)


def certified_greedy_token(old_logits: Tensor, max_logit_change: float) -> int | None:
    """Logit-margin certificate; a caller-supplied PROVEN bound is required.

    Does not certify stochastic sampling, latent similarity, or model correctness.
    """
    if max_logit_change < 0 or not torch.isfinite(old_logits).all() or old_logits.numel() < 2:
        return None
    values, ids = old_logits.topk(2)
    if float(values[0] - values[1]) > 2 * max_logit_change:
        return int(ids[0])
    return None
