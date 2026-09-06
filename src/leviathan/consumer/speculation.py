"""Speculative-decoding correctness primitives; no unmeasured speed claims.

Draft/target may be shallow/deep calls to the same weights. This file does not
pretend stock Qwen generate() exposes MTP or supports hybrid-cache rollback.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


def _probabilities(x: Tensor):
    if x.ndim < 1 or not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError("Finite nonnegative probabilities required")
    if not torch.allclose(x.sum(-1), torch.ones_like(x.sum(-1)), atol=1e-6, rtol=1e-6):
        raise ValueError("Probabilities must sum to one")


@dataclass(frozen=True)
class VerificationResult:
    tokens: tuple[int, ...]
    accepted: int
    rejected_at: int | None


def verify_greedy(draft: Tensor, target_logits: Tensor,
                  eos_token_id: int | None = None) -> VerificationResult:
    if draft.ndim != 1 or target_logits.ndim != 2 or len(target_logits) != len(draft) + 1:
        raise ValueError("Need k draft tokens and k+1 correctly aligned target distributions")
    if not torch.isfinite(target_logits).all():
        raise ValueError("Nonfinite target logits")
    out = []
    for i, token in enumerate(draft.tolist()):
        winner = int(target_logits[i].argmax())
        if token != winner:
            out.append(winner)
            return VerificationResult(tuple(out), i, i)
        out.append(token)
        if token == eos_token_id:
            return VerificationResult(tuple(out), i + 1, None)
    out.append(int(target_logits[-1].argmax()))
    return VerificationResult(tuple(out), len(draft), None)


def verify_sampled(draft: Tensor, draft_probs: Tensor, target_probs: Tensor, *,
                   generator: torch.Generator | None = None,
                   eos_token_id: int | None = None) -> VerificationResult:
    if draft.ndim != 1 or draft_probs.ndim != 2 or target_probs.ndim != 2:
        raise ValueError("Invalid speculative dimensions")
    if draft_probs.shape[0] != len(draft) or target_probs.shape != (len(draft) + 1, draft_probs.shape[1]):
        raise ValueError("Need k draft and k+1 target distributions, aligned to prefix positions")
    _probabilities(draft_probs)
    _probabilities(target_probs)
    out = []
    for i, token in enumerate(draft.tolist()):
        if not 0 <= token < draft_probs.shape[1] or draft_probs[i, token] <= 0:
            raise ValueError("Draft token must have positive proposal probability")
        acceptance = (target_probs[i, token] / draft_probs[i, token]).clamp(max=1)
        if torch.rand((), generator=generator, device=target_probs.device) < acceptance:
            out.append(token)
            if token == eos_token_id:
                return VerificationResult(tuple(out), i + 1, None)
            continue
        residual = (target_probs[i] - draft_probs[i]).clamp_min(0)
        if residual.sum() <= 0:
            raise FloatingPointError("Rejected token without positive correction mass")
        replacement = int(torch.multinomial(residual / residual.sum(), 1, generator=generator))
        out.append(replacement)
        return VerificationResult(tuple(out), i, i)
    out.append(int(torch.multinomial(target_probs[-1], 1, generator=generator)))
    return VerificationResult(tuple(out), len(draft), None)


def same_model_generate(model, input_ids: Tensor, *, max_new_tokens: int = 32,
                        draft_tokens: int = 3, draft_depth: int = 1,
                        target_depth: int = 4, sampled: bool = False,
                        generator: torch.Generator | None = None,
                        eos_token_id: int | None = None) -> tuple[Tensor, dict]:
    """Executable same-model shallow/deep reference with full-prefix replay.

    No second model, no hybrid cache corruption. This deliberately recomputes full
    prefixes and is a correctness oracle, not a fast MTP implementation. Sampling
    targets raw temperature-1 distributions, without other generate processors.
    """
    from .recurrence import QwenNRDFWrapper
    if model.training:
        raise ValueError("Speculative decoding requires eval mode")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("Reference supports one nonempty unpadded sequence")
    if min(max_new_tokens, draft_tokens, draft_depth) < 1 or target_depth < draft_depth:
        raise ValueError("Invalid speculative budgets")
    wrappers = [m for m in model.modules() if isinstance(m, QwenNRDFWrapper)]
    if not wrappers or any(target_depth > m.config.max_loops for m in wrappers):
        raise ValueError("Install recurrent grafts supporting the target depth")
    saved = [(m, m.loops, m.adaptive) for m in wrappers]
    prefix = input_ids.clone()
    accepted = drafted = target_calls = draft_calls = 0
    def set_depth(depth):
        for m in wrappers:
            m.loops, m.adaptive = depth, False
    try:
        with torch.inference_mode():
            while prefix.shape[-1] - input_ids.shape[-1] < max_new_tokens:
                room = max_new_tokens - (prefix.shape[-1] - input_ids.shape[-1])
                k = min(draft_tokens, room)
                set_depth(draft_depth)
                trial = prefix
                proposals, q = [], []
                for _ in range(k):
                    logits = model(input_ids=trial, use_cache=False).logits[0, -1].float()
                    draft_calls += 1
                    probs = logits.softmax(-1)
                    token = int(torch.multinomial(probs, 1, generator=generator)) if sampled else int(logits.argmax())
                    proposals.append(token)
                    q.append(probs)
                    trial = torch.cat((trial, trial.new_tensor([[token]])), -1)
                    if token == eos_token_id:
                        break
                set_depth(target_depth)
                logits = model(input_ids=trial, use_cache=False).logits[0, prefix.shape[-1]-1:].float()
                target_calls += 1
                ids = input_ids.new_tensor(proposals)
                verification = (verify_sampled(ids, torch.stack(q), logits.softmax(-1),
                    generator=generator, eos_token_id=eos_token_id) if sampled else
                    verify_greedy(ids, logits, eos_token_id=eos_token_id))
                accepted += verification.accepted
                drafted += len(proposals)
                emitted = list(verification.tokens)[:room]
                if eos_token_id is not None and eos_token_id in emitted:
                    emitted = emitted[:emitted.index(eos_token_id) + 1]
                prefix = torch.cat((prefix, prefix.new_tensor([emitted])), -1)
                if emitted[-1] == eos_token_id:
                    break
    finally:
        for m, depth, adaptive in saved:
            m.loops, m.adaptive = depth, adaptive
    return prefix, {"accepted_draft_tokens": accepted, "drafted_tokens": drafted,
        "target_calls": target_calls, "draft_calls": draft_calls,
        "algorithm": "same-weights shallow/deep full-prefix correctness reference",
        "mtp_kernel": False, "speedup_claim": False}
