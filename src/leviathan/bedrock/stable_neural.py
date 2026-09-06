"""Numerically guarded frozen recurrence for real half-precision pretrained models.

This module keeps the original FrozenExecutor as a historical/raw control and adds a
manifold-constrained recurrence path. It creates no parameters and performs no fit.
The central repair is simple: do not feed the *exit* of a pretrained layer band
straight back into a much earlier layer. Re-enter near the actual donor band input,
probe the donor band there, and apply only the resulting bounded innovation at the
original band exit.

The guard is architectural safety, not evidence of better language quality.
"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import math

import torch
from torch import Tensor

from .cells import CellPolicy, FrozenCellBank
from .contracts import Meter, stable_hash
from .neural import (
    FastAssociations,
    FrozenExecutor,
    FrozenPolicy,
    _CellCall,
    _hidden,
    _replace_hidden,
    project_delta,
)
from ..consumer.efficiency import CacheScope, ExactDeltaCache


@dataclass(frozen=True)
class StableFrozenPolicy(FrozenPolicy):
    """Frozen recurrence with a donor-input trust region.

    ``reentry_radius`` bounds how far a recurrent probe can move from the *actual
    pretrained band input*. ``relative_radius`` independently bounds the correction
    at the band output. ``pointwise_multiplier`` is an FP16 guard relative to the
    donor input's observed per-position maximum magnitude.
    """

    feedback: str = "transported"
    reentry_radius: float = 0.08
    pointwise_multiplier: float = 2.0
    nonfinite_fallback: bool = True

    def __post_init__(self):
        if self.feedback not in {"transported", "anchored_difference", "repeat"}:
            raise ValueError("Unknown stable feedback rule")
        if self.exact_ffn_cache and self.cells.mode != "off":
            raise ValueError("Exact FFN cache is only enabled for pure stateless donor calls")
        if not 1 <= self.passes <= 16 or self.halt_patience < 1:
            raise ValueError("Invalid bounded recurrence")
        if not 0 <= self.gain <= 1 or not 0 <= self.fast_gain <= 1:
            raise ValueError("Gains must be finite and in [0,1]")
        for name, value, high in (
            ("relative_radius", self.relative_radius, 2.0),
            ("reentry_radius", self.reentry_radius, 1.0),
        ):
            if not math.isfinite(value) or not 0 <= value <= high:
                raise ValueError(f"Invalid {name}")
        if not math.isfinite(self.pointwise_multiplier) or self.pointwise_multiplier < 1:
            raise ValueError("pointwise_multiplier must be finite and >= 1")
        if not math.isfinite(self.halt_delta) or self.halt_delta < 0:
            raise ValueError("Invalid convergence threshold")


def transport_reentry(entry: Tensor, target: Tensor, *, radius: float,
                      pointwise_multiplier: float) -> Tensor:
    """Move toward ``target`` while staying near the donor's observed band input.

    Real-arithmetic L2 property before ordinary rounding:
        ||reentry-entry|| <= radius * ||entry||.
    The pointwise envelope is a second conservative FP16 stability guard. It is not a
    proof that every subsequent matrix product cannot overflow; the executor therefore
    retains an explicit non-finite donor fallback as a second line of defense.
    """
    if entry.shape != target.shape:
        raise ValueError("Re-entry tensors must align")
    if not torch.isfinite(entry).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Cannot construct transport from non-finite state")
    candidate = project_delta(entry, target.float() - entry.float(), radius)
    entry32 = entry.float()
    peak = entry32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    cap = peak * pointwise_multiplier
    guarded = candidate.float().clamp(min=-cap, max=cap).to(entry.dtype)
    if not torch.isfinite(guarded).all():
        raise FloatingPointError("Non-finite transported re-entry")
    return guarded


class StableFrozenExecutor(FrozenExecutor):
    """Frozen executor with manifold-constrained re-entry and fail-closed recurrence.

    Neutral policies still execute the untouched donor path. For transported policies:

      e = original band input
      a = F(e)  (original band output)
      z_r = project_near(e, current_r)
      innovation_r = F(z_r) - a
      current_(r+1) = project_near(a, current_r + gain*innovation_r)

    Thus the reused band is probed close to the distribution it actually saw during
    the donor forward pass. If any probe is non-finite in FP16, recurrence is discarded
    and the exact donor band output ``a`` is returned for that call.
    """

    stability_version = "transport-v1"

    def run(self, input_ids: Tensor, *, policy: FrozenPolicy = StableFrozenPolicy(),
            meter: Meter | None = None, request_id: str = "ephemeral",
            fast: FastAssociations | None = None, **kwargs):
        if not isinstance(policy, StableFrozenPolicy):
            return super().run(input_ids, policy=policy, meter=meter, request_id=request_id,
                               fast=fast, **kwargs)
        meter = meter or Meter()
        if input_ids.ndim != 2 or input_ids.shape[-1] < 1:
            raise ValueError("Nonempty batched token IDs required")
        if any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value")) or kwargs.get("use_cache", False):
            raise ValueError("Stable recurrence forbids external caches; use direct donor for cached production")

        with self._lock, torch.inference_mode():
            if not self.unchanged():
                raise RuntimeError("Frozen parameter tripwire failed")
            n = len(self.decoder.layers)
            start = policy.start if policy.start >= 0 else n + policy.start
            end = policy.end if policy.end >= 0 else n + policy.end
            if not 0 <= start <= end < n:
                raise ValueError("Decoder band is outside model")
            meter.charge("model_calls")
            meter.charge("layer_calls", n)
            trace = {
                "policy": asdict(policy),
                "donor_layer_calls": n,
                "extra_layer_calls": 0,
                "passes_executed": 1,
                "halts_are_correctness_certificates": False,
                "cells": [],
                "no_new_parameters": True,
                "request_id": request_id,
                "stability_version": self.stability_version,
                "route_status": "experimental",
                "nonfinite_replay_fallbacks": 0,
                "reentry_relative_l2_max": [],
            }
            if policy.neutral:
                result = self.model(input_ids=input_ids, use_cache=False, **kwargs)
                trace["neutral_direct_path"] = True
                trace["route_status"] = "donor"
                self.last_trace = trace
                return result

            captures = {}
            band_entry = None
            in_replay = False
            with ExitStack() as stack:
                if policy.cells.mode != "off" or policy.exact_ffn_cache:
                    layer = self.decoder.layers[end]
                    donor = layer.mlp
                    key = (end, policy.cells.width)
                    if key not in self._banks:
                        self._banks[key] = FrozenCellBank(donor, policy.cells.width)
                    cache_key = (request_id, end, stable_hash(asdict(policy.cells)))
                    cache = None
                    if policy.exact_ffn_cache:
                        cache = self._row_caches.setdefault(cache_key, ExactDeltaCache())
                    scope = CacheScope(
                        self.revision, 0, request_id, str(next(self.model.parameters()).dtype),
                        f"FFN:{end}:{cache_key[-1]}",
                    )
                    layer.mlp = _CellCall(donor, self._banks[key], policy.cells, cache, scope, trace["cells"])
                    stack.callback(setattr, layer, "mlp", donor)

                def capture(index):
                    def hook(module, args, kw):
                        nonlocal band_entry
                        if in_replay:
                            return
                        if len(args) > 1:
                            raise TypeError("Decoder positional context not supported; use named masks/positions")
                        if index == start:
                            band_entry = (args[0] if args else kw["hidden_states"]).detach()
                        clean = dict(kw)
                        clean.pop("hidden_states", None)
                        for field in ("past_key_values", "past_key_value"):
                            if clean.get(field) is not None:
                                raise RuntimeError("Unexpected cache in stable frozen recurrence")
                        captures[index] = clean
                    return hook

                for index in range(start, end + 1):
                    handle = self.decoder.layers[index].register_forward_pre_hook(capture(index), with_kwargs=True)
                    stack.callback(handle.remove)

                def loop(module, args, kw, output):
                    nonlocal in_replay
                    if in_replay:
                        return output
                    anchor = _hidden(output)
                    if band_entry is None or not torch.isfinite(anchor).all() or not torch.isfinite(band_entry).all():
                        raise FloatingPointError("Donor band itself produced non-finite state")
                    current = anchor
                    if fast is not None and policy.fast_gain:
                        current = fast.apply(
                            current,
                            scope=(request_id, self.revision, end),
                            radius=policy.relative_radius * policy.fast_gain,
                        )
                    active = torch.ones(current.shape[:-1], dtype=torch.bool, device=current.device)
                    stable = torch.zeros_like(active, dtype=torch.long)
                    steps = torch.ones_like(stable)
                    delta_trace = []
                    in_replay = True
                    try:
                        for recurrence_index in range(1, policy.passes):
                            if policy.gain == 0 or not bool(active.any()):
                                break
                            previous = current
                            try:
                                if policy.feedback == "transported":
                                    transformed = transport_reentry(
                                        band_entry,
                                        current,
                                        radius=policy.reentry_radius,
                                        pointwise_multiplier=policy.pointwise_multiplier,
                                    )
                                    rel = (transformed.float() - band_entry.float()).norm(dim=-1) / band_entry.float().norm(dim=-1).clamp_min(1e-12)
                                    trace["reentry_relative_l2_max"].append(float(rel.max()))
                                elif policy.feedback == "anchored_difference":
                                    transformed = transport_reentry(
                                        band_entry,
                                        band_entry + policy.gain * (current - band_entry),
                                        radius=policy.reentry_radius,
                                        pointwise_multiplier=policy.pointwise_multiplier,
                                    )
                                else:
                                    transformed = current

                                finite = True
                                for index in range(start, end + 1):
                                    meter.charge("layer_calls")
                                    transformed = _hidden(self.decoder.layers[index](transformed, **captures[index]))
                                    trace["extra_layer_calls"] += 1
                                    if not torch.isfinite(transformed).all():
                                        trace["fallback_layer"] = index
                                        finite = False
                                        break
                                if not finite:
                                    raise FloatingPointError("Non-finite state inside reused band")

                                if policy.feedback == "transported":
                                    innovation = transformed.float() - anchor.float()
                                    proposed_delta = (current.float() - anchor.float()) + policy.gain * innovation
                                    proposed = project_delta(anchor, proposed_delta, policy.relative_radius)
                                elif policy.feedback == "anchored_difference":
                                    proposed = project_delta(
                                        anchor,
                                        policy.gain * (transformed.float() - anchor.float()),
                                        policy.relative_radius,
                                    )
                                else:
                                    proposed = project_delta(
                                        anchor,
                                        (current.float() - anchor.float()) + policy.gain * (transformed.float() - current.float()),
                                        policy.relative_radius,
                                    )
                            except FloatingPointError as exc:
                                if not policy.nonfinite_fallback:
                                    raise
                                current = anchor
                                trace["nonfinite_replay_fallbacks"] += 1
                                trace["route_status"] = "donor_fallback_nonfinite"
                                trace["fallback_reason"] = str(exc)
                                active.zero_()
                                break

                            delta = (proposed.float() - previous.float()).norm(dim=-1) / previous.float().norm(dim=-1).clamp_min(1e-12)
                            current = torch.where(active[..., None], proposed, current)
                            steps = steps + active.long()
                            stable = torch.where(delta <= policy.halt_delta, stable + 1, torch.zeros_like(stable))
                            if policy.halt_delta > 0:
                                active = active & (stable < policy.halt_patience)
                            delta_trace.append(float(delta.max()))
                            trace["passes_executed"] += 1
                    finally:
                        in_replay = False
                    trace["position_depths"] = steps.detach().cpu().tolist()
                    trace["max_relative_step_deltas"] = delta_trace
                    return _replace_hidden(output, current)

                handle = self.decoder.layers[end].register_forward_hook(loop, with_kwargs=True)
                stack.callback(handle.remove)
                result = self.model(input_ids=input_ids, use_cache=False, **kwargs)

            if not self.unchanged():
                raise RuntimeError("Donor changed during stable frozen execution")
            self.last_trace = trace
            return result
