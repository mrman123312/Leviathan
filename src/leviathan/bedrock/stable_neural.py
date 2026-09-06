"""Numerically guarded frozen recurrence for real half-precision pretrained models.

This module keeps the original FrozenExecutor as a historical/raw control and adds a
donor-input-bounded recurrence path. It creates no parameters and performs no fit.
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
from .decisions import StopPolicy, PredictionSummary, summarize, compare, initial_stop, stable_stop
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
    prediction_stop: StopPolicy | None = None
    branch_direction: str = "trajectory"
    branch_sign: int = 1
    branch_mix: float = 0.0

    def __post_init__(self):
        if self.branch_direction not in {"trajectory", "causal_context", "orthogonal_context"}:
            raise ValueError("Unknown branch direction")
        if self.branch_sign not in {-1, 1} or not 0 <= self.branch_mix <= 1:
            raise ValueError("Invalid branch sign/mix")
        if self.prediction_stop is not None and not isinstance(self.prediction_stop, StopPolicy):
            raise TypeError("prediction_stop requires a typed StopPolicy")
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


def branch_target(entry: Tensor, current: Tensor, policy: StableFrozenPolicy) -> Tensor:
    """Deterministic, per-position causal alternatives from inherited activations.

    Signed context directions are not asserted to be semantic hypotheses. Previous
    token states contain no future information. No random/new learned vector exists.
    Radius projection is applied later by transport_reentry.
    """
    if policy.branch_direction == "trajectory" or policy.branch_mix == 0:
        return current
    e=entry.float();base=current.float()-e
    previous=torch.cat((e[:,:1],e[:,:-1]),dim=1)
    direction=previous-e
    if policy.branch_direction == "orthogonal_context":
        axis=torch.nn.functional.normalize(base,dim=-1)
        direction=direction-(direction*axis).sum(-1,keepdim=True)*axis
    direction=torch.nn.functional.normalize(direction,dim=-1)*e.norm(dim=-1,keepdim=True)
    # Remain FP32 until the checked projection/cast, avoiding FP16 intermediate overflow.
    return e+(1-policy.branch_mix)*base+policy.branch_mix*policy.branch_sign*direction


class StableFrozenExecutor(FrozenExecutor):
    """Frozen executor with donor-input-bounded re-entry and fail-closed recurrence.

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
            # Historical/raw policies remain available as explicit controls.
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
            if policy.prediction_stop is not None and end != n-1:
                raise ValueError("Prediction stopping requires the final decoder band; no untrained logit lens")
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
                "prediction_head_calls": 0,
                "prediction_probe_positions": 0,
                "prediction_stopping": policy.prediction_stop is not None,
                "prediction_steps": [],
                "branch_direction": policy.branch_direction,
                "branch_sign": policy.branch_sign,
                "attention_rows_physically_compacted": False,
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
                    prediction_previous = None
                    prediction_streak = torch.zeros_like(steps)
                    def probe(hidden, previous=None):
                        norm=getattr(self.decoder,"norm",None)
                        head=getattr(self.model,"lm_head",None)
                        if head is None:
                            raise TypeError("Prediction stopping requires the owning model output head")
                        stop_policy=policy.prediction_stop
                        flat_hidden=hidden.reshape(-1,hidden.shape[-1])
                        snapshots=[];metrics=[]
                        for first in range(0,len(flat_hidden),stop_policy.chunk_positions):
                            last=min(first+stop_policy.chunk_positions,len(flat_hidden))
                            h=flat_hidden[first:last]
                            logits=head(norm(h) if norm is not None else h).float()
                            trace["prediction_head_calls"]+=1
                            trace["prediction_probe_positions"]+=last-first
                            if previous is None:
                                snapshots.append(summarize(logits,stop_policy.topk))
                            else:
                                old=PredictionSummary(**{k:getattr(previous,k)[first:last]
                                    for k in previous.__dataclass_fields__})
                                snap,metric=compare(old,logits,topk=stop_policy.topk)
                                snapshots.append(snap);metrics.append(metric)
                        combined=PredictionSummary(**{k:torch.cat([getattr(a,k) for a in snapshots])
                            for k in PredictionSummary.__dataclass_fields__})
                        signals={k:torch.cat([a[k] for a in metrics]).reshape(steps.shape)
                                 for k in metrics[0]} if metrics else None
                        return combined,signals
                    if policy.prediction_stop is not None:
                        prediction_previous,_=probe(current)
                        initial=initial_stop(prediction_previous,policy.prediction_stop).reshape(steps.shape)
                        active=active & ~initial
                        trace["initial_prediction_halted_positions"]=int(initial.sum())
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
                                        branch_target(band_entry,current,policy),
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
                            if policy.prediction_stop is not None:
                                prediction_now,signals=probe(current,prediction_previous)
                                stable_prediction=stable_stop(signals,delta,policy.prediction_stop)
                                prediction_streak=torch.where(stable_prediction,prediction_streak+1,
                                                              torch.zeros_like(prediction_streak))
                                enough=(steps+1>=policy.prediction_stop.min_passes)
                                predicted_halt=enough & (prediction_streak>=policy.prediction_stop.patience)
                                trace["prediction_steps"].append({
                                    "pass":recurrence_index+1,
                                    "active_positions_before":int(active.sum()),
                                    "halted_positions":int((active & predicted_halt).sum()),
                                    "max_coarse_js":float(signals["coarse_js"].max()),
                                    "max_top_logprob_change":float(signals["max_top_logprob_change"].max()),
                                    "max_entropy_change":float(signals["entropy_change"].max())})
                                prediction_previous=prediction_now
                            else:
                                predicted_halt=torch.zeros_like(active)
                            steps = steps + active.long()
                            stable = torch.where(delta <= policy.halt_delta, stable + 1, torch.zeros_like(stable))
                            if policy.halt_delta > 0:
                                active = active & (stable < policy.halt_patience)
                            active=active & ~predicted_halt
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
