"""Executable DeepSeek V4 MoP-0 reference path.

Two V4 expert layouts are supported:

* DeepSeek's reference inference layout: a ModuleList of routed experts exposing
  w1/w2/w3.
* Hugging Face Transformers V4 layout: one packed DeepseekV4Experts module with
  3-D gate_up_proj and down_proj expert tensors.

The optimized Leviathan kernel does not exist yet. This module therefore favors
correctness over speed and reconstructs every originally selected expert from all
of its contiguous intermediate-channel tiles. It is a prompt-level parity oracle,
not a serving-speed implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

try:  # Keep the core package importable without the heavyweight inference extras.
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised on minimal installs.
    torch = None
    nn = None
    F = None


_BaseModule = nn.Module if nn is not None else object


@dataclass(frozen=True, slots=True)
class PatchReport:
    moe_modules: int
    wrapped_experts: int
    already_wrapped: int
    unavailable_experts: int
    tile_width: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptParityResult:
    max_abs_logit_diff: float
    mean_abs_logit_diff: float
    rms_logit_diff: float
    relative_l2_diff: float
    last_token_argmax_match: bool
    baseline_last_token_id: int
    mop0_last_token_id: int
    wrapped_experts: int
    moe_modules: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MoP0ExpertWrapper(_BaseModule):
    """Reference wrapper for DeepSeek's per-expert w1/w2/w3 layout."""

    def __init__(self, expert: Any, *, tile_width: int = 128) -> None:
        if nn is None:
            raise RuntimeError("MoP-0 reference execution requires PyTorch")
        super().__init__()
        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        for name in ("w1", "w2", "w3"):
            if not hasattr(expert, name):
                raise TypeError(f"expert is missing required projection {name}")
        self.expert = expert
        self.tile_width = int(tile_width)

    @property
    def swiglu_limit(self) -> float:
        return float(getattr(self.expert, "swiglu_limit", 0.0) or 0.0)

    def forward(self, x: Any, weights: Any | None = None) -> Any:
        dtype = x.dtype
        gate = self.expert.w1(x).float()
        up = self.expert.w3(x).float()

        limit = self.swiglu_limit
        if limit > 0:
            up = torch.clamp(up, min=-limit, max=limit)
            gate = torch.clamp(gate, max=limit)

        activated = F.silu(gate) * up
        if weights is not None:
            activated = weights * activated

        intermediate_size = int(activated.shape[-1])
        if intermediate_size % self.tile_width:
            raise ValueError(
                f"expert intermediate size {intermediate_size} is not divisible by "
                f"tile_width={self.tile_width}"
            )

        output = None
        # The donor's custom Linear may own quantization details that cannot safely be
        # sliced yet. A sparse full-width input lets the unchanged w2 execute each tile.
        for start in range(0, intermediate_size, self.tile_width):
            stop = start + self.tile_width
            tile_input = torch.zeros_like(activated)
            tile_input[..., start:stop] = activated[..., start:stop]
            contribution = self.expert.w2(tile_input.to(dtype))
            output = contribution if output is None else output + contribution

        if output is None:
            raise RuntimeError("MoP-0 expert produced no tiles")
        return output


class MoP0PackedExpertsWrapper(_BaseModule):
    """Reference wrapper for Transformers' packed DeepseekV4Experts layout.

    Hugging Face stores all routed experts as:

      gate_up_proj: [experts, 2 * intermediate, hidden]
      down_proj:    [experts, hidden, intermediate]

    The router is left untouched. For every routed token/expert pair we reproduce the
    normal gate/up activation once, then evaluate down_proj in contiguous channel
    slices and sum the slices before applying the original route weight.
    """

    def __init__(self, expert_bank: Any, *, tile_width: int = 128) -> None:
        if nn is None:
            raise RuntimeError("MoP-0 reference execution requires PyTorch")
        super().__init__()
        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        for name in ("gate_up_proj", "down_proj"):
            if not hasattr(expert_bank, name):
                raise TypeError(f"packed expert bank is missing {name}")
        self.expert_bank = expert_bank
        self.tile_width = int(tile_width)

    @property
    def num_experts(self) -> int:
        declared = getattr(self.expert_bank, "num_experts", None)
        if declared is not None:
            return int(declared)
        return int(self.expert_bank.gate_up_proj.shape[0])

    @property
    def intermediate_size(self) -> int:
        declared = getattr(self.expert_bank, "intermediate_dim", None)
        if declared is not None:
            return int(declared)
        return int(self.expert_bank.down_proj.shape[-1])

    def _activate(self, gate_up: Any) -> Any:
        apply_gate = getattr(self.expert_bank, "_apply_gate", None)
        if callable(apply_gate):
            return apply_gate(gate_up)

        gate, up = gate_up.chunk(2, dim=-1)
        limit = float(getattr(self.expert_bank, "limit", 0.0) or 0.0)
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        act_fn = getattr(self.expert_bank, "act_fn", F.silu)
        return act_fn(gate) * up

    def forward(self, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
        intermediate_size = self.intermediate_size
        if intermediate_size % self.tile_width:
            raise ValueError(
                f"packed expert intermediate size {intermediate_size} is not divisible by "
                f"tile_width={self.tile_width}"
            )

        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_index_tensor in hit:
            expert_index = int(expert_index_tensor[0].item())
            if expert_index >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_index])

            gate_up = F.linear(
                hidden_states[token_idx],
                self.expert_bank.gate_up_proj[expert_index],
            )
            activated = self._activate(gate_up)

            expert_output = None
            for start in range(0, intermediate_size, self.tile_width):
                stop = start + self.tile_width
                contribution = F.linear(
                    activated[..., start:stop],
                    self.expert_bank.down_proj[expert_index, :, start:stop],
                )
                expert_output = (
                    contribution if expert_output is None else expert_output + contribution
                )

            expert_output = expert_output * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, expert_output.to(final.dtype))
        return final


def _looks_like_routed_moe(module: Any) -> bool:
    return getattr(module, "experts", None) is not None and hasattr(module, "gate")


def _is_packed_expert_bank(experts: Any) -> bool:
    return hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")


def install_mop0_reference(model: Any, *, tile_width: int = 128) -> PatchReport:
    """Patch routed experts in-place while deliberately leaving shared experts alone."""
    if nn is None:
        raise RuntimeError(
            "MoP-0 reference execution requires PyTorch; install Leviathan with "
            "`python -m pip install -e '.[inference]'`"
        )

    moe_modules = 0
    wrapped = 0
    already = 0
    unavailable = 0

    # Freeze the traversal before replacing modules so newly inserted wrappers do not
    # alter which parent modules this pass sees.
    modules = tuple(model.modules())
    for module in modules:
        if not _looks_like_routed_moe(module):
            continue
        experts = module.experts

        if isinstance(experts, MoP0PackedExpertsWrapper):
            moe_modules += 1
            already += experts.num_experts
            continue

        if _is_packed_expert_bank(experts):
            moe_modules += 1
            packed = MoP0PackedExpertsWrapper(experts, tile_width=tile_width)
            wrapped += packed.num_experts
            module.experts = packed
            continue

        if not hasattr(experts, "__len__"):
            continue

        candidate_count = sum(1 for expert in experts if expert is not None)
        if candidate_count == 0:
            continue
        moe_modules += 1

        for index in range(len(experts)):
            expert = experts[index]
            if expert is None:
                unavailable += 1
                continue
            if isinstance(expert, MoP0ExpertWrapper):
                already += 1
                continue
            if not all(hasattr(expert, name) for name in ("w1", "w2", "w3")):
                continue
            experts[index] = MoP0ExpertWrapper(expert, tile_width=tile_width)
            wrapped += 1

    if wrapped == 0 and already == 0:
        raise RuntimeError(
            "no supported DeepSeek V4 routed experts were found. Expected either "
            "Transformers packed gate_up_proj/down_proj experts or DeepSeek reference "
            "experts exposing w1/w2/w3."
        )

    return PatchReport(
        moe_modules=moe_modules,
        wrapped_experts=wrapped,
        already_wrapped=already,
        unavailable_experts=unavailable,
        tile_width=tile_width,
    )


def restore_original_experts(model: Any) -> int:
    """Undo either reference layout wrapper and return the expert count restored."""
    if nn is None:
        return 0
    restored = 0
    modules = tuple(model.modules())
    for module in modules:
        if not _looks_like_routed_moe(module):
            continue
        experts = module.experts
        if isinstance(experts, MoP0PackedExpertsWrapper):
            restored += experts.num_experts
            module.experts = experts.expert_bank
            continue
        if not hasattr(experts, "__len__"):
            continue
        for index in range(len(experts)):
            expert = experts[index]
            if isinstance(expert, MoP0ExpertWrapper):
                experts[index] = expert.expert
                restored += 1
    return restored


def _extract_logits(output: Any) -> Any:
    if torch is not None and torch.is_tensor(output):
        return output
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    raise TypeError("model output does not expose logits")


def _forward_model(model: Any, model_inputs: Mapping[str, Any]) -> Any:
    # Transformers accepts keyword tensors. DeepSeek's standalone reference Transformer
    # accepts input_ids directly. Support both without maintaining separate parity code.
    try:
        return model(**dict(model_inputs))
    except TypeError as first_error:
        if set(model_inputs) == {"input_ids"}:
            try:
                return model(model_inputs["input_ids"])
            except TypeError:
                pass
        raise first_error


def compare_prompt_logits(
    model: Any,
    model_inputs: Mapping[str, Any],
    *,
    tile_width: int = 128,
) -> PromptParityResult:
    """Run one prompt through the donor and MoP-0 reference and compare logits.

    The original experts are restored before returning, even if the MoP run fails.
    """
    if torch is None:
        raise RuntimeError("prompt parity requires PyTorch")

    with torch.inference_mode():
        baseline = _extract_logits(_forward_model(model, model_inputs)).detach().float()

    report = install_mop0_reference(model, tile_width=tile_width)
    try:
        with torch.inference_mode():
            mop0 = _extract_logits(_forward_model(model, model_inputs)).detach().float()
    finally:
        restore_original_experts(model)

    if baseline.shape != mop0.shape:
        raise RuntimeError(
            f"baseline/MoP-0 logit shapes differ: {tuple(baseline.shape)} vs "
            f"{tuple(mop0.shape)}"
        )

    delta = mop0 - baseline
    abs_delta = delta.abs()
    baseline_norm = torch.linalg.vector_norm(baseline)
    delta_norm = torch.linalg.vector_norm(delta)
    relative = float(delta_norm / baseline_norm) if float(baseline_norm) else math.inf

    baseline_last = baseline.reshape(-1, baseline.shape[-1])[-1]
    mop_last = mop0.reshape(-1, mop0.shape[-1])[-1]
    baseline_id = int(torch.argmax(baseline_last).item())
    mop_id = int(torch.argmax(mop_last).item())

    return PromptParityResult(
        max_abs_logit_diff=float(abs_delta.max().item()),
        mean_abs_logit_diff=float(abs_delta.mean().item()),
        rms_logit_diff=float(torch.sqrt(torch.mean(delta.square())).item()),
        relative_l2_diff=relative,
        last_token_argmax_match=baseline_id == mop_id,
        baseline_last_token_id=baseline_id,
        mop0_last_token_id=mop_id,
        wrapped_experts=report.wrapped_experts,
        moe_modules=report.moe_modules,
    )
