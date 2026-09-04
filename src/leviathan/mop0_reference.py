"""Executable DeepSeek V4 MoP-0 reference path.

The optimized Leviathan kernel does not exist yet. This module therefore favors
correctness over speed: each original routed expert is evaluated as 128-channel
SwiGLU tiles, and every tile contribution is sent through the donor expert's
unchanged w2 projection before the contributions are summed.

Calling the full w2 once per tile is intentionally expensive. It avoids making
assumptions about the donor's FP8 storage format and gives us a prompt-level parity
oracle before building fused tile kernels.
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
    """Reference implementation of one exact routed-expert tile decomposition.

    It preserves the donor's w1, w2, w3 modules and swiglu_limit. The only change is
    that the activated intermediate vector is partitioned into contiguous tiles and
    the final projection is accumulated tile by tile.
    """

    def __init__(self, expert: Any, *, tile_width: int = 128) -> None:
        if nn is None:
            raise RuntimeError(
                "MoP-0 reference execution requires PyTorch; install Leviathan with "
                "the inference extras"
            )
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
        # Deliberately use the original full w2 operation on a sparse/masked vector.
        # That keeps this reference compatible with custom/quantized Linear modules.
        for start in range(0, intermediate_size, self.tile_width):
            stop = start + self.tile_width
            tile_input = torch.zeros_like(activated)
            tile_input[..., start:stop] = activated[..., start:stop]
            contribution = self.expert.w2(tile_input.to(dtype))
            output = contribution if output is None else output + contribution

        if output is None:  # Defensive only; V4 has a non-empty expert dimension.
            raise RuntimeError("MoP-0 expert produced no tiles")
        return output


def _looks_like_routed_moe(module: Any) -> bool:
    experts = getattr(module, "experts", None)
    return experts is not None and hasattr(module, "gate") and hasattr(experts, "__len__")


def install_mop0_reference(model: Any, *, tile_width: int = 128) -> PatchReport:
    """Wrap routed experts in-place while deliberately leaving shared experts alone."""
    if nn is None:
        raise RuntimeError(
            "MoP-0 reference execution requires PyTorch; install Leviathan with "
            "`python -m pip install -e '.[inference]'`"
        )

    moe_modules = 0
    wrapped = 0
    already = 0
    unavailable = 0

    for module in model.modules():
        if not _looks_like_routed_moe(module):
            continue
        experts = module.experts
        candidate_count = 0
        for expert in experts:
            if expert is not None:
                candidate_count += 1
        if candidate_count == 0:
            # On sharded ranks the ModuleList may contain mostly None, but at least
            # one local routed expert should normally exist. Do not count an unrelated
            # empty container as a MoE module.
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
            "no DeepSeek-style routed experts were found. Expected MoE modules with "
            "a gate and an experts container whose experts expose w1/w2/w3."
        )

    return PatchReport(
        moe_modules=moe_modules,
        wrapped_experts=wrapped,
        already_wrapped=already,
        unavailable_experts=unavailable,
        tile_width=tile_width,
    )


def restore_original_experts(model: Any) -> int:
    """Undo the reference wrapper in-place and return the number restored."""
    if nn is None:
        return 0
    restored = 0
    for module in model.modules():
        if not _looks_like_routed_moe(module):
            continue
        experts = module.experts
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
    # Most Transformers models accept keyword tensors. The official DeepSeek reference
    # Transformer accepts input_ids directly. Support both without model-specific forks.
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
