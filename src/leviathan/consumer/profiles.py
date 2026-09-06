"""Pinned donor identities. Post-training is never relabelled pretraining."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelProfile:
    id: str
    repo_id: str
    revision: str
    stage: str
    architecture: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    parameter_estimate_b: float
    context_limit: int
    default_context: int = 2048

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("An immutable 40-character upstream revision is required")
        if self.stage not in {"base", "posttrained"}:
            raise ValueError("Unknown training stage")

    def validate_config(self, raw: Mapping[str, Any]) -> None:
        text = raw.get("text_config", raw)
        expected = {"hidden_size": self.hidden_size, "intermediate_size": self.intermediate_size,
                    "num_hidden_layers": self.num_hidden_layers, "vocab_size": self.vocab_size}
        errors = [f"{key}: expected {value}, got {text.get(key)}"
                  for key, value in expected.items() if text.get(key) != value]
        if raw.get("model_type") != self.model_type:
            errors.append("model_type mismatch")
        if self.architecture not in raw.get("architectures", []):
            errors.append("architecture mismatch")
        if text.get("hidden_act") != "silu":
            errors.append("ancestral cell adapter requires SiLU/SwiGLU")
        if self.model_type == "qwen3_5":
            pattern = ["linear_attention"] * 3 + ["full_attention"]
            if text.get("layer_types") != pattern * 16:
                errors.append("hybrid attention schedule mismatch")
        if errors:
            raise ValueError("Checkpoint fingerprint rejected: " + "; ".join(errors))

    def cells_per_ffn(self, width: int = 128) -> int:
        if width <= 0 or self.intermediate_size % width:
            raise ValueError("Cell width must divide the intermediate width")
        return self.intermediate_size // width

    def memory_estimate(self, bits: int = 4, overhead_gib: float = 0.0) -> dict[str, Any]:
        if bits not in {4, 8, 16, 32} or not math.isfinite(overhead_gib) or overhead_gib < 0:
            raise ValueError("Invalid memory assumptions")
        lower = self.parameter_estimate_b * 1e9 * bits / 8 / 2**30
        return {"nominal_weights_gib": lower, "assumed_extra_gib": overhead_gib,
                "estimated_total_gib": lower + overhead_gib, "measured": False,
                "excludes": ["quantization metadata", "unquantized tensors", "KV/DeltaNet cache",
                             "activations", "CUDA context", "workspace", "adapters", "vision tower"]}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CANONICAL_MODEL_ID = "qwen3.8-27b"
CANONICAL_REPO_ID = "Qwen/Qwen3.8-27B"


PROFILES = {
    "qwen27b": ModelProfile(
        "qwen3.8-27b", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "posttrained", "Qwen3_5ForConditionalGeneration", "qwen3_5",
        5120, 17408, 64, 248320, 27.0, 262144),
    "rtx3060": ModelProfile(
        "qwen3-1.7b-base", "Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "base", "Qwen3ForCausalLM", "qwen3", 2048, 6144, 28, 151936, 1.7, 32768),
}


def get_profile(name: str) -> ModelProfile:
    if name in PROFILES:
        return PROFILES[name]
    for profile in PROFILES.values():
        if name in {profile.id, profile.repo_id}:
            return profile
    raise ValueError(f"Unknown profile {name!r}; choose {tuple(PROFILES)}")
