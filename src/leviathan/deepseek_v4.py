"""DeepSeek V4 full-checkpoint fingerprinting and Mixture-of-Parameters planning.

This module is intentionally tensor-framework independent. It validates that an
experiment is pointed at the complete DeepSeek-V4-Pro-Base architecture and builds
the exact, function-preserving channel-tiling plan used before any independent
Mixture-of-Parameters routing is allowed.

The actual 1.6T checkpoint stays outside Git and is handled by the model-asset
workflow. Training/inference backends can consume the manifest produced here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

CANONICAL_MODEL_ID = "deepseek-v4-pro-base"
CANONICAL_REPO_ID = "deepseek-ai/DeepSeek-V4-Pro-Base"

EXPECTED_ARCHITECTURE = "DeepseekV4ForCausalLM"
EXPECTED_MODEL_TYPE = "deepseek_v4"
EXPECTED_NUM_HIDDEN_LAYERS = 61
EXPECTED_HIDDEN_SIZE = 7168
EXPECTED_MOE_INTERMEDIATE_SIZE = 3072
EXPECTED_ROUTED_EXPERTS = 384
EXPECTED_SHARED_EXPERTS = 1
EXPECTED_EXPERTS_PER_TOKEN = 6
EXPECTED_MAX_POSITION_EMBEDDINGS = 1_048_576
EXPECTED_WEIGHT_SHARDS = 64

_SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")


@dataclass(frozen=True, slots=True)
class DeepSeekV4Fingerprint:
    architecture: str
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    max_position_embeddings: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeepSeekV4Fingerprint":
        architectures = raw.get("architectures", ())
        architecture = architectures[0] if architectures else ""
        fingerprint = cls(
            architecture=str(architecture),
            model_type=str(raw.get("model_type", "")),
            num_hidden_layers=int(raw.get("num_hidden_layers", -1)),
            hidden_size=int(raw.get("hidden_size", -1)),
            moe_intermediate_size=int(raw.get("moe_intermediate_size", -1)),
            n_routed_experts=int(raw.get("n_routed_experts", -1)),
            n_shared_experts=int(raw.get("n_shared_experts", -1)),
            num_experts_per_tok=int(raw.get("num_experts_per_tok", -1)),
            max_position_embeddings=int(raw.get("max_position_embeddings", -1)),
        )
        fingerprint.require_canonical_full_model()
        return fingerprint

    @classmethod
    def from_json(cls, path: Path) -> "DeepSeekV4Fingerprint":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_mapping(raw)

    def require_canonical_full_model(self) -> None:
        expected = {
            "architecture": EXPECTED_ARCHITECTURE,
            "model_type": EXPECTED_MODEL_TYPE,
            "num_hidden_layers": EXPECTED_NUM_HIDDEN_LAYERS,
            "hidden_size": EXPECTED_HIDDEN_SIZE,
            "moe_intermediate_size": EXPECTED_MOE_INTERMEDIATE_SIZE,
            "n_routed_experts": EXPECTED_ROUTED_EXPERTS,
            "n_shared_experts": EXPECTED_SHARED_EXPERTS,
            "num_experts_per_tok": EXPECTED_EXPERTS_PER_TOKEN,
            "max_position_embeddings": EXPECTED_MAX_POSITION_EMBEDDINGS,
        }
        actual = asdict(self)
        mismatches = {
            key: (actual[key], expected_value)
            for key, expected_value in expected.items()
            if actual[key] != expected_value
        }
        if mismatches:
            detail = ", ".join(
                f"{key}={actual_value!r} expected {expected_value!r}"
                for key, (actual_value, expected_value) in mismatches.items()
            )
            raise ValueError(
                "checkpoint config is not the canonical full DeepSeek-V4-Pro-Base: "
                + detail
            )


@dataclass(frozen=True, slots=True)
class MixtureOfParametersPlan:
    """Function-preserving expert-channel decomposition.

    An original routed expert remains exactly reconstructable because each tile owns
    a contiguous subset of the SwiGLU intermediate channels and the down-projection
    contributions are summed. The first route therefore expands every selected
    expert into all of its tiles; independent tile routing is a later trained phase.
    """

    tile_width: int
    tiles_per_expert: int
    routed_tiles_per_layer: int
    baseline_active_routed_tiles_per_token: int
    routed_experts: int
    shared_experts: int
    experts_per_token: int
    layers: int

    @classmethod
    def from_fingerprint(
        cls,
        fingerprint: DeepSeekV4Fingerprint,
        *,
        tile_width: int = 128,
    ) -> "MixtureOfParametersPlan":
        fingerprint.require_canonical_full_model()
        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        if fingerprint.moe_intermediate_size % tile_width:
            raise ValueError(
                "tile_width must exactly divide DeepSeek V4 moe_intermediate_size "
                f"({fingerprint.moe_intermediate_size})"
            )
        tiles_per_expert = fingerprint.moe_intermediate_size // tile_width
        return cls(
            tile_width=tile_width,
            tiles_per_expert=tiles_per_expert,
            routed_tiles_per_layer=fingerprint.n_routed_experts * tiles_per_expert,
            baseline_active_routed_tiles_per_token=(
                fingerprint.num_experts_per_tok * tiles_per_expert
            ),
            routed_experts=fingerprint.n_routed_experts,
            shared_experts=fingerprint.n_shared_experts,
            experts_per_token=fingerprint.num_experts_per_tok,
            layers=fingerprint.num_hidden_layers,
        )

    def exact_tile_route(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        """Expand an expert route into all corresponding tile IDs.

        This is the parity route. It does not change which expert function is
        evaluated; it only makes the expert's intermediate channels explicit.
        """
        ids = tuple(expert_ids)
        if len(ids) != self.experts_per_token:
            raise ValueError(
                f"expected exactly {self.experts_per_token} routed experts, got {len(ids)}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("expert route contains duplicate expert IDs")
        for expert_id in ids:
            if not 0 <= expert_id < self.routed_experts:
                raise ValueError(f"expert id out of range: {expert_id}")

        tile_ids: list[int] = []
        for expert_id in ids:
            start = expert_id * self.tiles_per_expert
            tile_ids.extend(range(start, start + self.tiles_per_expert))
        return tuple(tile_ids)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeepSeekV4Manifest:
    model_id: str
    repo_id: str
    full_checkpoint_verified: bool
    fingerprint: DeepSeekV4Fingerprint
    mop: MixtureOfParametersPlan

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "repo_id": self.repo_id,
            "full_checkpoint_verified": self.full_checkpoint_verified,
            "fingerprint": asdict(self.fingerprint),
            "mixture_of_parameters": self.mop.as_dict(),
            "invariants": {
                "single_cognitive_model": True,
                "function_preserving_initialization": True,
                "independent_tile_routing_enabled_at_initialization": False,
                "original_expert_route_reconstructable": True,
            },
        }


def verify_full_checkpoint_files(model_dir: Path) -> None:
    """Require every canonical V4-Pro-Base shard plus the safetensors index."""
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("missing model.safetensors.index.json")

    seen: set[str] = set()
    for path in model_dir.iterdir():
        match = _SHARD_RE.match(path.name)
        if not match:
            continue
        total = int(match.group(2))
        if total != EXPECTED_WEIGHT_SHARDS:
            raise ValueError(
                f"unexpected shard denominator in {path.name}; "
                f"expected {EXPECTED_WEIGHT_SHARDS}"
            )
        seen.add(path.name)

    expected = {
        f"model-{index:05d}-of-{EXPECTED_WEIGHT_SHARDS:05d}.safetensors"
        for index in range(1, EXPECTED_WEIGHT_SHARDS + 1)
    }
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise FileNotFoundError(
            "full DeepSeek V4 checkpoint not present: "
            f"missing={missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"extra={extra[:5]}{'...' if len(extra) > 5 else ''}"
        )


def build_manifest(
    model_dir: Path,
    *,
    tile_width: int = 128,
    require_weights: bool = True,
) -> DeepSeekV4Manifest:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing DeepSeek V4 config: {config_path}")

    fingerprint = DeepSeekV4Fingerprint.from_json(config_path)
    if require_weights:
        verify_full_checkpoint_files(model_dir)

    plan = MixtureOfParametersPlan.from_fingerprint(
        fingerprint,
        tile_width=tile_width,
    )
    return DeepSeekV4Manifest(
        model_id=CANONICAL_MODEL_ID,
        repo_id=CANONICAL_REPO_ID,
        full_checkpoint_verified=require_weights,
        fingerprint=fingerprint,
        mop=plan,
    )
