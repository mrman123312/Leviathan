"""DeepSeek V4 -> Leviathan Mixture-of-Parameters planning primitives.

This module is deliberately tensor-framework agnostic. It defines the exact channel
partition used by the first Leviathan MoP migration and validates that a local
DeepSeek-V4-Pro-Base config still matches the architecture we designed against.

The first migration (MoP-0) is function preserving. A routed SwiGLU expert with
intermediate vector z can be partitioned into channel tiles because the down
projection is linear:

    W_down @ z == sum_j W_down[:, S_j] @ z[S_j]

For the packed DeepSeek gate/up projection, tile S_j therefore owns:
- gate rows S_j
- up rows intermediate_size + S_j
- down columns S_j

All tiles belonging to an originally selected expert inherit that expert's router
weight. Selecting all tiles exactly reconstructs the original expert computation.
No expert averaging is involved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping, Sequence


DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "spec" / "deepseek-v4-mop.toml"


@dataclass(frozen=True, slots=True)
class DeepSeekV4Architecture:
    model_type: str
    architecture_class: str
    hidden_size: int
    num_hidden_layers: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    num_nextn_predict_layers: int
    hc_mult: int
    vocab_size: int
    expert_dtype: str
    weight_block_size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ParameterTile:
    """Logical slices that reconstruct one contiguous expert intermediate block."""

    expert_index: int
    tile_index: int
    start: int
    stop: int
    intermediate_size: int

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def gate_rows(self) -> tuple[int, int]:
        return (self.start, self.stop)

    @property
    def up_rows(self) -> tuple[int, int]:
        return (
            self.intermediate_size + self.start,
            self.intermediate_size + self.stop,
        )

    @property
    def down_columns(self) -> tuple[int, int]:
        return (self.start, self.stop)

    def as_manifest_dict(self) -> dict[str, Any]:
        return {
            "expert_index": self.expert_index,
            "tile_index": self.tile_index,
            "gate_rows": list(self.gate_rows),
            "up_rows": list(self.up_rows),
            "down_columns": list(self.down_columns),
        }


@dataclass(frozen=True, slots=True)
class DeepSeekV4MoPPlan:
    architecture: DeepSeekV4Architecture
    tile_width: int = 128

    def __post_init__(self) -> None:
        if self.tile_width <= 0:
            raise ValueError("tile_width must be positive")
        if self.architecture.moe_intermediate_size % self.tile_width:
            raise ValueError(
                "tile_width must exactly divide moe_intermediate_size for MoP-0 parity"
            )

    @property
    def tiles_per_expert(self) -> int:
        return self.architecture.moe_intermediate_size // self.tile_width

    @property
    def routed_tiles_per_layer(self) -> int:
        return self.architecture.n_routed_experts * self.tiles_per_expert

    @property
    def initial_active_routed_tiles(self) -> int:
        return self.architecture.num_experts_per_tok * self.tiles_per_expert

    @property
    def routed_tile_fraction_at_mop0(self) -> float:
        return self.initial_active_routed_tiles / self.routed_tiles_per_layer

    @property
    def tile_matches_quantization_block(self) -> bool:
        return self.tile_width in self.architecture.weight_block_size

    def tile(self, expert_index: int, tile_index: int) -> ParameterTile:
        if not 0 <= expert_index < self.architecture.n_routed_experts:
            raise IndexError("expert_index out of range")
        if not 0 <= tile_index < self.tiles_per_expert:
            raise IndexError("tile_index out of range")
        start = tile_index * self.tile_width
        return ParameterTile(
            expert_index=expert_index,
            tile_index=tile_index,
            start=start,
            stop=start + self.tile_width,
            intermediate_size=self.architecture.moe_intermediate_size,
        )

    def tiles_for_expert(self, expert_index: int) -> tuple[ParameterTile, ...]:
        return tuple(self.tile(expert_index, idx) for idx in range(self.tiles_per_expert))

    def expand_expert_route(
        self,
        expert_indices: Sequence[int],
    ) -> tuple[ParameterTile, ...]:
        """Expand an original expert route into the exact MoP-0 tile route."""

        return tuple(
            tile
            for expert_index in expert_indices
            for tile in self.tiles_for_expert(expert_index)
        )


def load_mop_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def architecture_from_spec(path: Path = DEFAULT_SPEC_PATH) -> DeepSeekV4Architecture:
    raw = load_mop_spec(path)
    architecture = raw["architecture"]
    quantization = raw["quantization"]
    return DeepSeekV4Architecture(
        model_type=str(architecture["model_type"]),
        architecture_class=str(architecture["architecture_class"]),
        hidden_size=int(architecture["hidden_size"]),
        num_hidden_layers=int(architecture["num_hidden_layers"]),
        moe_intermediate_size=int(architecture["moe_intermediate_size"]),
        n_routed_experts=int(architecture["n_routed_experts"]),
        n_shared_experts=int(architecture["n_shared_experts"]),
        num_experts_per_tok=int(architecture["num_experts_per_tok"]),
        num_attention_heads=int(architecture["num_attention_heads"]),
        num_key_value_heads=int(architecture["num_key_value_heads"]),
        head_dim=int(architecture["head_dim"]),
        max_position_embeddings=int(architecture["max_position_embeddings"]),
        num_nextn_predict_layers=int(architecture["num_nextn_predict_layers"]),
        hc_mult=int(architecture["hc_mult"]),
        vocab_size=int(architecture["vocab_size"]),
        expert_dtype=str(quantization["expert_dtype"]),
        weight_block_size=tuple(int(value) for value in quantization["weight_block_size"]),
    )


def plan_from_spec(path: Path = DEFAULT_SPEC_PATH) -> DeepSeekV4MoPPlan:
    raw = load_mop_spec(path)
    return DeepSeekV4MoPPlan(
        architecture=architecture_from_spec(path),
        tile_width=int(raw["mop"]["tile_width"]),
    )


def validate_deepseek_v4_config(
    config: Mapping[str, Any],
    *,
    plan: DeepSeekV4MoPPlan | None = None,
) -> tuple[str, ...]:
    """Return incompatibilities between a local checkpoint config and the pinned contract."""

    if plan is None:
        plan = plan_from_spec()
    expected = plan.architecture

    checks: dict[str, Any] = {
        "model_type": expected.model_type,
        "hidden_size": expected.hidden_size,
        "num_hidden_layers": expected.num_hidden_layers,
        "moe_intermediate_size": expected.moe_intermediate_size,
        "n_routed_experts": expected.n_routed_experts,
        "n_shared_experts": expected.n_shared_experts,
        "num_experts_per_tok": expected.num_experts_per_tok,
        "num_attention_heads": expected.num_attention_heads,
        "num_key_value_heads": expected.num_key_value_heads,
        "head_dim": expected.head_dim,
        "max_position_embeddings": expected.max_position_embeddings,
        "num_nextn_predict_layers": expected.num_nextn_predict_layers,
        "hc_mult": expected.hc_mult,
        "vocab_size": expected.vocab_size,
        "expert_dtype": expected.expert_dtype,
    }

    errors: list[str] = []
    architectures = config.get("architectures", [])
    if expected.architecture_class not in architectures:
        errors.append(
            f"architectures expected to contain {expected.architecture_class!r}, "
            f"got {architectures!r}"
        )

    for field, value in checks.items():
        actual = config.get(field)
        if actual != value:
            errors.append(f"{field}: expected {value!r}, got {actual!r}")

    quantization = config.get("quantization_config", {})
    actual_block_size = tuple(quantization.get("weight_block_size", ()))
    if actual_block_size != expected.weight_block_size:
        errors.append(
            "quantization_config.weight_block_size: "
            f"expected {expected.weight_block_size!r}, got {actual_block_size!r}"
        )

    if expected.moe_intermediate_size % plan.tile_width:
        errors.append("MoP tile width no longer divides the expert intermediate size")

    return tuple(errors)


def config_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_transplant_manifest(
    config_path: Path,
    *,
    revision: str,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> dict[str, Any]:
    """Build a small reproducibility manifest without loading the 1.6T checkpoint weights."""

    if not revision or revision in {"main", "master", "latest"}:
        raise ValueError("revision must be an immutable upstream commit, not a moving ref")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = plan_from_spec(spec_path)
    errors = validate_deepseek_v4_config(config, plan=plan)
    if errors:
        raise ValueError("DeepSeek V4 config is incompatible:\n- " + "\n- ".join(errors))

    spec = load_mop_spec(spec_path)
    return {
        "schema_version": "0.1",
        "model": {
            "registry_id": spec["model"]["registry_id"],
            "repo_id": spec["model"]["repo_id"],
            "revision": revision,
            "config_sha256": config_sha256(config_path),
        },
        "architecture": asdict(plan.architecture),
        "mop": {
            "target": spec["mop"]["target"],
            "tile_width": plan.tile_width,
            "tiles_per_expert": plan.tiles_per_expert,
            "routed_tiles_per_layer": plan.routed_tiles_per_layer,
            "initial_active_routed_tiles": plan.initial_active_routed_tiles,
            "routed_tile_fraction_at_mop0": plan.routed_tile_fraction_at_mop0,
            "shared_expert_policy": spec["mop"]["shared_expert_policy"],
            "original_router_policy": spec["mop"]["original_router_policy"],
            "function_preserving_mop0": True,
            "tile_matches_quantization_block": plan.tile_matches_quantization_block,
        },
        "gates": {
            "retention": spec["evaluation"]["retention"],
            "benchmarks": spec["evaluation"]["benchmarks"],
            "performance": spec["evaluation"]["performance"],
        },
    }
