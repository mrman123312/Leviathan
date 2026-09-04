"""Architecture maturity accounting for Leviathan L0-L10.

A design document is not the same thing as an embodied capability. This module
encodes the five gates used by the project:

    specification -> executable -> integrated -> learned -> demonstrated

Each gate can be not_started, partial, or passed. The score is descriptive only;
promotion decisions still belong to the transplant/evaluation system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tomllib
from typing import Any, Mapping


DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "spec" / "architecture-maturity.toml"
GATE_ORDER = ("specification", "executable", "integrated", "learned", "demonstrated")


class GateState(str, Enum):
    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    PASSED = "passed"

    @property
    def score(self) -> float:
        return {
            GateState.NOT_STARTED: 0.0,
            GateState.PARTIAL: 0.5,
            GateState.PASSED: 1.0,
        }[self]


@dataclass(frozen=True, slots=True)
class LayerMaturity:
    layer_id: str
    name: str
    target_fundamental: str
    evidence: str
    gates: Mapping[str, GateState]

    def __post_init__(self) -> None:
        missing = [gate for gate in GATE_ORDER if gate not in self.gates]
        extra = [gate for gate in self.gates if gate not in GATE_ORDER]
        if missing or extra:
            raise ValueError(f"invalid maturity gates: missing={missing}, extra={extra}")

        for index, gate in enumerate(GATE_ORDER):
            state = self.gates[gate]
            if index and state is not GateState.NOT_STARTED:
                previous = self.gates[GATE_ORDER[index - 1]]
                if previous is GateState.NOT_STARTED:
                    raise ValueError(
                        f"{self.layer_id}: {gate} cannot progress before "
                        f"{GATE_ORDER[index - 1]}"
                    )

    @property
    def score(self) -> float:
        return sum(self.gates[gate].score for gate in GATE_ORDER)

    @property
    def fully_demonstrated(self) -> bool:
        return all(self.gates[gate] is GateState.PASSED for gate in GATE_ORDER)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "name": self.name,
            "target_fundamental": self.target_fundamental,
            "evidence": self.evidence,
            "score": self.score,
            "gates": {gate: self.gates[gate].value for gate in GATE_ORDER},
        }


@dataclass(frozen=True, slots=True)
class ArchitectureMaturityPlan:
    layers: tuple[LayerMaturity, ...]
    build_order: tuple[str, ...]
    required_gates: tuple[str, ...] = GATE_ORDER

    def __post_init__(self) -> None:
        ids = [layer.layer_id for layer in self.layers]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate layer ids in maturity plan")
        unknown = [layer_id for layer_id in self.build_order if layer_id not in ids]
        if unknown:
            raise ValueError(f"build_order references unknown layers: {unknown}")
        if tuple(self.required_gates) != GATE_ORDER:
            raise ValueError("maturity gate order must remain specification->demonstrated")

    def layer(self, layer_id: str) -> LayerMaturity:
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        raise KeyError(layer_id)

    @property
    def demonstrated_layers(self) -> tuple[str, ...]:
        return tuple(layer.layer_id for layer in self.layers if layer.fully_demonstrated)


def _parse_layer(layer_id: str, raw: Mapping[str, Any]) -> LayerMaturity:
    gate_raw = raw["gates"]
    gates = {gate: GateState(str(gate_raw[gate])) for gate in GATE_ORDER}
    return LayerMaturity(
        layer_id=layer_id,
        name=str(raw["name"]),
        target_fundamental=str(raw["target_fundamental"]),
        evidence=str(raw.get("evidence", "")),
        gates=gates,
    )


def load_maturity_plan(path: Path = DEFAULT_SPEC_PATH) -> ArchitectureMaturityPlan:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    layer_table = raw["layers"]
    layers = tuple(_parse_layer(layer_id, data) for layer_id, data in layer_table.items())
    return ArchitectureMaturityPlan(
        layers=layers,
        build_order=tuple(str(x) for x in raw["development"]["build_order"]),
        required_gates=tuple(str(x) for x in raw["maturity"]["gates"]),
    )
