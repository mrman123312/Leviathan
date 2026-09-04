"""Function-preserving architecture transplantation primitives.

This module intentionally contains no tensor framework. It defines the control-plane rules
that training code must satisfy before a Leviathan architecture candidate can advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TransplantPhase(str, Enum):
    BASELINE_AND_FINGERPRINT = "baseline_and_fingerprint"
    INSERT_INERT_MODULES = "insert_inert_modules"
    TRAIN_NEW_PARAMETERS_ONLY = "train_new_parameters_only"
    GATE_WARMUP = "gate_warmup"
    SELECTIVE_UNFREEZE = "selective_unfreeze"
    CONTINUED_TRAINING = "continued_pretraining_and_agentic_posttraining"
    EVALUATION = "retention_calibration_safety_evaluation"
    SHADOW = "shadow_evaluation"
    PROMOTED = "candidate_promotion"


PHASE_ORDER = (
    TransplantPhase.BASELINE_AND_FINGERPRINT,
    TransplantPhase.INSERT_INERT_MODULES,
    TransplantPhase.TRAIN_NEW_PARAMETERS_ONLY,
    TransplantPhase.GATE_WARMUP,
    TransplantPhase.SELECTIVE_UNFREEZE,
    TransplantPhase.CONTINUED_TRAINING,
    TransplantPhase.EVALUATION,
    TransplantPhase.SHADOW,
    TransplantPhase.PROMOTED,
)


@dataclass(frozen=True, slots=True)
class EvaluationGate:
    capability_pass: bool = False
    retention_pass: bool = False
    calibration_pass: bool = False
    safety_pass: bool = False
    adversarial_pass: bool = False
    efficiency_pass: bool = False
    rollback_verified: bool = False

    @property
    def promotion_ready(self) -> bool:
        return all(
            (
                self.capability_pass,
                self.retention_pass,
                self.calibration_pass,
                self.safety_pass,
                self.adversarial_pass,
                self.efficiency_pass,
                self.rollback_verified,
            )
        )


@dataclass(slots=True)
class TransplantRun:
    substrate_id: str
    phase: TransplantPhase = TransplantPhase.BASELINE_AND_FINGERPRINT
    new_module_gate: float = 0.0
    core_frozen: bool = True
    rollback_artifact: str | None = None
    evaluation: EvaluationGate = field(default_factory=EvaluationGate)

    def __post_init__(self) -> None:
        self._validate_gate(self.new_module_gate)
        if self.phase in {
            TransplantPhase.INSERT_INERT_MODULES,
            TransplantPhase.TRAIN_NEW_PARAMETERS_ONLY,
        } and self.new_module_gate != 0.0:
            raise ValueError("new modules must remain inert during insertion/initial isolated training")

    @staticmethod
    def _validate_gate(value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("module gate must be in [0, 1]")

    def set_gate(self, value: float) -> None:
        self._validate_gate(value)
        if self.phase.value in {
            TransplantPhase.BASELINE_AND_FINGERPRINT.value,
            TransplantPhase.INSERT_INERT_MODULES.value,
            TransplantPhase.TRAIN_NEW_PARAMETERS_ONLY.value,
        } and value != 0.0:
            raise RuntimeError("cannot activate new path before gate-warmup phase")
        self.new_module_gate = value

    def advance(self) -> TransplantPhase:
        index = PHASE_ORDER.index(self.phase)
        if self.phase is TransplantPhase.PROMOTED:
            return self.phase

        next_phase = PHASE_ORDER[index + 1]

        if next_phase is TransplantPhase.GATE_WARMUP and not self.core_frozen:
            raise RuntimeError("core must remain frozen through initial new-parameter training")

        if next_phase is TransplantPhase.PROMOTED:
            if not self.evaluation.promotion_ready:
                raise RuntimeError("candidate cannot be promoted until every evaluation gate passes")
            if not self.rollback_artifact:
                raise RuntimeError("candidate cannot be promoted without a rollback artifact")

        self.phase = next_phase
        return self.phase

    def permit_selective_unfreeze(self) -> None:
        if self.phase not in {
            TransplantPhase.SELECTIVE_UNFREEZE,
            TransplantPhase.CONTINUED_TRAINING,
            TransplantPhase.EVALUATION,
            TransplantPhase.SHADOW,
        }:
            raise RuntimeError("core unfreezing is not allowed in the current phase")
        self.core_frozen = False
