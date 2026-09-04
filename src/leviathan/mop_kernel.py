"""Adapter from the one tensorized MoP output to the agent decision contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log

import numpy as np

from .kernel import (
    CognitiveCandidate,
    CognitiveContext,
    InferenceStatus,
    InferenceTrace,
    KernelManifest,
)
from .mop import UnifiedMoP


@dataclass(frozen=True, slots=True)
class VectorObservation:
    """Numerical observation accepted by the current proof-of-mechanism model."""

    features: tuple[float, ...]
    context: tuple[float, ...]


@dataclass(slots=True)
class VectorMoPKernel:
    """Use one ``UnifiedMoP`` as the agent's complete decision model.

    Candidate templates are an output vocabulary, analogous to token IDs.  They contain
    no model logic or state.  The single model emits one logit vector and this adapter
    deterministically decodes it.
    """

    model: UnifiedMoP
    candidates: tuple[CognitiveCandidate, ...]
    active_bases: int | None = None
    confidence_threshold: float = 0.0
    manifest: KernelManifest = field(default_factory=KernelManifest)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("at least one candidate token is required")
        if len(self.candidates) != self.model.config.output_dim:
            raise ValueError("candidate count must equal the model output dimension")
        if len({candidate.id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate ids must be unique")
        if not isfinite(self.confidence_threshold) or not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        # Validate the requested sparse width once at construction.
        self.model.active_parameter_count(self.active_bases)

    @property
    def model_id(self) -> str:
        return f"{self.model.model_id}:decision"

    def infer(self, context: CognitiveContext) -> InferenceTrace:
        observation = context.observation
        if not isinstance(observation, VectorObservation):
            raise TypeError("VectorMoPKernel requires a VectorObservation")
        features = np.asarray([observation.features], dtype=np.float64)
        route_context = np.asarray([observation.context], dtype=np.float64)
        forward = self.model.forward(
            features,
            route_context,
            active_bases=self.active_bases,
        )
        logits = forward.output[0]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        output_entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
        uncertainty = output_entropy / log(len(probabilities)) if len(probabilities) > 1 else 0.0
        route_gates = np.clip(forward.gates[0], 1e-12, 1.0)
        route_entropy = float(-np.sum(route_gates * np.log(route_gates)))

        if confidence < self.confidence_threshold:
            return InferenceTrace(
                status=InferenceStatus.NO_DECISION,
                decision=None,
                confidence=confidence,
                uncertainty=uncertainty,
                refinement_steps=0,
                forward_passes=1,
                active_parameters=self.model.active_parameter_count(self.active_bases),
                total_parameters=self.model.parameter_count,
                route_entropy=route_entropy,
                reason="model confidence is below the decision threshold",
            )
        candidate = self.candidates[index]
        if candidate.mode not in context.allowed_modes:
            return InferenceTrace(
                status=InferenceStatus.NO_DECISION,
                decision=None,
                confidence=confidence,
                uncertainty=uncertainty,
                refinement_steps=0,
                forward_passes=1,
                active_parameters=self.model.active_parameter_count(self.active_bases),
                total_parameters=self.model.parameter_count,
                route_entropy=route_entropy,
                reason="the model selected a disallowed cognitive mode",
            )
        return InferenceTrace(
            status=InferenceStatus.DECIDED,
            decision=candidate,
            confidence=confidence,
            uncertainty=uncertainty,
            refinement_steps=0,
            forward_passes=1,
            active_parameters=self.model.active_parameter_count(self.active_bases),
            total_parameters=self.model.parameter_count,
            route_entropy=route_entropy,
        )
