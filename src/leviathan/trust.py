"""Trust and promotion helpers for the Leviathan scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from .types import LearningCandidate, ProvenanceKind, Verification


PROVENANCE_PRIOR: dict[ProvenanceKind, float] = {
    ProvenanceKind.REAL_OBSERVATION: 0.95,
    ProvenanceKind.FORMAL_RESULT: 0.99,
    ProvenanceKind.DETERMINISTIC_EXECUTION: 0.97,
    ProvenanceKind.TRUSTED_MEASUREMENT: 0.92,
    ProvenanceKind.EXTERNAL_SOURCE: 0.75,
    ProvenanceKind.INDEPENDENT_MODEL: 0.65,
    ProvenanceKind.LEARNED_VERIFIER: 0.55,
    ProvenanceKind.SIMULATION: 0.45,
    ProvenanceKind.SELF_INFERENCE: 0.35,
    ProvenanceKind.SELF_EVALUATION: 0.20,
}


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    semantic_memory: float = 0.50
    procedural_memory: float = 0.65
    plastic_parameters: float = 0.80
    core_parameters: float = 0.93


def verification_trust(verification: Verification) -> float:
    """Combine declared confidence, independence and provenance prior.

    This is a research baseline only. Real trust calibration must be empirical and
    domain-specific.
    """

    prior = PROVENANCE_PRIOR[verification.provenance.kind]
    return max(
        0.0,
        min(
            1.0,
            prior
            * verification.confidence
            * (0.5 + 0.5 * verification.independence_score),
        ),
    )


def candidate_trust(candidate: LearningCandidate) -> float:
    """Conservative geometric-like aggregation of learning evidence."""

    components = (
        candidate.truth_quality,
        candidate.provenance_quality,
        candidate.novelty,
        candidate.utility,
        candidate.consistency,
        candidate.transfer_value,
        candidate.verifier_reliability,
    )

    # A simple product intentionally makes one very weak component matter.
    # Exponent rescales the value into a more interpretable 0..1 range.
    product = 1.0
    for value in components:
        product *= max(0.0, min(1.0, value))
    return product ** (1.0 / len(components))


def may_promote(
    candidate: LearningCandidate,
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> bool:
    score = candidate_trust(candidate)
    target_threshold = {
        "semantic_memory": thresholds.semantic_memory,
        "procedural_memory": thresholds.procedural_memory,
        "plastic_parameters": thresholds.plastic_parameters,
        "core_parameters": thresholds.core_parameters,
    }.get(candidate.target)

    if target_threshold is None:
        raise ValueError(f"Unknown learning target: {candidate.target}")

    return score >= target_threshold


def requires_external_governance(target: str) -> bool:
    """Core parameter promotion always crosses the learner/governor boundary."""

    return target == "core_parameters"
