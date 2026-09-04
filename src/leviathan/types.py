"""Core shared types for the Leviathan research scaffold.

The scaffold is deliberately small. It encodes the architectural contracts from the
research documents without pretending that the hard learning/world-model problems
have already been solved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CognitiveMode(str, Enum):
    DIRECT = "direct"
    RECALL = "recall"
    SKILL = "skill"
    REASON = "reason"
    TREE_SEARCH = "tree_search"
    EVOLVE = "evolve"
    SIMULATE = "simulate"
    EXPERIMENT = "experiment"
    TOOL = "tool"
    PARALLELIZE = "parallelize"
    ASK = "ask"
    ACT = "act"
    WAIT_OBSERVE = "wait_observe"


class ProvenanceKind(str, Enum):
    REAL_OBSERVATION = "real_observation"
    FORMAL_RESULT = "formal_result"
    DETERMINISTIC_EXECUTION = "deterministic_execution"
    TRUSTED_MEASUREMENT = "trusted_measurement"
    EXTERNAL_SOURCE = "external_source"
    INDEPENDENT_MODEL = "independent_model"
    LEARNED_VERIFIER = "learned_verifier"
    SIMULATION = "simulation"
    SELF_INFERENCE = "self_inference"
    SELF_EVALUATION = "self_evaluation"


class UncertaintyKind(str, Enum):
    ALEATORIC = "aleatoric"
    EPISTEMIC = "epistemic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Provenance:
    kind: ProvenanceKind
    source_id: str
    trust_prior: float
    source_version: str | None = None


@dataclass(slots=True)
class Belief:
    id: str
    value: Any
    confidence: float
    provenance: Provenance
    uncertainty: UncertaintyKind = UncertaintyKind.UNKNOWN
    proposition: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    contradiction_refs: list[str] = field(default_factory=list)
    causal_parents: list[str] = field(default_factory=list)
    causal_children: list[str] = field(default_factory=list)
    status: str = "inferred"


@dataclass(slots=True)
class MetaState:
    task_type: str
    goal: str
    success_probability: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    stakes: float
    risk_budget: float
    compute_budget: float
    latency_budget: float
    available_verifiers: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()
    candidate_skills: tuple[str, ...] = ()
    world_model_confidence: float = 0.0
    branching_factor_estimate: float = 1.0
    expected_information_gain: float = 0.0
    hardware_load: float = 0.0
    recent_failures: int = 0


@dataclass(slots=True)
class ModeEstimate:
    mode: CognitiveMode
    expected_success_gain: float = 0.0
    information_gain: float = 0.0
    transfer_value: float = 0.0
    compute_cost: float = 0.0
    latency_cost: float = 0.0
    risk_cost: float = 0.0
    irreversibility_cost: float = 0.0
    hardware_cost: float = 0.0
    confidence: float = 0.0


@dataclass(slots=True)
class Verification:
    target_id: str
    verifier_type: str
    passed: bool | None
    confidence: float
    independence_score: float
    provenance: Provenance
    verifier_id: str = ""
    result: Any = None


@dataclass(slots=True)
class LearningCandidate:
    id: str
    target: str
    source_episode_ids: tuple[str, ...]
    truth_quality: float
    provenance_quality: float
    novelty: float
    utility: float
    consistency: float
    transfer_value: float
    verifier_reliability: float
