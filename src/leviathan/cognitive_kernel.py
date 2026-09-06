"""Executable single-model cognitive architecture kernel for Leviathan.

This module turns the high-level L2-L9 architecture into explicit, inspectable
state transitions without pretending that the learned versions already exist.

The kernel compiles:

    problem
      -> representation plan
      -> cognitive program
      -> dynamic DAG
      -> predictions/hypotheses
      -> evidence update
      -> learning destination
      -> reusable skill candidate

It deliberately owns exactly one semantic model id. Neural execution can later be
bound to the canonical DeepSeek/Leviathan model, but the control state never creates
a committee of language models or independent cognitive agents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import Any, Iterable, Mapping, Sequence

from .types import Belief, MetaState, Provenance, ProvenanceKind, UncertaintyKind


DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "spec" / "cognitive-kernel.toml"


class RepresentationKind(str, Enum):
    TOKEN = "token"
    CONCEPT = "concept"
    ENTITY_EVENT = "entity_event"
    GRAPH = "graph"
    CAUSAL = "causal"
    SYMBOLIC = "symbolic"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    PROCEDURAL = "procedural"


class CognitiveOperator(str, Enum):
    ENCODE = "encode"
    ABSTRACT = "abstract"
    EXPAND = "expand"
    RECALL = "recall"
    BIND = "bind"
    COMPARE = "compare"
    HYPOTHESIZE = "hypothesize"
    PREDICT = "predict"
    SIMULATE = "simulate"
    SEARCH = "search"
    VERIFY = "verify"
    UPDATE_BELIEF = "update_belief"
    PLAN = "plan"
    EXECUTE = "execute"
    ASK = "ask"
    WRITE_MEMORY = "write_memory"
    COMPILE_SKILL = "compile_skill"


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


class LearningDestination(str, Enum):
    IGNORE = "ignore"
    EPISODIC = "episodic_memory"
    SEMANTIC = "semantic_memory"
    PROCEDURAL = "procedural_memory"
    PLASTIC = "plastic_parameters"
    CORE = "core_parameters"


@dataclass(frozen=True, slots=True)
class GoalState:
    objective: str
    constraints: tuple[str, ...] = ()
    success_tests: tuple[str, ...] = ()
    priority: float = 0.5
    risk_limit: float = 0.5
    authorization_class: str = "ordinary"

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("goal objective cannot be empty")
        for name, value in (("priority", self.priority), ("risk_limit", self.risk_limit)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RepresentationPlan:
    primary: RepresentationKind
    auxiliaries: tuple[RepresentationKind, ...]
    preserve_exact_input: bool
    abstraction_budget: int
    rationale: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.abstraction_budget <= 0:
            raise ValueError("abstraction_budget must be positive")
        if self.primary in self.auxiliaries:
            raise ValueError("primary representation cannot be repeated as auxiliary")


@dataclass(frozen=True, slots=True)
class CognitiveInstruction:
    id: str
    operator: CognitiveOperator
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    expected_information_gain: float = 0.0
    expected_cost: float = 0.0
    risk: float = 0.0
    reversible: bool = True
    verifier_required: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("instruction id cannot be empty")
        for name, value in (
            ("expected_information_gain", self.expected_information_gain),
            ("expected_cost", self.expected_cost),
            ("risk", self.risk),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class CognitiveProgram:
    goal: GoalState
    representation: RepresentationPlan
    instructions: tuple[CognitiveInstruction, ...]
    max_steps: int
    max_parallel_width: int

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ValueError("cognitive program cannot be empty")
        if self.max_steps <= 0 or self.max_parallel_width <= 0:
            raise ValueError("program budgets must be positive")
        ids = [instruction.id for instruction in self.instructions]
        if len(ids) != len(set(ids)):
            raise ValueError("instruction ids must be unique")
        known = set(ids)
        for instruction in self.instructions:
            unknown = set(instruction.dependencies) - known
            if unknown:
                raise ValueError(
                    f"{instruction.id} depends on unknown instructions: {sorted(unknown)}"
                )

    @property
    def fingerprint(self) -> str:
        payload = {
            "goal": asdict(self.goal),
            "representation": {
                "primary": self.representation.primary.value,
                "auxiliaries": [item.value for item in self.representation.auxiliaries],
                "preserve_exact_input": self.representation.preserve_exact_input,
                "abstraction_budget": self.representation.abstraction_budget,
                "rationale": self.representation.rationale,
            },
            "instructions": [
                {
                    **asdict(instruction),
                    "operator": instruction.operator.value,
                }
                for instruction in self.instructions
            ],
            "max_steps": self.max_steps,
            "max_parallel_width": self.max_parallel_width,
        }
        encoded = json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(slots=True)
class CognitiveGraphNode:
    instruction: CognitiveInstruction
    status: NodeStatus = NodeStatus.PENDING
    result_ref: str | None = None
    error: str | None = None


@dataclass(slots=True)
class DynamicCognitiveGraph:
    nodes: dict[str, CognitiveGraphNode]
    execution_order: list[str] = field(default_factory=list)

    @classmethod
    def from_program(cls, program: CognitiveProgram) -> "DynamicCognitiveGraph":
        graph = cls(
            nodes={
                instruction.id: CognitiveGraphNode(instruction=instruction)
                for instruction in program.instructions
            }
        )
        graph._validate_acyclic()
        graph.refresh_ready()
        return graph

    def _validate_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("cognitive program contains a dependency cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for parent in self.nodes[node_id].instruction.dependencies:
                visit(parent)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id)

    def refresh_ready(self) -> None:
        changed = True
        while changed:
            changed = False
            for node in self.nodes.values():
                if node.status is not NodeStatus.PENDING:
                    continue
                dependencies = [self.nodes[item].status for item in node.instruction.dependencies]
                if any(status in {NodeStatus.FAILED, NodeStatus.BLOCKED} for status in dependencies):
                    node.status = NodeStatus.BLOCKED
                    changed = True
                elif all(status is NodeStatus.SUCCEEDED for status in dependencies):
                    node.status = NodeStatus.READY
                    changed = True

    def ready(self, *, limit: int | None = None) -> tuple[CognitiveGraphNode, ...]:
        self.refresh_ready()
        candidates = [node for node in self.nodes.values() if node.status is NodeStatus.READY]
        if limit is not None:
            candidates = candidates[:limit]
        return tuple(candidates)

    def start(self, node_id: str) -> None:
        node = self.nodes[node_id]
        self.refresh_ready()
        if node.status is not NodeStatus.READY:
            raise RuntimeError(f"node {node_id} is not ready")
        node.status = NodeStatus.RUNNING
        self.execution_order.append(node_id)

    def finish(self, node_id: str, *, result_ref: str) -> None:
        node = self.nodes[node_id]
        if node.status is not NodeStatus.RUNNING:
            raise RuntimeError(f"node {node_id} is not running")
        node.status = NodeStatus.SUCCEEDED
        node.result_ref = result_ref
        self.refresh_ready()

    def fail(self, node_id: str, *, error: str) -> None:
        node = self.nodes[node_id]
        if node.status is not NodeStatus.RUNNING:
            raise RuntimeError(f"node {node_id} is not running")
        node.status = NodeStatus.FAILED
        node.error = error
        self.refresh_ready()

    @property
    def complete(self) -> bool:
        terminal = {NodeStatus.SUCCEEDED, NodeStatus.FAILED, NodeStatus.BLOCKED}
        return all(node.status in terminal for node in self.nodes.values())


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    statement: str
    confidence: float
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("hypothesis confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Prediction:
    id: str
    hypothesis_id: str | None
    action_id: str | None
    expected_observation: Any
    confidence: float
    horizon: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("prediction confidence must be in [0, 1]")
        if self.horizon <= 0:
            raise ValueError("prediction horizon must be positive")


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    target_id: str
    supports: bool
    strength: float
    independence: float
    provenance: Provenance

    def __post_init__(self) -> None:
        for name, value in (("strength", self.strength), ("independence", self.independence)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class BeliefUpdate:
    belief_id: str
    prior_confidence: float
    posterior_confidence: float
    evidence_id: str
    evidence_effect: float


class EvidenceUpdater:
    """Transparent baseline Bayesian-like update with independence discounting."""

    def update(self, belief: Belief, evidence: Evidence) -> BeliefUpdate:
        prior = min(max(float(belief.confidence), 1e-6), 1.0 - 1e-6)
        effect = evidence.strength * evidence.independence * evidence.provenance.trust_prior
        signed = effect if evidence.supports else -effect

        # Update in log-odds space. This is an explicit baseline, not the final learned
        # belief updater, and preserves source trust rather than allowing summarization
        # to magically increase epistemic status.
        prior_logit = math_logit(prior)
        posterior = sigmoid(prior_logit + 4.0 * signed)
        return BeliefUpdate(
            belief_id=belief.id,
            prior_confidence=float(belief.confidence),
            posterior_confidence=posterior,
            evidence_id=evidence.id,
            evidence_effect=signed,
        )


@dataclass(frozen=True, slots=True)
class LearningRoute:
    destination: LearningDestination
    reason: str
    requires_replay: bool
    requires_shadow_eval: bool
    may_touch_ancestral_weights: bool


class LearningRouter:
    """Route verified experience to the least dangerous useful storage mechanism."""

    def route(
        self,
        *,
        verified: bool,
        truth_quality: float,
        novelty: float,
        transfer_value: float,
        repeated_successes: int,
        rollback_available: bool,
        independent_verification: bool,
    ) -> LearningRoute:
        metrics = (truth_quality, novelty, transfer_value)
        if any(not 0.0 <= value <= 1.0 for value in metrics):
            raise ValueError("learning qualities must be in [0, 1]")
        if repeated_successes < 0:
            raise ValueError("repeated_successes cannot be negative")

        if not verified or truth_quality < 0.55:
            return LearningRoute(
                LearningDestination.EPISODIC,
                "unverified or weak evidence is retained only as an episode",
                requires_replay=False,
                requires_shadow_eval=False,
                may_touch_ancestral_weights=False,
            )

        if novelty < 0.20 and transfer_value < 0.30:
            return LearningRoute(
                LearningDestination.IGNORE,
                "verified but redundant low-transfer experience",
                False,
                False,
                False,
            )

        if repeated_successes >= 8 and transfer_value >= 0.70:
            return LearningRoute(
                LearningDestination.PROCEDURAL,
                "repeated verified transferable trajectory is a skill candidate",
                True,
                False,
                False,
            )

        if truth_quality >= 0.90 and transfer_value >= 0.85 and independent_verification:
            if rollback_available:
                return LearningRoute(
                    LearningDestination.PLASTIC,
                    "high-trust transferable knowledge may enter transactional plastic weights",
                    True,
                    True,
                    False,
                )

        return LearningRoute(
            LearningDestination.SEMANTIC,
            "verified knowledge belongs in semantic memory before parameter consolidation",
            False,
            False,
            False,
        )

    def core_candidate(
        self,
        *,
        independent_verification: bool,
        rollback_available: bool,
        replay_pass: bool,
        calibration_pass: bool,
        safety_pass: bool,
        shadow_pass: bool,
        external_promotion_authority: bool,
    ) -> LearningRoute:
        required = (
            independent_verification,
            rollback_available,
            replay_pass,
            calibration_pass,
            safety_pass,
            shadow_pass,
            external_promotion_authority,
        )
        if not all(required):
            raise RuntimeError("core consolidation candidate failed governance requirements")
        return LearningRoute(
            LearningDestination.CORE,
            "externally promoted verified consolidation candidate",
            True,
            True,
            True,
        )


@dataclass(slots=True)
class SkillCandidate:
    id: str
    program_fingerprint: str
    purpose: str
    source_episode_ids: tuple[str, ...]
    successes: int = 0
    failures: int = 0
    verifier_required: bool = True
    compiled: bool = False

    @property
    def trials(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    def record(self, *, verified_success: bool) -> None:
        if verified_success:
            self.successes += 1
        else:
            self.failures += 1

    def ready_to_compile(self, *, min_trials: int = 8, min_success_rate: float = 0.90) -> bool:
        return self.trials >= min_trials and self.success_rate >= min_success_rate


class CognitiveCompiler:
    """Accumulate repeated verified trajectories into reusable procedure candidates."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillCandidate] = {}
        self._episode_results: dict[tuple[str, str], bool] = {}

    def observe(
        self,
        program: CognitiveProgram,
        *,
        episode_id: str,
        verified_success: bool,
    ) -> SkillCandidate:
        key = program.fingerprint
        evidence_key = (key, episode_id)
        if evidence_key in self._episode_results:
            if self._episode_results[evidence_key] != verified_success:
                raise ValueError("An episode cannot be recounted with a different outcome")
            return self._skills[key]
        self._episode_results[evidence_key] = verified_success
        skill = self._skills.get(key)
        if skill is None:
            skill = SkillCandidate(
                id=f"skill-{key[:12]}",
                program_fingerprint=key,
                purpose=program.goal.objective,
                source_episode_ids=(episode_id,),
            )
            self._skills[key] = skill
        elif episode_id not in skill.source_episode_ids:
            skill.source_episode_ids = (*skill.source_episode_ids, episode_id)
        skill.record(verified_success=verified_success)
        return skill

    def compile(self, fingerprint: str) -> SkillCandidate:
        skill = self._skills[fingerprint]
        if not skill.ready_to_compile():
            raise RuntimeError("skill has not earned compilation")
        skill.compiled = True
        return skill


@dataclass(frozen=True, slots=True)
class CausalAccountabilityRecord:
    goal_id: str
    belief_ids: tuple[str, ...]
    program_fingerprint: str
    action_ids: tuple[str, ...]
    prediction_ids: tuple[str, ...]
    outcome_ref: str
    verification_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    sequence: int
    event_type: str
    module: str
    input_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AppendOnlyEventLog:
    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []

    def append(
        self,
        *,
        event_type: str,
        module: str,
        input_refs: Sequence[str] = (),
        output_refs: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            sequence=len(self._events),
            event_type=event_type,
            module=module,
            input_refs=tuple(input_refs),
            output_refs=tuple(output_refs),
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)


class RepresentationCompiler:
    """Transparent baseline compiler for the representation-control interface."""

    def compile(self, *, task_type: str, problem: str, uncertainty: float) -> RepresentationPlan:
        if not problem.strip():
            raise ValueError("problem cannot be empty")
        if not 0.0 <= uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")

        label = task_type.lower()
        text = problem.lower()
        rationale: list[str] = []

        if any(token in label or token in text for token in ("math", "equation", "proof", "algebra")):
            primary = RepresentationKind.SYMBOLIC
            auxiliaries = (RepresentationKind.CONCEPT,)
            rationale.append("symbolic structure detected")
        elif any(token in label or token in text for token in ("code", "program", "debug", "software")):
            primary = RepresentationKind.PROCEDURAL
            auxiliaries = (RepresentationKind.GRAPH, RepresentationKind.SYMBOLIC)
            rationale.append("procedural/dependency structure detected")
        elif any(token in label or token in text for token in ("cause", "diagnos", "why", "mechanism")):
            primary = RepresentationKind.CAUSAL
            auxiliaries = (RepresentationKind.ENTITY_EVENT, RepresentationKind.TEMPORAL)
            rationale.append("causal diagnosis structure detected")
        elif any(token in label or token in text for token in ("geometry", "spatial", "map", "position")):
            primary = RepresentationKind.SPATIAL
            auxiliaries = (RepresentationKind.GRAPH,)
            rationale.append("spatial relation structure detected")
        else:
            primary = RepresentationKind.CONCEPT
            auxiliaries = (RepresentationKind.ENTITY_EVENT,)
            rationale.append("general semantic abstraction")

        budget = 1 + int(round(3 * uncertainty))
        return RepresentationPlan(
            primary=primary,
            auxiliaries=auxiliaries,
            preserve_exact_input=True,
            abstraction_budget=budget,
            rationale=tuple(rationale),
        )


class CognitiveProgramCompiler:
    """Build an explicit reasoning algorithm from representation and metacognitive state."""

    def compile(
        self,
        *,
        goal: GoalState,
        representation: RepresentationPlan,
        state: MetaState,
    ) -> CognitiveProgram:
        instructions: list[CognitiveInstruction] = []

        def add(
            operator: CognitiveOperator,
            *,
            depends: Sequence[str] = (),
            info: float = 0.0,
            cost: float = 0.1,
            risk: float = 0.0,
            verifier: bool = False,
        ) -> str:
            node_id = f"n{len(instructions):02d}-{operator.value}"
            instructions.append(
                CognitiveInstruction(
                    id=node_id,
                    operator=operator,
                    dependencies=tuple(depends),
                    expected_information_gain=info,
                    expected_cost=cost,
                    risk=risk,
                    verifier_required=verifier,
                )
            )
            return node_id

        encoded = add(CognitiveOperator.ENCODE, cost=0.05)
        abstracted = add(CognitiveOperator.ABSTRACT, depends=(encoded,), cost=0.08)
        recalled = add(CognitiveOperator.RECALL, depends=(abstracted,), info=0.15, cost=0.08)

        predecessor = recalled
        if state.epistemic_uncertainty >= 0.30:
            hypothesis = add(
                CognitiveOperator.HYPOTHESIZE,
                depends=(predecessor,),
                info=0.20,
                cost=0.15,
            )
            prediction = add(
                CognitiveOperator.PREDICT,
                depends=(hypothesis,),
                info=0.15,
                cost=0.12,
            )
            predecessor = prediction

        if state.branching_factor_estimate > 3:
            predecessor = add(
                CognitiveOperator.SEARCH,
                depends=(predecessor,),
                info=0.20,
                cost=0.25,
            )

        if state.world_model_confidence > 0.50 and state.epistemic_uncertainty > 0.40:
            predecessor = add(
                CognitiveOperator.SIMULATE,
                depends=(predecessor,),
                info=0.20,
                cost=0.25,
            )

        plan = add(CognitiveOperator.PLAN, depends=(predecessor,), cost=0.10)
        verify = add(
            CognitiveOperator.VERIFY,
            depends=(plan,),
            info=0.15,
            cost=0.15,
            verifier=True,
        )
        update = add(CognitiveOperator.UPDATE_BELIEF, depends=(verify,), cost=0.08)
        add(CognitiveOperator.COMPILE_SKILL, depends=(update,), cost=0.05, verifier=True)

        max_steps = max(4, min(64, int(round(8 + 40 * state.compute_budget))))
        max_parallel = max(1, min(8, int(round(1 + 7 * (1.0 - state.latency_budget)))))
        return CognitiveProgram(
            goal=goal,
            representation=representation,
            instructions=tuple(instructions),
            max_steps=max_steps,
            max_parallel_width=max_parallel,
        )


@dataclass(slots=True)
class LeviathanCognitiveKernel:
    """One-model architecture state tying L2-L9 reference primitives together."""

    model_id: str
    representation_compiler: RepresentationCompiler = field(default_factory=RepresentationCompiler)
    program_compiler: CognitiveProgramCompiler = field(default_factory=CognitiveProgramCompiler)
    evidence_updater: EvidenceUpdater = field(default_factory=EvidenceUpdater)
    learning_router: LearningRouter = field(default_factory=LearningRouter)
    cognitive_compiler: CognitiveCompiler = field(default_factory=CognitiveCompiler)
    event_log: AppendOnlyEventLog = field(default_factory=AppendOnlyEventLog)

    def __post_init__(self) -> None:
        if not self.model_id or any(separator in self.model_id for separator in (",", ";", "|")):
            raise ValueError("LeviathanCognitiveKernel owns exactly one semantic model id")

    def compile_problem(
        self,
        *,
        problem: str,
        task_type: str,
        goal: GoalState,
        state: MetaState,
    ) -> tuple[CognitiveProgram, DynamicCognitiveGraph]:
        representation = self.representation_compiler.compile(
            task_type=task_type,
            problem=problem,
            uncertainty=state.epistemic_uncertainty,
        )
        program = self.program_compiler.compile(
            goal=goal,
            representation=representation,
            state=state,
        )
        graph = DynamicCognitiveGraph.from_program(program)
        self.event_log.append(
            event_type="program_compiled",
            module="cognitive_kernel",
            output_refs=(program.fingerprint,),
            metadata={
                "model_id": self.model_id,
                "primary_representation": representation.primary.value,
                "instruction_count": len(program.instructions),
            },
        )
        return program, graph


def math_logit(probability: float) -> float:
    import math

    return math.log(probability / (1.0 - probability))


def sigmoid(value: float) -> float:
    import math

    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def load_cognitive_kernel_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)
