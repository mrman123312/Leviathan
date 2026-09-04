"""Bounded reference runtime for Leviathan's Cognitive Parameter Cells.

This module is intentionally backend-agnostic.  A cell can wrap a neural parameter
bundle, an adapter, a retrieval operator, or a deterministic research double.  The
runtime makes the *coordination hypothesis* executable without claiming that a Python
object is itself a new neural substrate.

Cells may propose cognition and recruit peers.  They cannot execute actions, verify
their own output, write durable memory, or alter governance.  Those authorities stay
with :class:`leviathan.agent.LeviathanAgent` and its injected external ports.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from math import ceil, isfinite, log
from typing import Any, Protocol, runtime_checkable

from .types import CognitiveMode, MetaState


def _unit_interval(name: str, value: float) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


def _non_negative(name: str, value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True, slots=True)
class MetaSnapshot:
    """Immutable controller state exposed to cells."""

    task_type: str
    goal: str
    success_probability: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    stakes: float
    risk_budget: float
    compute_budget: float
    latency_budget: float
    available_verifiers: tuple[str, ...]
    available_tools: tuple[str, ...]
    candidate_skills: tuple[str, ...]
    world_model_confidence: float
    branching_factor_estimate: float
    expected_information_gain: float
    hardware_load: float
    recent_failures: int

    @classmethod
    def from_state(cls, state: MetaState, *, immutable_goal: str) -> MetaSnapshot:
        """Copy a mutable ``MetaState`` while restoring the externally held goal."""

        return cls(
            task_type=state.task_type,
            goal=immutable_goal,
            success_probability=state.success_probability,
            epistemic_uncertainty=state.epistemic_uncertainty,
            aleatoric_uncertainty=state.aleatoric_uncertainty,
            stakes=state.stakes,
            risk_budget=state.risk_budget,
            compute_budget=state.compute_budget,
            latency_budget=state.latency_budget,
            available_verifiers=tuple(state.available_verifiers),
            available_tools=tuple(state.available_tools),
            candidate_skills=tuple(state.candidate_skills),
            world_model_confidence=state.world_model_confidence,
            branching_factor_estimate=state.branching_factor_estimate,
            expected_information_gain=state.expected_information_gain,
            hardware_load=state.hardware_load,
            recent_failures=state.recent_failures,
        )


@dataclass(frozen=True, slots=True)
class CognitiveCandidate:
    """One concrete state transition, response, experiment, tool call, or action."""

    id: str
    mode: CognitiveMode
    payload: Any
    expected_observation: Any | None = None
    preconditions: tuple[str, ...] = ()
    risk: float = 0.0
    reversible: bool = True
    authorization_class: str = "internal"
    verifier: str | None = None
    expected_success_gain: float = 0.0
    information_gain: float = 0.0
    transfer_value: float = 0.0
    compute_cost: float = 0.0
    latency_cost: float = 0.0
    irreversibility_cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("candidate id must not be empty")
        if not self.authorization_class.strip():
            raise ValueError("authorization_class must not be empty")
        _unit_interval("risk", self.risk)
        _unit_interval("expected_success_gain", self.expected_success_gain)
        _unit_interval("information_gain", self.information_gain)
        _unit_interval("transfer_value", self.transfer_value)
        _non_negative("compute_cost", self.compute_cost)
        _non_negative("latency_cost", self.latency_cost)
        _non_negative("irreversibility_cost", self.irreversibility_cost)


@dataclass(frozen=True, slots=True)
class CandidateSupport:
    candidate: CognitiveCandidate
    weight: float
    share: float
    supporters: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConsensusSnapshot:
    round_index: int
    ranking: tuple[CandidateSupport, ...]
    leader_id: str | None
    confidence: float
    disagreement: float


@dataclass(frozen=True, slots=True)
class CellContext:
    """Read-only cognitive state supplied to a cell for one discussion round."""

    agent_id: str
    goal_id: str
    meta: MetaSnapshot
    observation_id: str
    observation: Any
    routing_keys: frozenset[str] = frozenset()
    evidence_refs: tuple[str, ...] = ()
    round_index: int = 0
    prior_consensus: ConsensusSnapshot | None = None
    allowed_modes: frozenset[CognitiveMode] = field(
        default_factory=lambda: frozenset(CognitiveMode)
    )


@dataclass(frozen=True, slots=True)
class CellProposal:
    cell_id: str
    candidate: CognitiveCandidate
    confidence: float
    request_cell_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("cell_id must not be empty")
        _unit_interval("proposal confidence", self.confidence)


@runtime_checkable
class CognitiveParameterCell(Protocol):
    """Behavioral interface for a stateful, composable parameter cell."""

    cell_id: str
    reliability: float

    def activation(self, context: CellContext) -> float:
        """Return task relevance in ``[0, 1]``."""

    def propose(self, context: CellContext) -> CellProposal | None:
        """Produce an independent round-zero proposal."""

    def revise(
        self,
        context: CellContext,
        consensus: ConsensusSnapshot,
    ) -> CellProposal | None:
        """Revise after seeing the aggregate market, never private peer state."""


@dataclass(slots=True)
class ScriptedCell:
    """Small deterministic cell for demos, baselines, and orchestration tests.

    It is deliberately named ``ScriptedCell`` so results from it cannot be confused
    with evidence that a neural Mixture-of-Parameters substrate has been trained.
    """

    cell_id: str
    candidate: CognitiveCandidate
    keys: frozenset[str] = frozenset()
    reliability: float = 0.80
    confidence: float = 0.80
    request_cell_ids: tuple[str, ...] = ()
    revision_candidate: CognitiveCandidate | None = None

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("cell id must not be empty")
        _unit_interval("scripted cell reliability", self.reliability)
        _unit_interval("scripted cell confidence", self.confidence)

    def activation(self, context: CellContext) -> float:
        if not self.keys:
            return 0.50
        overlap = len(self.keys & context.routing_keys)
        return min(1.0, 0.10 + overlap / len(self.keys))

    def propose(self, context: CellContext) -> CellProposal:
        return CellProposal(
            cell_id=self.cell_id,
            candidate=self.candidate,
            confidence=self.confidence,
            request_cell_ids=self.request_cell_ids,
            evidence_refs=context.evidence_refs,
        )

    def revise(
        self,
        context: CellContext,
        consensus: ConsensusSnapshot,
    ) -> CellProposal:
        return CellProposal(
            cell_id=self.cell_id,
            candidate=self.revision_candidate or self.candidate,
            confidence=self.confidence,
            request_cell_ids=self.request_cell_ids,
            evidence_refs=context.evidence_refs,
        )


class DeliberationStatus(str, Enum):
    CONVERGED = "converged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROPOSAL = "no_proposal"


class CandidateIdCollision(ValueError):
    """Raised when one logical candidate ID names incompatible state transitions."""


@dataclass(frozen=True, slots=True)
class CellFailure:
    cell_id: str
    round_index: int
    error_type: str


@dataclass(frozen=True, slots=True)
class DeliberationTrace:
    status: DeliberationStatus
    decision: CognitiveCandidate | None
    confidence: float
    disagreement: float
    rounds: int
    cell_calls: int
    active_cell_ids: tuple[str, ...]
    coalition_cell_ids: tuple[str, ...]
    snapshots: tuple[ConsensusSnapshot, ...]
    proposals: tuple[CellProposal, ...]
    failures: tuple[CellFailure, ...]
    unresolved_requests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EcologyConfig:
    """Hard limits for one bounded internal discussion."""

    initial_cells: int = 3
    max_active_cells: int = 16
    max_rounds: int = 5
    max_cell_calls: int = 64
    min_rounds: int = 1
    min_supporting_cells: int = 2
    consensus_threshold: float = 0.62
    disagreement_threshold: float = 0.35
    expansion_factor: float = 2.0
    max_requests_per_proposal: int = 4
    compiled_coalition_min_successes: int = 2

    def __post_init__(self) -> None:
        if self.initial_cells < 1:
            raise ValueError("initial_cells must be at least 1")
        if self.max_active_cells < self.initial_cells:
            raise ValueError("max_active_cells must be >= initial_cells")
        if self.max_rounds < 1 or not 1 <= self.min_rounds <= self.max_rounds:
            raise ValueError("round limits are inconsistent")
        if self.max_cell_calls < self.initial_cells:
            raise ValueError("max_cell_calls must cover the initial cells")
        if self.min_supporting_cells < 1:
            raise ValueError("min_supporting_cells must be at least 1")
        _unit_interval("consensus_threshold", self.consensus_threshold)
        _unit_interval("disagreement_threshold", self.disagreement_threshold)
        if not isfinite(self.expansion_factor) or self.expansion_factor <= 1.0:
            raise ValueError("expansion_factor must be finite and greater than 1")
        if self.max_requests_per_proposal < 0:
            raise ValueError("max_requests_per_proposal must be non-negative")
        if self.compiled_coalition_min_successes < 1:
            raise ValueError("compiled_coalition_min_successes must be at least 1")


@dataclass(slots=True)
class CoalitionRecord:
    cell_ids: tuple[str, ...]
    successes: int = 0
    failures: int = 0
    trust_sum: float = 0.0

    @property
    def mean_trust(self) -> float:
        total = self.successes + self.failures
        return self.trust_sum / total if total else 0.0

    @property
    def score(self) -> float:
        # A Beta(1, 1) prior prevents one lucky outcome from dominating routing.
        reliability = (self.successes + 1) / (self.successes + self.failures + 2)
        return reliability * self.mean_trust


class ParameterEcology:
    """Recruit, discuss, escalate, and converge over a sparse cell reservoir.

    The reference implementation scans the registered cells.  A neural or large-scale
    backend should replace that scan with a learned hierarchical index while retaining
    the same trace, budget, and governance contracts.
    """

    def __init__(
        self,
        cells: Iterable[CognitiveParameterCell],
        *,
        config: EcologyConfig | None = None,
    ) -> None:
        self.config = config if config is not None else EcologyConfig()
        self._cells: dict[str, CognitiveParameterCell] = {}
        self._coalitions: dict[tuple[frozenset[str], str], CoalitionRecord] = {}
        for cell in cells:
            if not cell.cell_id.strip():
                raise ValueError("cell id must not be empty")
            if cell.cell_id in self._cells:
                raise ValueError(f"duplicate cell id: {cell.cell_id}")
            _unit_interval(f"reliability for {cell.cell_id}", cell.reliability)
            self._cells[cell.cell_id] = cell
        if not self._cells:
            raise ValueError("at least one cell is required")

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._cells))

    def deliberate(self, context: CellContext) -> DeliberationTrace:
        activations, failures = self._activation_scores(context)
        active = self._seed_cells(context, activations)
        disabled = {failure.cell_id for failure in failures}
        active = [cell_id for cell_id in active if cell_id not in disabled]

        proposals: list[CellProposal] = []
        snapshots: list[ConsensusSnapshot] = []
        unresolved: set[str] = set()
        calls = 0
        previous: ConsensusSnapshot | None = None

        for round_index in range(self.config.max_rounds):
            round_context = replace(
                context,
                round_index=round_index,
                prior_consensus=previous,
            )
            round_proposals: list[CellProposal] = []
            for cell_id in tuple(active):
                if cell_id in disabled or calls >= self.config.max_cell_calls:
                    continue
                cell = self._cells[cell_id]
                calls += 1
                try:
                    proposal = (
                        cell.propose(round_context)
                        if previous is None
                        else cell.revise(round_context, previous)
                    )
                    if proposal is None:
                        continue
                    self._validate_proposal(cell_id, proposal, context.allowed_modes)
                    round_proposals.append(proposal)
                except Exception as exc:  # noqa: BLE001 - isolate one cell from the run
                    failures.append(CellFailure(cell_id, round_index, type(exc).__name__))
                    disabled.add(cell_id)

            proposals.extend(round_proposals)
            if not round_proposals:
                return self._trace(
                    status=(
                        DeliberationStatus.BUDGET_EXHAUSTED
                        if calls >= self.config.max_cell_calls
                        else DeliberationStatus.NO_PROPOSAL
                    ),
                    snapshots=snapshots,
                    proposals=proposals,
                    failures=failures,
                    unresolved=unresolved,
                    active=active,
                    calls=calls,
                )

            try:
                previous = self._aggregate(round_index, round_proposals, activations)
            except CandidateIdCollision:
                failures.extend(
                    CellFailure(proposal.cell_id, round_index, "CandidateIdCollision")
                    for proposal in round_proposals
                )
                return self._trace(
                    status=DeliberationStatus.NO_PROPOSAL,
                    snapshots=snapshots,
                    proposals=proposals,
                    failures=failures,
                    unresolved=unresolved,
                    active=active,
                    calls=calls,
                )
            snapshots.append(previous)
            leader = previous.ranking[0] if previous.ranking else None
            converged = (
                round_index + 1 >= self.config.min_rounds
                and leader is not None
                and len(leader.supporters) >= self.config.min_supporting_cells
                and previous.confidence >= self.config.consensus_threshold
                and previous.disagreement <= self.config.disagreement_threshold
            )
            if converged:
                return self._trace(
                    status=DeliberationStatus.CONVERGED,
                    snapshots=snapshots,
                    proposals=proposals,
                    failures=failures,
                    unresolved=unresolved,
                    active=active,
                    calls=calls,
                )

            requested: list[str] = []
            for proposal in round_proposals:
                for request in proposal.request_cell_ids[: self.config.max_requests_per_proposal]:
                    if request not in self._cells:
                        unresolved.add(request)
                    elif request not in requested:
                        requested.append(request)
            active = self._expand(active, disabled, requested, activations)

            if calls >= self.config.max_cell_calls:
                break

        return self._trace(
            status=DeliberationStatus.BUDGET_EXHAUSTED,
            snapshots=snapshots,
            proposals=proposals,
            failures=failures,
            unresolved=unresolved,
            active=active,
            calls=calls,
        )

    def record_verified_outcome(
        self,
        *,
        routing_keys: frozenset[str],
        trace: DeliberationTrace,
        passed: bool,
        trust: float,
    ) -> None:
        """Record externally verified coalition performance as procedural routing.

        This method does not decide whether evidence is independent enough.  The agent
        owns that governance decision and calls this method only after its verification
        gate.  No parameters or core policies are changed here.
        """

        _unit_interval("coalition trust", trust)
        if trace.decision is None or not trace.coalition_cell_ids or not routing_keys:
            return
        key = (routing_keys, trace.decision.id)
        record = self._coalitions.get(key)
        if record is None:
            record = CoalitionRecord(cell_ids=trace.coalition_cell_ids)
            self._coalitions[key] = record
        if passed:
            record.successes += 1
        else:
            record.failures += 1
        record.trust_sum += trust

    def compiled_coalition(
        self,
        routing_keys: frozenset[str],
    ) -> tuple[str, ...]:
        eligible = [
            record
            for (keys, _), record in self._coalitions.items()
            if keys == routing_keys
            and record.successes >= self.config.compiled_coalition_min_successes
            and record.successes > record.failures
        ]
        if not eligible:
            return ()
        return max(eligible, key=lambda record: (record.score, record.cell_ids)).cell_ids

    def _activation_scores(
        self,
        context: CellContext,
    ) -> tuple[dict[str, float], list[CellFailure]]:
        scores: dict[str, float] = {}
        failures: list[CellFailure] = []
        for cell_id, cell in self._cells.items():
            try:
                relevance = _unit_interval(
                    f"activation for {cell_id}",
                    cell.activation(context),
                )
                scores[cell_id] = relevance * cell.reliability
            except Exception as exc:  # noqa: BLE001 - isolate one cell from the run
                failures.append(CellFailure(cell_id, -1, type(exc).__name__))
        return scores, failures

    def _seed_cells(
        self,
        context: CellContext,
        activations: dict[str, float],
    ) -> list[str]:
        seeds = list(self.compiled_coalition(context.routing_keys))
        ordered = sorted(activations, key=lambda item: (-activations[item], item))
        for cell_id in ordered:
            if cell_id not in seeds:
                seeds.append(cell_id)
            if len(seeds) >= min(self.config.initial_cells, len(ordered)):
                break
        return seeds[: self.config.max_active_cells]

    def _expand(
        self,
        active: list[str],
        disabled: set[str],
        requested: list[str],
        activations: dict[str, float],
    ) -> list[str]:
        next_active = [cell_id for cell_id in active if cell_id not in disabled]
        target = min(
            self.config.max_active_cells,
            len(self._cells),
            max(len(next_active) + 1, ceil(len(next_active) * self.config.expansion_factor)),
        )
        ordered = requested + sorted(
            activations,
            key=lambda item: (-activations[item], item),
        )
        for cell_id in ordered:
            if cell_id not in disabled and cell_id not in next_active:
                next_active.append(cell_id)
            if len(next_active) >= target:
                break
        return next_active

    @staticmethod
    def _validate_proposal(
        expected_cell_id: str,
        proposal: CellProposal,
        allowed_modes: frozenset[CognitiveMode],
    ) -> None:
        if proposal.cell_id != expected_cell_id:
            raise ValueError("a cell may not submit a proposal under another cell's id")
        if proposal.candidate.mode not in allowed_modes:
            raise ValueError(f"mode is not allowed in this cycle: {proposal.candidate.mode.value}")

    def _aggregate(
        self,
        round_index: int,
        proposals: list[CellProposal],
        activations: dict[str, float],
    ) -> ConsensusSnapshot:
        candidates: dict[str, CognitiveCandidate] = {}
        candidate_signatures: dict[str, tuple[Any, ...]] = {}
        weights: defaultdict[str, float] = defaultdict(float)
        confidences: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
        supporters: defaultdict[str, list[str]] = defaultdict(list)

        for proposal in proposals:
            candidate = proposal.candidate
            signature = self._candidate_signature(candidate)
            prior_signature = candidate_signatures.get(candidate.id)
            if prior_signature is not None and prior_signature != signature:
                raise CandidateIdCollision(candidate.id)
            candidates[candidate.id] = candidate
            candidate_signatures[candidate.id] = signature
            cell = self._cells[proposal.cell_id]
            weight = activations.get(proposal.cell_id, 0.0) * proposal.confidence
            # Reliability appears once in activation.  Reusing it here would square the prior.
            weights[candidate.id] += weight
            confidences[candidate.id].append((proposal.confidence, max(weight, 1e-12)))
            supporters[candidate.id].append(cell.cell_id)

        total = sum(weights.values())
        if total <= 0.0:
            shares = {candidate_id: 1.0 / len(candidates) for candidate_id in candidates}
        else:
            shares = {candidate_id: weight / total for candidate_id, weight in weights.items()}

        ranking = []
        for candidate_id, candidate in candidates.items():
            ranking.append(
                CandidateSupport(
                    candidate=candidate,
                    weight=weights[candidate_id],
                    share=shares[candidate_id],
                    supporters=tuple(sorted(supporters[candidate_id])),
                )
            )
        ranking.sort(key=lambda item: (-item.weight, item.candidate.id))

        if total <= 0.0:
            disagreement = 1.0
        elif len(shares) <= 1:
            disagreement = 0.0
        else:
            entropy = -sum(share * log(share) for share in shares.values() if share > 0.0)
            disagreement = entropy / log(len(shares))

        leader = ranking[0] if ranking else None
        leader_confidence = 0.0
        if leader is not None and total > 0.0:
            leader_values = confidences[leader.candidate.id]
            leader_weight = sum(weight for _, weight in leader_values)
            mean = (
                sum(confidence * weight for confidence, weight in leader_values) / leader_weight
                if leader_weight > 0.0
                else 0.0
            )
            leader_confidence = leader.share * mean

        return ConsensusSnapshot(
            round_index=round_index,
            ranking=tuple(ranking),
            leader_id=leader.candidate.id if leader else None,
            confidence=leader_confidence,
            disagreement=disagreement,
        )

    @staticmethod
    def _candidate_signature(candidate: CognitiveCandidate) -> tuple[Any, ...]:
        """Fail-closed identity check that never invokes arbitrary payload equality."""

        return (
            candidate.mode.value,
            repr(candidate.payload),
            repr(candidate.expected_observation),
            candidate.preconditions,
            candidate.risk,
            candidate.reversible,
            candidate.authorization_class,
            candidate.verifier,
            candidate.expected_success_gain,
            candidate.information_gain,
            candidate.transfer_value,
            candidate.compute_cost,
            candidate.latency_cost,
            candidate.irreversibility_cost,
        )

    @staticmethod
    def _trace(
        *,
        status: DeliberationStatus,
        snapshots: list[ConsensusSnapshot],
        proposals: list[CellProposal],
        failures: list[CellFailure],
        unresolved: set[str],
        active: list[str],
        calls: int,
    ) -> DeliberationTrace:
        latest = snapshots[-1] if snapshots else None
        leader = latest.ranking[0] if latest and latest.ranking else None
        return DeliberationTrace(
            status=status,
            decision=leader.candidate if leader else None,
            confidence=latest.confidence if latest else 0.0,
            disagreement=latest.disagreement if latest else 1.0,
            rounds=len(snapshots),
            cell_calls=calls,
            active_cell_ids=tuple(active),
            coalition_cell_ids=leader.supporters if leader else (),
            snapshots=tuple(snapshots),
            proposals=tuple(proposals),
            failures=tuple(failures),
            unresolved_requests=tuple(sorted(unresolved)),
        )
