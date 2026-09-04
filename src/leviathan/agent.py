"""A single governed Leviathan agent built around a sparse parameter ecology.

``LeviathanAgent`` is the sole owner of identity, goal continuity, action authority,
the event journal, and durable learning.  Internal cells are proposal mechanisms, not
subagents.  This preserves one coherent agent while allowing heterogeneous cognition.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
from math import isfinite
from typing import Any, Protocol, runtime_checkable

from .cells import (
    CellContext,
    CognitiveCandidate,
    DeliberationStatus,
    DeliberationTrace,
    MetaSnapshot,
    ParameterEcology,
)
from .controller import BaselineMetaController
from .trust import verification_trust
from .types import CognitiveMode, MetaState, Provenance, Verification

EXTERNAL_ACTION_MODES = frozenset(
    {
        CognitiveMode.TOOL,
        CognitiveMode.EXPERIMENT,
        CognitiveMode.ACT,
    }
)


def _unit_interval(name: str, value: float) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GoalFrame:
    """Externally supplied goal that lower cognition cannot rewrite."""

    original: str
    success_criteria: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    risk_budget: float = 0.25
    id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.original.strip():
            raise ValueError("goal must not be empty")
        _unit_interval("goal risk_budget", self.risk_budget)
        object.__setattr__(
            self,
            "id",
            _digest(
                {
                    "original": self.original,
                    "success_criteria": self.success_criteria,
                    "constraints": self.constraints,
                    "risk_budget": self.risk_budget,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    """Frozen constitution and operational limits held outside the learner."""

    constitution: tuple[str, ...] = (
        "preserve_original_goal",
        "prediction_before_action",
        "learner_not_governor",
        "generator_not_verifier",
        "no_raw_core_updates",
    )
    max_cycles: int = 32
    max_action_risk: float = 0.35
    allowed_authorization_classes: frozenset[str] = frozenset({"internal", "read", "sandbox"})
    irreversible_authorization_classes: frozenset[str] = frozenset()
    require_prediction_before_action: bool = True
    require_verifier_before_action: bool = True
    minimum_verifier_independence: float = 0.50
    minimum_verification_trust: float = 0.45

    def __post_init__(self) -> None:
        if not self.constitution:
            raise ValueError("constitution must not be empty")
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        _unit_interval("max_action_risk", self.max_action_risk)
        _unit_interval("minimum_verifier_independence", self.minimum_verifier_independence)
        _unit_interval("minimum_verification_trust", self.minimum_verification_trust)
        if not self.allowed_authorization_classes:
            raise ValueError("at least one authorization class is required")

    @property
    def digest(self) -> str:
        return _digest(
            {
                "constitution": self.constitution,
                "max_cycles": self.max_cycles,
                "max_action_risk": self.max_action_risk,
                "allowed_authorization_classes": sorted(self.allowed_authorization_classes),
                "irreversible_authorization_classes": sorted(
                    self.irreversible_authorization_classes
                ),
                "require_prediction_before_action": self.require_prediction_before_action,
                "require_verifier_before_action": self.require_verifier_before_action,
                "minimum_verifier_independence": self.minimum_verifier_independence,
                "minimum_verification_trust": self.minimum_verification_trust,
            }
        )


@dataclass(frozen=True, slots=True)
class AgentObservation:
    id: str
    payload: Any
    provenance: Provenance
    routing_keys: frozenset[str] = frozenset()
    evidence_refs: tuple[str, ...] = ()
    satisfied_preconditions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("observation id must not be empty")


@dataclass(frozen=True, slots=True)
class ActionContract:
    """Prediction and authority record created before an external action."""

    id: str
    agent_id: str
    goal_id: str
    policy_digest: str
    candidate_id: str
    mode: CognitiveMode
    payload: Any
    expected_observation: Any
    preconditions: tuple[str, ...]
    risk: float
    reversible: bool
    authorization_class: str
    verifier: str | None


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    contract_id: str
    result: Any
    observation: Any | None = None
    success: bool | None = None
    terminal: bool = False
    cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("contract_id must not be empty")
        if not isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("outcome cost must be finite and non-negative")


@runtime_checkable
class ActionExecutor(Protocol):
    """Environment/tool port.  It receives a frozen contract, not agent state."""

    def execute(self, contract: ActionContract) -> ActionOutcome: ...


@runtime_checkable
class OutcomeVerifier(Protocol):
    """Independent evidence port kept outside the cell ecology."""

    verifier_id: str

    def verify(
        self,
        contract: ActionContract,
        outcome: ActionOutcome,
    ) -> Iterable[Verification]: ...


class AgentStatus(str, Enum):
    READY = "ready"
    DELIBERATING = "deliberating"
    DECIDED = "decided"
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"
    BLOCKED = "blocked"
    NO_DECISION = "no_decision"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    sequence: int
    cycle: int
    kind: str
    goal_id: str
    refs: tuple[str, ...] = ()
    detail: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Episode:
    id: str
    cycle: int
    goal_id: str
    observation_id: str
    trace: DeliberationTrace
    status: AgentStatus
    contract: ActionContract | None = None
    outcome: ActionOutcome | None = None
    verifications: tuple[Verification, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    agent_id: str
    goal: GoalFrame
    policy_digest: str
    cycle: int
    status: AgentStatus
    observation_ids: tuple[str, ...]
    event_count: int
    episode_count: int


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    status: AgentStatus
    trace: DeliberationTrace
    episode: Episode
    candidate: CognitiveCandidate | None
    contract: ActionContract | None = None
    outcome: ActionOutcome | None = None
    verifications: tuple[Verification, ...] = ()
    reason: str | None = None


class LeviathanAgent:
    """One stateful agent with bounded recursive cognition and guarded effects."""

    def __init__(
        self,
        *,
        agent_id: str,
        goal: GoalFrame,
        ecology: ParameterEcology,
        policy: AgentPolicy | None = None,
        controller: BaselineMetaController | None = None,
        executor: ActionExecutor | None = None,
        verifier: OutcomeVerifier | None = None,
    ) -> None:
        if not agent_id.strip():
            raise ValueError("agent_id must not be empty")
        self._agent_id = agent_id
        self._goal = goal
        self._goal_digest = goal.id
        self._policy = policy if policy is not None else AgentPolicy()
        self._policy_digest = self._policy.digest
        self._ecology = ecology
        self._controller = controller or BaselineMetaController()
        self._executor = executor
        self._verifier = verifier
        self._cycle = 0
        self._status = AgentStatus.READY
        self._observation_ids: list[str] = []
        self._events: list[AgentEvent] = []
        self._episodes: list[Episode] = []

    @property
    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            agent_id=self._agent_id,
            goal=self._goal,
            policy_digest=self._policy_digest,
            cycle=self._cycle,
            status=self._status,
            observation_ids=tuple(self._observation_ids),
            event_count=len(self._events),
            episode_count=len(self._episodes),
        )

    @property
    def events(self) -> tuple[AgentEvent, ...]:
        return tuple(self._events)

    @property
    def episodes(self) -> tuple[Episode, ...]:
        return tuple(self._episodes)

    def step(self, observation: AgentObservation, meta_state: MetaState) -> AgentTurnResult:
        """Run one observe-orient-discuss-contract-act-verify cycle.

        A non-converged cell market never falls through to action.  The caller can add
        evidence, change the budget, or explicitly reconfigure the research system and
        then begin another cycle.
        """

        self._assert_integrity()
        if self._cycle >= self._policy.max_cycles:
            return self._blocked_without_cycle(observation, "agent cycle budget exhausted")
        if observation.id in self._observation_ids:
            return self._blocked_without_cycle(observation, "duplicate observation id")

        self._cycle += 1
        self._status = AgentStatus.DELIBERATING
        self._observation_ids.append(observation.id)
        self._event("observation_received", refs=(observation.id,))

        sanitized_meta = replace(meta_state, goal=self._goal.original)
        if meta_state.goal != self._goal.original:
            self._event(
                "goal_restored",
                detail=(("supplied_goal_ignored", meta_state.goal),),
            )

        allowed_modes = set(self._controller.propose_modes(sanitized_meta))
        # These are boundary transitions, not endorsements.  Risk and authority are
        # checked later against the frozen goal and policy.
        allowed_modes.update({CognitiveMode.ACT, CognitiveMode.ASK, CognitiveMode.WAIT_OBSERVE})
        context = CellContext(
            agent_id=self._agent_id,
            goal_id=self._goal.id,
            meta=MetaSnapshot.from_state(sanitized_meta, immutable_goal=self._goal.original),
            observation_id=observation.id,
            observation=observation.payload,
            routing_keys=observation.routing_keys,
            evidence_refs=observation.evidence_refs,
            allowed_modes=frozenset(allowed_modes),
        )
        trace = self._ecology.deliberate(context)
        self._event(
            "deliberation_closed",
            refs=(observation.id,),
            detail=(
                ("status", trace.status.value),
                ("rounds", str(trace.rounds)),
                ("cell_calls", str(trace.cell_calls)),
                ("disagreement", f"{trace.disagreement:.6f}"),
            ),
        )

        if trace.status is not DeliberationStatus.CONVERGED or trace.decision is None:
            reason = (
                "internal deliberation did not converge within its hard limits"
                if trace.decision is not None
                else "internal deliberation produced no admissible proposal"
            )
            return self._close_episode(
                observation=observation,
                trace=trace,
                status=AgentStatus.NO_DECISION,
                reason=reason,
            )

        candidate = trace.decision
        if candidate.mode not in EXTERNAL_ACTION_MODES:
            return self._close_episode(
                observation=observation,
                trace=trace,
                status=AgentStatus.DECIDED,
            )

        blocked_reason = self._action_block_reason(candidate, observation, sanitized_meta)
        if blocked_reason is not None:
            self._event(
                "action_blocked",
                refs=(candidate.id,),
                detail=(("reason", blocked_reason),),
            )
            return self._close_episode(
                observation=observation,
                trace=trace,
                status=AgentStatus.BLOCKED,
                reason=blocked_reason,
            )

        contract = self._make_contract(candidate)
        # This event is deliberately emitted before authorization/execution.
        self._event(
            "prediction_recorded",
            refs=(contract.id, candidate.id),
            detail=(("expected", repr(contract.expected_observation)),),
        )
        self._event("action_authorized", refs=(contract.id,))

        assert self._executor is not None  # guaranteed by _action_block_reason
        try:
            outcome = self._executor.execute(contract)
            if outcome.contract_id != contract.id:
                raise ValueError("executor returned an outcome for a different contract")
        except Exception as exc:  # noqa: BLE001 - an external port must not crash agent state
            reason = f"executor failed: {type(exc).__name__}"
            self._event(
                "action_failed",
                refs=(contract.id,),
                detail=(("error", type(exc).__name__),),
            )
            return self._close_episode(
                observation=observation,
                trace=trace,
                status=AgentStatus.FAILED,
                contract=contract,
                reason=reason,
            )

        self._event("action_executed", refs=(contract.id,))
        verifications, verification_error = self._verify(contract, outcome)
        for verification in verifications:
            self._event(
                "verification_recorded",
                refs=(contract.id, verification.verifier_id),
                detail=(
                    ("passed", str(verification.passed)),
                    ("trust", f"{verification_trust(verification):.6f}"),
                ),
            )

        eligible, passed, trust = self._verification_gate(contract, verifications)
        if eligible:
            verified_success = passed and outcome.success is not False
            self._ecology.record_verified_outcome(
                routing_keys=observation.routing_keys,
                trace=trace,
                passed=verified_success,
                trust=trust,
            )
            status = AgentStatus.VERIFIED if verified_success else AgentStatus.REJECTED
            if verified_success:
                reason = None
            elif outcome.success is False:
                reason = "the executor reported that the contracted action failed"
            else:
                reason = "independent verification rejected the outcome"
        else:
            status = AgentStatus.UNVERIFIED
            reason = verification_error or "no sufficiently independent verification was available"

        self._assert_integrity()
        return self._close_episode(
            observation=observation,
            trace=trace,
            status=status,
            contract=contract,
            outcome=outcome,
            verifications=verifications,
            reason=reason,
        )

    def _action_block_reason(
        self,
        candidate: CognitiveCandidate,
        observation: AgentObservation,
        meta_state: MetaState,
    ) -> str | None:
        risk_limit = min(
            self._policy.max_action_risk,
            self._goal.risk_budget,
            meta_state.risk_budget,
        )
        if candidate.risk > risk_limit:
            return "candidate risk exceeds the active risk budget"
        if candidate.authorization_class not in self._policy.allowed_authorization_classes:
            return "candidate authorization class is not permitted"
        if (
            not candidate.reversible
            and candidate.authorization_class not in self._policy.irreversible_authorization_classes
        ):
            return "irreversible action lacks an explicit authorization class"
        if self._policy.require_prediction_before_action and candidate.expected_observation is None:
            return "external action has no recorded expected observation"
        missing = set(candidate.preconditions) - observation.satisfied_preconditions
        if missing:
            return f"unsatisfied action preconditions: {', '.join(sorted(missing))}"
        if self._executor is None:
            return "no action executor is connected"
        if self._policy.require_verifier_before_action and self._verifier is None:
            return "no independent verifier is connected"
        if candidate.verifier:
            if candidate.verifier not in meta_state.available_verifiers:
                return "candidate's required verifier is unavailable"
            if self._verifier is None or self._verifier.verifier_id != candidate.verifier:
                return "connected verifier does not match the action contract"
        return None

    def _make_contract(self, candidate: CognitiveCandidate) -> ActionContract:
        return ActionContract(
            id=f"{self._agent_id}:{self._cycle}:{candidate.id}",
            agent_id=self._agent_id,
            goal_id=self._goal.id,
            policy_digest=self._policy_digest,
            candidate_id=candidate.id,
            mode=candidate.mode,
            payload=candidate.payload,
            expected_observation=candidate.expected_observation,
            preconditions=candidate.preconditions,
            risk=candidate.risk,
            reversible=candidate.reversible,
            authorization_class=candidate.authorization_class,
            verifier=candidate.verifier,
        )

    def _verify(
        self,
        contract: ActionContract,
        outcome: ActionOutcome,
    ) -> tuple[tuple[Verification, ...], str | None]:
        if self._verifier is None:
            return (), None
        try:
            return tuple(self._verifier.verify(contract, outcome)), None
        except Exception as exc:  # noqa: BLE001 - verification failure becomes evidence state
            self._event(
                "verification_failed",
                refs=(contract.id,),
                detail=(("error", type(exc).__name__),),
            )
            return (), f"verifier failed: {type(exc).__name__}"

    def _verification_gate(
        self,
        contract: ActionContract,
        verifications: tuple[Verification, ...],
    ) -> tuple[bool, bool, float]:
        eligible = [
            verification
            for verification in verifications
            if verification.target_id == contract.id
            and (contract.verifier is None or verification.verifier_id == contract.verifier)
            and verification.independence_score >= self._policy.minimum_verifier_independence
            and verification_trust(verification) >= self._policy.minimum_verification_trust
        ]
        if not eligible:
            return False, False, 0.0
        conclusive = [verification for verification in eligible if verification.passed is not None]
        if not conclusive:
            return False, False, 0.0
        # One strong negative vetoes promotion.  Agreement never multiplies trust because
        # verifier correlations are not yet modeled by this reference scaffold.
        passed = all(verification.passed is True for verification in conclusive)
        relevant = (
            [verification for verification in conclusive if not verification.passed]
            if not passed
            else conclusive
        )
        trust = max(verification_trust(verification) for verification in relevant)
        return True, passed, trust

    def _close_episode(
        self,
        *,
        observation: AgentObservation,
        trace: DeliberationTrace,
        status: AgentStatus,
        contract: ActionContract | None = None,
        outcome: ActionOutcome | None = None,
        verifications: tuple[Verification, ...] = (),
        reason: str | None = None,
    ) -> AgentTurnResult:
        episode = Episode(
            id=f"{self._agent_id}:episode:{self._cycle}",
            cycle=self._cycle,
            goal_id=self._goal.id,
            observation_id=observation.id,
            trace=trace,
            status=status,
            contract=contract,
            outcome=outcome,
            verifications=verifications,
            reason=reason,
        )
        self._episodes.append(episode)
        self._status = status
        self._event(
            "episode_closed",
            refs=(episode.id,),
            detail=(("status", status.value),),
        )
        return AgentTurnResult(
            status=status,
            trace=trace,
            episode=episode,
            candidate=trace.decision,
            contract=contract,
            outcome=outcome,
            verifications=verifications,
            reason=reason,
        )

    def _blocked_without_cycle(
        self,
        observation: AgentObservation,
        reason: str,
    ) -> AgentTurnResult:
        # A synthetic empty trace keeps the result type total without pretending a cycle ran.
        trace = DeliberationTrace(
            status=DeliberationStatus.NO_PROPOSAL,
            decision=None,
            confidence=0.0,
            disagreement=1.0,
            rounds=0,
            cell_calls=0,
            active_cell_ids=(),
            coalition_cell_ids=(),
            snapshots=(),
            proposals=(),
            failures=(),
            unresolved_requests=(),
        )
        self._status = AgentStatus.BLOCKED
        self._event("cycle_blocked", refs=(observation.id,), detail=(("reason", reason),))
        episode = Episode(
            id=f"{self._agent_id}:blocked:{len(self._episodes) + 1}",
            cycle=self._cycle,
            goal_id=self._goal.id,
            observation_id=observation.id,
            trace=trace,
            status=AgentStatus.BLOCKED,
            reason=reason,
        )
        self._episodes.append(episode)
        return AgentTurnResult(
            status=AgentStatus.BLOCKED,
            trace=trace,
            episode=episode,
            candidate=None,
            reason=reason,
        )

    def _event(
        self,
        kind: str,
        *,
        refs: tuple[str, ...] = (),
        detail: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._events.append(
            AgentEvent(
                sequence=len(self._events) + 1,
                cycle=self._cycle,
                kind=kind,
                goal_id=self._goal.id,
                refs=refs,
                detail=detail,
            )
        )

    def _assert_integrity(self) -> None:
        if self._goal.id != self._goal_digest:
            raise RuntimeError("goal integrity check failed")
        if self._policy.digest != self._policy_digest:
            raise RuntimeError("governance integrity check failed")
