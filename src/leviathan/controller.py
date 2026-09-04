"""A minimal metacognitive routing scaffold.

This is not a trained controller. It provides a transparent baseline policy and the
utility function that later learned policies can replace. Keeping a simple baseline is
important: a learned meta-controller should be evaluated against something explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import CognitiveMode, MetaState, ModeEstimate


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    success: float = 1.0
    information: float = 0.35
    transfer: float = 0.15
    compute: float = 0.20
    latency: float = 0.20
    risk: float = 0.60
    irreversibility: float = 0.75
    hardware: float = 0.10


def utility(estimate: ModeEstimate, weights: UtilityWeights = UtilityWeights()) -> float:
    """Score a proposed cognitive mode.

    These weights are placeholders for experiments, not a claim about the correct AGI
    objective. In production-quality research, hard constraints should protect safety
    properties that should not be traded away inside a scalar utility.
    """

    return (
        weights.success * estimate.expected_success_gain
        + weights.information * estimate.information_gain
        + weights.transfer * estimate.transfer_value
        - weights.compute * estimate.compute_cost
        - weights.latency * estimate.latency_cost
        - weights.risk * estimate.risk_cost
        - weights.irreversibility * estimate.irreversibility_cost
        - weights.hardware * estimate.hardware_cost
    )


class BaselineMetaController:
    """Transparent heuristic controller used as a research baseline."""

    def propose_modes(self, state: MetaState) -> list[CognitiveMode]:
        modes: list[CognitiveMode] = []

        # Reuse known structure first.
        if state.candidate_skills:
            modes.append(CognitiveMode.SKILL)

        # Memory is cheap and often useful.
        modes.append(CognitiveMode.RECALL)

        # Exact tools should be preferred when available.
        if state.available_tools:
            modes.append(CognitiveMode.TOOL)

        # Low uncertainty / low stakes permits direct action.
        if state.epistemic_uncertainty < 0.20 and state.stakes < 0.50:
            modes.append(CognitiveMode.DIRECT)

        # Ordinary deliberation remains the default fallback.
        modes.append(CognitiveMode.REASON)

        # Epistemic uncertainty should create evidence-seeking behavior.
        if state.epistemic_uncertainty > 0.45:
            if state.world_model_confidence > 0.55:
                modes.append(CognitiveMode.SIMULATE)
            modes.append(CognitiveMode.EXPERIMENT)

        # Large branching factors favor branch-aware search.
        if state.branching_factor_estimate > 3:
            modes.append(CognitiveMode.TREE_SEARCH)
        if state.branching_factor_estimate > 12 and state.available_verifiers:
            modes.append(CognitiveMode.EVOLVE)

        # Parallel work is valuable when the environment supports independent tools.
        if len(state.available_tools) >= 2 and state.latency_budget < 0.5:
            modes.append(CognitiveMode.PARALLELIZE)

        # Deduplicate without destroying priority order.
        return list(dict.fromkeys(modes))

    def choose(
        self,
        state: MetaState,
        estimates: Iterable[ModeEstimate],
        *,
        weights: UtilityWeights = UtilityWeights(),
    ) -> ModeEstimate:
        """Choose the highest-utility allowed mode from external estimates."""

        allowed = set(self.propose_modes(state))
        candidates = [e for e in estimates if e.mode in allowed]
        if not candidates:
            raise ValueError("No allowed cognitive-mode estimate was provided")
        return max(candidates, key=lambda e: utility(e, weights))


def should_continue_deliberating(
    *,
    expected_quality_gain: float,
    expected_information_gain: float,
    compute_cost: float,
    latency_cost: float,
    risk_cost: float,
) -> bool:
    """Simple marginal-value stopping rule."""

    benefit = expected_quality_gain + 0.35 * expected_information_gain
    cost = 0.20 * compute_cost + 0.20 * latency_cost + 0.60 * risk_cost
    return benefit > cost
