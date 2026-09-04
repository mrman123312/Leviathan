from __future__ import annotations

import unittest

from leviathan.cognitive_kernel import (
    CognitiveCompiler,
    CognitiveOperator,
    DynamicCognitiveGraph,
    Evidence,
    EvidenceUpdater,
    GoalState,
    LearningDestination,
    LearningRouter,
    LeviathanCognitiveKernel,
    NodeStatus,
    RepresentationCompiler,
    RepresentationKind,
    load_cognitive_kernel_spec,
)
from leviathan.types import (
    Belief,
    MetaState,
    Provenance,
    ProvenanceKind,
    UncertaintyKind,
)


def meta_state(**overrides):
    values = dict(
        task_type="diagnosis",
        goal="find the causal explanation",
        success_probability=0.4,
        epistemic_uncertainty=0.7,
        aleatoric_uncertainty=0.1,
        stakes=0.3,
        risk_budget=0.4,
        compute_budget=0.6,
        latency_budget=0.4,
        world_model_confidence=0.8,
        branching_factor_estimate=5,
    )
    values.update(overrides)
    return MetaState(**values)


class CognitiveKernelTests(unittest.TestCase):
    def test_spec_preserves_single_model_and_learning_governance(self) -> None:
        spec = load_cognitive_kernel_spec()
        self.assertTrue(spec["single_cognitive_model"])
        self.assertEqual(spec["invariant"]["semantic_model_count"], 1)
        self.assertFalse(spec["invariant"]["subagent_committee"])
        self.assertFalse(spec["learning"]["raw_experience_to_core"])
        self.assertTrue(spec["learning"]["core_requires_external_promotion_authority"])

    def test_representation_compiler_changes_internal_language_by_problem(self) -> None:
        compiler = RepresentationCompiler()
        math_plan = compiler.compile(task_type="math", problem="prove x = x", uncertainty=0.2)
        causal_plan = compiler.compile(
            task_type="diagnosis",
            problem="why did the machine fail?",
            uncertainty=0.8,
        )
        self.assertEqual(math_plan.primary, RepresentationKind.SYMBOLIC)
        self.assertEqual(causal_plan.primary, RepresentationKind.CAUSAL)
        self.assertGreater(causal_plan.abstraction_budget, math_plan.abstraction_budget)

    def test_kernel_compiles_one_problem_into_explicit_dynamic_graph(self) -> None:
        kernel = LeviathanCognitiveKernel(model_id="deepseek-v4-pro-base")
        goal = GoalState(objective="identify the failure mechanism")
        program, graph = kernel.compile_problem(
            problem="Why did the pump stop after the pressure spike?",
            task_type="diagnosis",
            goal=goal,
            state=meta_state(),
        )
        operators = [item.operator for item in program.instructions]
        self.assertIn(CognitiveOperator.HYPOTHESIZE, operators)
        self.assertIn(CognitiveOperator.PREDICT, operators)
        self.assertIn(CognitiveOperator.SIMULATE, operators)
        self.assertIn(CognitiveOperator.VERIFY, operators)
        self.assertIn(CognitiveOperator.COMPILE_SKILL, operators)
        self.assertTrue(graph.ready())
        self.assertEqual(kernel.event_log.events[-1].metadata["model_id"], "deepseek-v4-pro-base")

    def test_kernel_rejects_multi_model_id_syntax(self) -> None:
        for model_id in ("model-a,model-b", "model-a|model-b", "model-a;model-b"):
            with self.assertRaises(ValueError):
                LeviathanCognitiveKernel(model_id=model_id)

    def test_graph_enforces_dependencies_and_blocks_failure_descendants(self) -> None:
        kernel = LeviathanCognitiveKernel(model_id="deepseek-v4-pro-base")
        program, graph = kernel.compile_problem(
            problem="Why did the system fail?",
            task_type="diagnosis",
            goal=GoalState(objective="diagnose"),
            state=meta_state(),
        )
        first = graph.ready()[0]
        graph.start(first.instruction.id)
        graph.fail(first.instruction.id, error="synthetic failure")
        self.assertEqual(graph.nodes[first.instruction.id].status, NodeStatus.FAILED)
        self.assertTrue(any(node.status is NodeStatus.BLOCKED for node in graph.nodes.values()))

    def test_evidence_update_discounts_non_independent_evidence(self) -> None:
        provenance = Provenance(
            kind=ProvenanceKind.TRUSTED_MEASUREMENT,
            source_id="sensor-a",
            trust_prior=0.9,
        )
        belief = Belief(
            id="b1",
            value=True,
            confidence=0.5,
            provenance=provenance,
            uncertainty=UncertaintyKind.EPISTEMIC,
        )
        updater = EvidenceUpdater()
        strong = updater.update(
            belief,
            Evidence("e1", "b1", True, 0.8, 1.0, provenance),
        )
        correlated = updater.update(
            belief,
            Evidence("e2", "b1", True, 0.8, 0.1, provenance),
        )
        self.assertGreater(strong.posterior_confidence, correlated.posterior_confidence)
        self.assertGreater(strong.posterior_confidence, belief.confidence)

    def test_learning_router_uses_memory_before_parameters(self) -> None:
        router = LearningRouter()
        episodic = router.route(
            verified=False,
            truth_quality=0.4,
            novelty=0.8,
            transfer_value=0.8,
            repeated_successes=0,
            rollback_available=False,
            independent_verification=False,
        )
        self.assertEqual(episodic.destination, LearningDestination.EPISODIC)
        self.assertFalse(episodic.may_touch_ancestral_weights)

        skill = router.route(
            verified=True,
            truth_quality=0.95,
            novelty=0.8,
            transfer_value=0.9,
            repeated_successes=8,
            rollback_available=True,
            independent_verification=True,
        )
        self.assertEqual(skill.destination, LearningDestination.PROCEDURAL)
        self.assertFalse(skill.may_touch_ancestral_weights)

    def test_core_learning_cannot_bypass_governance(self) -> None:
        router = LearningRouter()
        with self.assertRaises(RuntimeError):
            router.core_candidate(
                independent_verification=True,
                rollback_available=True,
                replay_pass=True,
                calibration_pass=True,
                safety_pass=True,
                shadow_pass=True,
                external_promotion_authority=False,
            )
        core = router.core_candidate(
            independent_verification=True,
            rollback_available=True,
            replay_pass=True,
            calibration_pass=True,
            safety_pass=True,
            shadow_pass=True,
            external_promotion_authority=True,
        )
        self.assertEqual(core.destination, LearningDestination.CORE)
        self.assertTrue(core.may_touch_ancestral_weights)

    def test_cognitive_compilation_requires_repeated_verified_success(self) -> None:
        kernel = LeviathanCognitiveKernel(model_id="deepseek-v4-pro-base")
        program, _ = kernel.compile_problem(
            problem="Find the mechanism",
            task_type="diagnosis",
            goal=GoalState(objective="solve repeatable task"),
            state=meta_state(),
        )
        compiler = CognitiveCompiler()
        for index in range(7):
            skill = compiler.observe(program, episode_id=f"ep-{index}", verified_success=True)
        self.assertFalse(skill.ready_to_compile())
        skill = compiler.observe(program, episode_id="ep-7", verified_success=True)
        self.assertTrue(skill.ready_to_compile())
        compiled = compiler.compile(program.fingerprint)
        self.assertTrue(compiled.compiled)


if __name__ == "__main__":
    unittest.main()
