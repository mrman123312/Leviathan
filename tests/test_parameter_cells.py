from __future__ import annotations

import unittest

from leviathan.parameter_cells import (
    CellAction,
    CellBudget,
    CellIdentity,
    CoalitionRegistry,
    DisagreementThresholds,
    MoPStage,
    stage_sequence,
)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    from leviathan.parameter_cells import (
        AssociativeCellRecruiter,
        CellizedPackedExpertsWrapper,
        SparseCellCommunication,
        install_parameter_cell_reference,
        restore_parameter_cell_reference,
    )
except ImportError:
    torch = None
    nn = None
    F = None
    AssociativeCellRecruiter = CellizedPackedExpertsWrapper = SparseCellCommunication = None
    install_parameter_cell_reference = restore_parameter_cell_reference = None


if torch is not None:
    class TinyPackedExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 3
            self.hidden_dim = 8
            self.intermediate_dim = 8
            self.gate_up_proj = nn.Parameter(
                torch.randn(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim) / 8
            )
            self.down_proj = nn.Parameter(
                torch.randn(self.num_experts, self.hidden_dim, self.intermediate_dim) / 8
            )
            self.act_fn = F.silu
            self.limit = 2.5

        def _apply_gate(self, gate_up):
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            return self.act_fn(gate) * up

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx_tensor in hit:
                expert_idx = int(expert_idx_tensor[0].item())
                top_k_pos, token_idx = torch.where(mask[expert_idx])
                current = self._apply_gate(
                    F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
                )
                current = (
                    F.linear(current, self.down_proj[expert_idx])
                    * top_k_weights[token_idx, top_k_pos, None]
                )
                final.index_add_(0, token_idx, current.to(final.dtype))
            return final


    class FakePackedMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Identity()
            self.experts = TinyPackedExperts()
            self.shared_experts = nn.Identity()


    class FakePackedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = FakePackedMoE()
else:
    TinyPackedExperts = FakePackedMoE = FakePackedModel = None


class ParameterCellControlTests(unittest.TestCase):
    def test_stage_roadmap_is_complete(self) -> None:
        self.assertEqual(stage_sequence(), tuple(MoPStage))

    def test_cell_identity_is_stable_and_not_a_separate_agent(self) -> None:
        cell = CellIdentity(layer_index=7, expert_index=3, tile_index=5, tiles_per_expert=24)
        self.assertEqual(cell.local_cell_id, 77)
        self.assertEqual(cell.stable_key, "L7:E3:T5")

    def test_disagreement_controls_bounded_escalation(self) -> None:
        thresholds = DisagreementThresholds(communicate=0.1, recruit=0.3)
        self.assertEqual(thresholds.action(0.01), CellAction.COMMIT)
        self.assertEqual(thresholds.action(0.20), CellAction.COMMUNICATE)
        self.assertEqual(thresholds.action(0.40), CellAction.RECRUIT)
        budget = CellBudget(seed_cells=64, recruited_cells_per_round=32, max_active_cells=256)
        self.assertEqual(budget.max_rounds, 2)

    def test_coalitions_require_repeated_verified_success(self) -> None:
        registry = CoalitionRegistry()
        for _ in range(7):
            registry.record([9, 2, 4, 2], verified_success=True)
        self.assertEqual(registry.candidates(), ())
        registry.record([2, 4, 9], verified_success=True)
        candidate = registry.candidates()[0]
        self.assertEqual(candidate.cell_ids, (2, 4, 9))
        self.assertEqual(candidate.success_rate, 1.0)


@unittest.skipIf(torch is None, "PyTorch inference extra not installed in core CI")
class ParameterCellTorchTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.experts = TinyPackedExperts()
        self.hidden = torch.randn(6, 8)
        self.indices = torch.tensor(
            [[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1]],
            dtype=torch.long,
        )
        self.weights = torch.softmax(torch.randn(6, 2), dim=-1)

    def test_zero_gated_cell_membrane_preserves_exact_expert_function(self) -> None:
        expected = self.experts(self.hidden, self.indices, self.weights)
        wrapped = CellizedPackedExpertsWrapper(
            self.experts,
            tile_width=2,
            collect_telemetry=True,
        )
        actual = wrapped(self.hidden, self.indices, self.weights)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))
        self.assertEqual(float(wrapped.membrane.influence.item()), 0.0)
        self.assertIsNotNone(wrapped.last_telemetry)
        self.assertGreater(wrapped.last_telemetry.active_cell_token_pairs, 0)
        self.assertGreater(wrapped.last_telemetry.unique_cells_seen, 0)

    def test_new_cell_path_cannot_affect_output_before_gate_warmup(self) -> None:
        wrapped = CellizedPackedExpertsWrapper(self.experts, tile_width=2)
        with self.assertRaises(RuntimeError):
            wrapped.set_influence(
                0.1,
                transplant_phase="train_new_parameters_only",
            )
        wrapped.set_influence(0.1, transplant_phase="gate_warmup")
        self.assertAlmostEqual(float(wrapped.membrane.influence.item()), 0.1, places=6)

    def test_model_level_install_and_restore_preserve_shared_path(self) -> None:
        model = FakePackedModel()
        original = model.moe.experts
        shared = model.moe.shared_experts
        report = install_parameter_cell_reference(model, tile_width=2)
        self.assertEqual(report.moe_modules, 1)
        self.assertEqual(report.wrapped_experts, 3)
        self.assertGreater(report.control_parameters, 0)
        self.assertIsInstance(model.moe.experts, CellizedPackedExpertsWrapper)
        self.assertIs(model.moe.shared_experts, shared)

        restored = restore_parameter_cell_reference(model)
        self.assertEqual(restored, 3)
        self.assertIs(model.moe.experts, original)
        self.assertIs(model.moe.shared_experts, shared)

    def test_sparse_communication_never_crosses_group_boundary(self) -> None:
        communication = SparseCellCommunication(message_dim=4, max_neighbors=2)
        messages = torch.randn(4, 4)
        groups = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        output = communication(messages, groups)
        self.assertEqual(tuple(output.shape), (4, 4))

        changed = messages.clone()
        changed[2:] = changed[2:] + 100.0
        output_changed = communication(changed, groups)
        self.assertTrue(torch.allclose(output[:2], output_changed[:2], atol=1e-6, rtol=1e-6))

    def test_associative_recruiter_respects_exclusions(self) -> None:
        recruiter = AssociativeCellRecruiter(num_cells=12, key_dim=4)
        queries = torch.randn(2, 4)
        excluded = torch.tensor([0, 1, 2], dtype=torch.long)
        indices, scores = recruiter(queries, k=4, excluded_cell_ids=excluded)
        self.assertEqual(tuple(indices.shape), (2, 4))
        self.assertEqual(tuple(scores.shape), (2, 4))
        self.assertTrue(all(int(value) not in {0, 1, 2} for value in indices.flatten()))


if __name__ == "__main__":
    unittest.main()
