from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

from leviathan.parameter_cells import (
    CellBudget,
    CellExecutionConfig,
    CellizedPackedExpertsWrapper,
    DisagreementThresholds,
    MoPStage,
)


if torch is not None:
    class TinyPackedExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 4
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
else:
    TinyPackedExperts = None


@unittest.skipIf(torch is None, "PyTorch inference extra not installed in core CI")
class LiveParameterEcologyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.experts = TinyPackedExperts()
        self.hidden = torch.randn(5, 8)
        self.indices = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
            dtype=torch.long,
        )
        self.weights = torch.softmax(torch.randn(5, 2), dim=-1)

    def _full_reference_wrapper(self) -> CellizedPackedExpertsWrapper:
        return CellizedPackedExpertsWrapper(
            self.experts,
            tile_width=2,
            collect_telemetry=True,
            disagreement=DisagreementThresholds(communicate=0.0, recruit=1e-12),
            execution=CellExecutionConfig(
                stage=MoPStage.LOCAL_STATE,
                independent_top_k=4,
                state_update_rate=0.5,
            ),
            budget=CellBudget(
                seed_cells=4,
                recruited_cells_per_round=3,
                max_active_cells=16,
                max_rounds=2,
                max_neighbors=4,
            ),
        )

    def test_all_new_mechanisms_execute_while_zero_gates_preserve_donor(self) -> None:
        expected = self.experts(self.hidden, self.indices, self.weights)
        wrapped = self._full_reference_wrapper()
        actual = wrapped(self.hidden, self.indices, self.weights)

        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))
        self.assertEqual(float(wrapped.membrane.influence.item()), 0.0)
        self.assertEqual(float(wrapped.membrane.communication_influence.item()), 0.0)
        self.assertEqual(float(wrapped.membrane.state_influence.item()), 0.0)
        self.assertEqual(float(wrapped.recruitment_influence.item()), 0.0)
        self.assertEqual(float(wrapped.independent_route_influence.item()), 0.0)

        telemetry = wrapped.last_telemetry
        self.assertIsNotNone(telemetry)
        self.assertEqual(telemetry.communication_rounds, 2)
        self.assertGreater(telemetry.recruited_cell_token_pairs, 0)
        self.assertGreater(telemetry.unique_recruited_cells, 0)
        self.assertGreater(telemetry.independent_route_cell_token_pairs, 0)
        self.assertGreater(telemetry.local_state_updates, 0)
        self.assertGreater(float(wrapped.local_state.abs().sum().item()), 0.0)
        self.assertTrue(wrapped.last_recruited_cell_ids)
        self.assertTrue(wrapped.last_independent_cell_ids)

    def test_independent_cross_expert_route_can_take_control_only_after_gate_warmup(self) -> None:
        wrapped = CellizedPackedExpertsWrapper(
            self.experts,
            tile_width=2,
            execution=CellExecutionConfig(
                stage=MoPStage.INDEPENDENT_TILE_ROUTING,
                independent_top_k=4,
            ),
        )
        donor = wrapped(self.hidden, self.indices, self.weights)
        with self.assertRaises(RuntimeError):
            wrapped.set_independent_route_influence(
                1.0,
                transplant_phase="train_new_parameters_only",
            )
        wrapped.set_independent_route_influence(1.0, transplant_phase="gate_warmup")
        routed = wrapped(self.hidden, self.indices, self.weights)
        self.assertEqual(tuple(routed.shape), tuple(donor.shape))
        self.assertFalse(torch.allclose(donor, routed, atol=1e-6, rtol=1e-6))

    def test_peer_communication_changes_live_refinement_when_opened(self) -> None:
        wrapped = CellizedPackedExpertsWrapper(
            self.experts,
            tile_width=2,
            execution=CellExecutionConfig(stage=MoPStage.ONE_COMMUNICATION_ROUND),
        )
        wrapped.set_influence(0.2, transplant_phase="gate_warmup")
        without_peer_influence = wrapped(self.hidden, self.indices, self.weights)
        wrapped.set_communication_influence(1.0, transplant_phase="gate_warmup")
        with_peer_influence = wrapped(self.hidden, self.indices, self.weights)
        self.assertFalse(
            torch.allclose(without_peer_influence, with_peer_influence, atol=1e-6, rtol=1e-6)
        )

    def test_recruited_ancestral_cells_can_contribute_only_when_gate_is_opened(self) -> None:
        wrapped = CellizedPackedExpertsWrapper(
            self.experts,
            tile_width=2,
            disagreement=DisagreementThresholds(communicate=0.0, recruit=1e-12),
            execution=CellExecutionConfig(stage=MoPStage.DISAGREEMENT_RECRUITMENT),
            budget=CellBudget(
                seed_cells=4,
                recruited_cells_per_round=2,
                max_active_cells=16,
                max_rounds=2,
                max_neighbors=4,
            ),
        )
        donor = wrapped(self.hidden, self.indices, self.weights)
        self.assertGreater(wrapped.last_telemetry.recruited_cell_token_pairs, 0)
        wrapped.set_recruitment_influence(0.5, transplant_phase="gate_warmup")
        recruited = wrapped(self.hidden, self.indices, self.weights)
        self.assertFalse(torch.allclose(donor, recruited, atol=1e-6, rtol=1e-6))

    def test_opened_ecology_paths_are_differentiable(self) -> None:
        wrapped = self._full_reference_wrapper()
        wrapped.set_influence(0.1, transplant_phase="gate_warmup")
        wrapped.set_communication_influence(0.5, transplant_phase="gate_warmup")
        wrapped.set_recruitment_influence(0.1, transplant_phase="gate_warmup")
        wrapped.set_state_influence(0.5, transplant_phase="gate_warmup")
        wrapped.set_independent_route_influence(0.1, transplant_phase="gate_warmup")

        hidden = self.hidden.clone().requires_grad_(True)
        output = wrapped(hidden, self.indices, self.weights)
        loss = output.float().square().mean()
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        trainable_grads = [
            parameter.grad
            for name, parameter in wrapped.named_parameters()
            if not name.startswith("expert_bank.") and parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in trainable_grads))
        self.assertTrue(
            all(torch.isfinite(grad).all() for grad in trainable_grads if grad is not None)
        )

    def test_ephemeral_local_state_updates_and_resets(self) -> None:
        wrapped = self._full_reference_wrapper()
        wrapped(self.hidden, self.indices, self.weights)
        self.assertGreater(float(wrapped.local_state.abs().sum().item()), 0.0)
        wrapped.reset_local_state()
        self.assertEqual(float(wrapped.local_state.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
