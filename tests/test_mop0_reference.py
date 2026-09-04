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

from leviathan.mop0_reference import (
    MoP0ExpertWrapper,
    MoP0PackedExpertsWrapper,
    install_mop0_reference,
    restore_original_experts,
)


if torch is not None:
    class TinyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Linear(8, 8, bias=False)
            self.w2 = nn.Linear(8, 8, bias=False)
            self.w3 = nn.Linear(8, 8, bias=False)
            self.swiglu_limit = 0.0

        def forward(self, x, weights=None):
            y = F.silu(self.w1(x).float()) * self.w3(x).float()
            if weights is not None:
                y = weights * y
            return self.w2(y.to(x.dtype))


    class FakeMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Identity()
            self.experts = nn.ModuleList([TinyExpert(), TinyExpert()])
            self.shared_expert = TinyExpert()


    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = FakeMoE()


    class TinyPackedExperts(nn.Module):
        """Small analogue of Transformers DeepseekV4Experts."""

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
    TinyExpert = FakeMoE = FakeModel = None
    TinyPackedExperts = FakePackedMoE = FakePackedModel = None


class MoP0ReferenceTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch inference extra not installed in core CI")
    def test_one_expert_exactly_reconstructs_in_fp32(self) -> None:
        torch.manual_seed(7)
        expert = TinyExpert()
        wrapped = MoP0ExpertWrapper(expert, tile_width=2)
        x = torch.randn(5, 8)
        expected = expert(x)
        actual = wrapped(x)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

    @unittest.skipIf(torch is None, "PyTorch inference extra not installed in core CI")
    def test_patch_wraps_routed_but_not_shared_expert_and_restores(self) -> None:
        model = FakeModel()
        shared_before = model.moe.shared_expert
        first_before = model.moe.experts[0]
        report = install_mop0_reference(model, tile_width=2)

        self.assertEqual(report.moe_modules, 1)
        self.assertEqual(report.wrapped_experts, 2)
        self.assertIs(model.moe.shared_expert, shared_before)
        self.assertIsInstance(model.moe.experts[0], MoP0ExpertWrapper)

        restored = restore_original_experts(model)
        self.assertEqual(restored, 2)
        self.assertIs(model.moe.experts[0], first_before)
        self.assertIs(model.moe.shared_expert, shared_before)

    @unittest.skipIf(torch is None, "PyTorch inference extra not installed in core CI")
    def test_packed_transformers_experts_reconstruct_exact_route(self) -> None:
        torch.manual_seed(11)
        experts = TinyPackedExperts()
        wrapped = MoP0PackedExpertsWrapper(experts, tile_width=2)

        hidden_states = torch.randn(6, 8)
        top_k_index = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [0, 2],
                [1, 0],
                [2, 1],
            ],
            dtype=torch.long,
        )
        top_k_weights = torch.softmax(torch.randn(6, 2), dim=-1)

        expected = experts(hidden_states, top_k_index, top_k_weights)
        actual = wrapped(hidden_states, top_k_index, top_k_weights)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

    @unittest.skipIf(torch is None, "PyTorch inference extra not installed in core CI")
    def test_packed_expert_bank_install_and_restore(self) -> None:
        model = FakePackedModel()
        packed_before = model.moe.experts
        shared_before = model.moe.shared_experts

        report = install_mop0_reference(model, tile_width=2)
        self.assertEqual(report.moe_modules, 1)
        self.assertEqual(report.wrapped_experts, 3)
        self.assertIsInstance(model.moe.experts, MoP0PackedExpertsWrapper)
        self.assertIs(model.moe.shared_experts, shared_before)

        restored = restore_original_experts(model)
        self.assertEqual(restored, 3)
        self.assertIs(model.moe.experts, packed_before)
        self.assertIs(model.moe.shared_experts, shared_before)


if __name__ == "__main__":
    unittest.main()
