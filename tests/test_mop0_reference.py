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
    install_mop0_reference,
    restore_original_experts,
)


@unittest.skipIf(torch is None, "PyTorch inference extra not installed in core CI")
class MoP0ReferenceTests(unittest.TestCase):
    class Expert(nn.Module):
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
            self.experts = nn.ModuleList([MoP0ReferenceTests.Expert(), MoP0ReferenceTests.Expert()])
            self.shared_expert = MoP0ReferenceTests.Expert()

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = MoP0ReferenceTests.FakeMoE()

    def test_one_expert_exactly_reconstructs_in_fp32(self) -> None:
        torch.manual_seed(7)
        expert = self.Expert()
        wrapped = MoP0ExpertWrapper(expert, tile_width=2)
        x = torch.randn(5, 8)
        expected = expert(x)
        actual = wrapped(x)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

    def test_patch_wraps_routed_but_not_shared_expert_and_restores(self) -> None:
        model = self.FakeModel()
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


if __name__ == "__main__":
    unittest.main()
