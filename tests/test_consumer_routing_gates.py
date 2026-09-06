from __future__ import annotations
import unittest
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from leviathan.consumer.cells import SwiGLUCells, CellEcology, EcologyConfig
except ImportError:
    torch = None

@unittest.skipIf(torch is None, "PyTorch extra not installed")
class RecruitmentGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(713)
        class Donor(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(16, 64, bias=False)
                self.up_proj = nn.Linear(16, 64, bias=False)
                self.down_proj = nn.Linear(64, 16, bias=False)
                self.act_fn = F.silu
        self.bank = SwiGLUCells(Donor(), 8)
        self.hidden, self.latent = torch.randn(4,16), torch.randn(4,16)
    def make(self, extra=2, threshold=0.):
        return CellEcology(self.bank, EcologyConfig(latent_dim=16, seed_cells=2,
            recruit_cells=extra, max_cells=4, recruit_threshold=threshold))
    def test_closed_recruitment_does_not_renormalize_seed_output(self):
        observation, no_recruit = self.make(), self.make(extra=0)
        no_recruit.load_state_dict(observation.state_dict())
        a = observation(self.hidden, self.latent)
        b = no_recruit(self.hidden, self.latent)
        self.assertGreater(a.recruited, 0)
        torch.testing.assert_close(a.proposal, b.proposal, atol=0, rtol=0)
    def test_closed_communication_is_an_identity_map(self):
        ecology = self.make()
        controls = torch.randn(4, 3, 16)
        mask = torch.ones(4, 3, dtype=torch.bool)
        torch.testing.assert_close(ecology._discuss(controls, mask), controls, atol=0, rtol=0)
    def test_opened_recruitment_trains_query_weights(self):
        ecology = self.make()
        ecology.recruit_gate.data.fill_(.5)
        ecology(self.hidden, self.latent).proposal.square().sum().backward()
        self.assertIsNotNone(ecology.recruit_query.weight.grad)
        self.assertGreater(float(ecology.recruit_query.weight.grad.abs().sum()), 0)
    def test_second_round_does_not_run_behaviorally_without_recruits(self):
        from dataclasses import replace
        ecology = self.make(threshold=1e9)
        ecology.communication_gate.data.fill_(.5)
        ecology.recruit_gate.data.fill_(.5)
        once = self.make(threshold=1e9)
        once.config = replace(once.config, rounds=1)
        once.load_state_dict(ecology.state_dict())
        a, b = ecology(self.hidden, self.latent), once(self.hidden, self.latent)
        self.assertEqual(a.recruited,0)
        torch.testing.assert_close(a.proposal, b.proposal, atol=0, rtol=0)

if __name__ == '__main__':
    unittest.main()
