from __future__ import annotations
import importlib.util
import types
import unittest
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from test_bedrock_neural import TinyLM
    from leviathan.bedrock.stable_neural import StableFrozenExecutor, StableFrozenPolicy, transport_reentry

@unittest.skipUnless(HAS_TORCH, "PyTorch optional")
class BedrockStabilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(991)
        torch.set_num_threads(1)
        self.model = TinyLM().eval()
        self.engine = StableFrozenExecutor(self.model, model_id="one", revision="fixed")
        self.ids = torch.tensor([[1, 2, 3, 5, 8]])

    def test_transport_reentry_is_finite_and_l2_bounded(self):
        entry = torch.randn(3, 16)
        target = 1000 * torch.randn(3, 16)
        moved = transport_reentry(entry, target, radius=.08, pointwise_multiplier=2.)
        relative = (moved-entry).norm(dim=-1) / entry.norm(dim=-1)
        self.assertTrue((relative <= .08001).all())
        self.assertTrue(torch.isfinite(moved).all())

    def test_transport_changes_computation_without_weight_change(self):
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        base = self.engine.run(self.ids).logits
        policy = StableFrozenPolicy(start=1,end=2,passes=3,gain=.1,reentry_radius=.08)
        changed = self.engine.run(self.ids, policy=policy).logits
        self.assertFalse(torch.equal(base, changed))
        self.assertEqual(self.engine.last_trace["route_status"], "experimental")
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in before.items()))

    def test_nonfinite_replay_falls_back_to_exact_donor_band_output(self):
        baseline = self.engine.run(self.ids).logits
        target = self.model.model.layers[2]
        original = target.forward
        calls = {"n":0}
        def unstable(this, hidden_states, *args, **kwargs):
            calls["n"] += 1
            out = original(hidden_states, *args, **kwargs)
            if calls["n"] >= 2:
                return torch.full_like(out, float("inf"))
            return out
        target.forward = types.MethodType(unstable, target)
        try:
            calls["n"] = 0
            policy = StableFrozenPolicy(start=1,end=2,passes=2,gain=.1,reentry_radius=.08)
            result = self.engine.run(self.ids, policy=policy).logits
            torch.testing.assert_close(result, baseline, atol=0, rtol=0)
            self.assertEqual(self.engine.last_trace["route_status"], "donor_fallback_nonfinite")
            self.assertEqual(self.engine.last_trace["nonfinite_replay_fallbacks"], 1)
        finally:
            target.forward = original

    def test_transport_preserves_prefix_causality_fixture(self):
        policy = StableFrozenPolicy(start=1,end=2,passes=4,gain=.08,reentry_radius=.06,
                                    halt_delta=.01,halt_patience=1)
        full = self.engine.run(self.ids, policy=policy).logits
        prefix = self.engine.run(self.ids[:,:3], policy=policy).logits
        torch.testing.assert_close(full[:,:3], prefix, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
