from __future__ import annotations
import unittest
from leviathan.consumer.profiles import PROFILES, get_profile
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from leviathan.consumer.cells import SwiGLUCells, CellEcology, EcologyConfig
    from leviathan.consumer.quantization import Int4Linear, slice_weight
    from leviathan.consumer.recurrence import NRDFConfig, QwenNRDFWrapper, RecurrentFabric, FastOverlay
    from leviathan.consumer.efficiency import CacheScope, ExactDeltaCache, ByteLRU, certified_topk_reuse, coefficient_delta
    from leviathan.consumer.speculation import verify_greedy, verify_sampled
    from leviathan.consumer.training import EvaluationSplit, content_hash, halting_supervision
except ImportError:
    torch = None

if torch is not None:
    class TinySwiGLU(nn.Module):
        def __init__(self, hidden=16, intermediate=32, bias=False):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=bias)
            self.up_proj = nn.Linear(hidden, intermediate, bias=bias)
            self.down_proj = nn.Linear(intermediate, hidden, bias=bias)
            self.act_fn = F.silu
        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class ConsumerProfileTests(unittest.TestCase):
    def test_training_stage_and_pins(self):
        self.assertEqual(get_profile("rtx3060").stage, "base")
        self.assertEqual(get_profile("qwen27b").stage, "posttrained")
        for p in PROFILES.values():
            self.assertEqual(len(p.revision), 40)
        self.assertEqual(get_profile("qwen27b").cells_per_ffn(), 136)
        self.assertEqual(get_profile("rtx3060").cells_per_ffn(), 48)
    def test_memory_is_not_mislabelled_measurement(self):
        self.assertFalse(get_profile("rtx3060").memory_estimate()["measured"])
        self.assertGreater(get_profile("qwen27b").memory_estimate()["nominal_weights_gib"], 12)
    def test_wrong_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            get_profile("qwen27b").validate_config({"model_type": "qwen3_5"})

@unittest.skipIf(torch is None, "CPU PyTorch optional extra is not installed")
class ConsumerNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        torch.set_num_threads(1)
        self.donor = TinySwiGLU()
        self.x = torch.randn(6, 16)
        self.cfg = NRDFConfig(latent_dim=16, heads=2, slots=3, max_loops=3,
                              cell_width=8, chunk_tokens=2)
    def test_dense_ancestral_parity_with_bias_once(self):
        donor = TinySwiGLU(bias=True)
        for width in (4, 8, 16, 32):
            torch.testing.assert_close(SwiGLUCells(donor, width).reconstruct(self.x), donor(self.x),
                                       atol=1e-6, rtol=1e-6)
    def test_packed_int4_slice_and_ancestral_parity(self):
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = getattr(self.donor, name)
            packed = Int4Linear.from_linear(layer, group_size=8)
            self.assertLess(packed.storage_bytes, layer.weight.numel() * 4)
            test_input = torch.randn(6, layer.in_features)
            torch.testing.assert_close(packed(test_input), F.linear(test_input, packed.weight_slice()))
            weight = packed.weight_slice()
            torch.testing.assert_close(packed.weight_slice(slice(1, 4), slice(3, 11)), weight[1:4, 3:11])
            setattr(self.donor, name, packed)
        torch.testing.assert_close(SwiGLUCells(self.donor, 8).reconstruct(self.x), self.donor(self.x),
                                   atol=1e-6, rtol=1e-6)
    def test_buffer_only_int4_donor_accepts_zero_gated_recurrence(self):
        from dataclasses import replace
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.donor, name, Int4Linear.from_linear(getattr(self.donor, name), group_size=8))
        wrapper = QwenNRDFWrapper(self.donor, replace(self.cfg, ancestral_cells=True)).train()
        actual = wrapper(self.x)
        torch.testing.assert_close(actual, self.donor(self.x), atol=0, rtol=0)
        actual.square().mean().backward()
        self.assertIsNotNone(wrapper.gate.grad)
    def test_foreign_quantization_not_silently_sliced(self):
        class Foreign(nn.Linear):
            pass
        with self.assertRaises(TypeError):
            slice_weight(Foreign(4, 4))
    def test_zero_gate_preserves_donor_but_receives_gradient(self):
        wrapper = QwenNRDFWrapper(self.donor, self.cfg).train()
        actual = wrapper(self.x)
        torch.testing.assert_close(actual, self.donor(self.x), atol=0, rtol=0)
        actual.square().mean().backward()
        self.assertIsNotNone(wrapper.gate.grad)
        self.assertGreater(abs(float(wrapper.gate.grad)), 0)
        self.assertEqual(float(wrapper.fabric.output.weight.grad.abs().sum()), 0)
    def test_opened_recurrence_backward_and_no_state_leak(self):
        wrapper = QwenNRDFWrapper(self.donor, self.cfg).train()
        wrapper.set_influence(0.2, experimental=True)
        out1 = wrapper(self.x)
        wrapper(self.x + 7)
        out2 = wrapper(self.x)
        torch.testing.assert_close(out1, out2, atol=0, rtol=0)
        out2.square().sum().backward()
        self.assertGreater(float(wrapper.fabric.attention.in_proj_weight.grad.abs().sum()), 0)
        permutation = torch.tensor([5, 1, 3, 0, 4, 2])
        torch.testing.assert_close(wrapper(self.x[permutation]), out1[permutation], atol=1e-6, rtol=1e-6)
    def test_variable_depth_and_halting_compaction(self):
        from dataclasses import replace
        fabric = RecurrentFabric(16, replace(self.cfg, delta_threshold=1e9)).eval()
        _, trace = fabric(self.x, adaptive=True)
        self.assertTrue((trace.loops == 1).all())
        self.assertEqual(trace.active_rows_per_loop, (6, 0))
        for depth in (1, 2, 3):
            _, trace = fabric(self.x, loops=depth)
            self.assertTrue((trace.loops == depth).all())
        with self.assertRaises(ValueError):
            fabric(self.x, loops=4)
    def test_cell_communication_recruitment_and_state_gradients(self):
        bank = SwiGLUCells(self.donor, width=8)
        ec = CellEcology(bank, EcologyConfig(latent_dim=16, seed_cells=2, recruit_cells=2,
                                            max_cells=4, recruit_threshold=0))
        for gate in (ec.communication_gate, ec.state_gate, ec.recruit_gate):
            gate.data.fill_(0.5)
        latent = torch.randn(6, 16, requires_grad=True)
        a = ec(self.x, latent)
        b = ec(self.x, latent, a.state)
        b.proposal.square().sum().backward()
        self.assertEqual(a.recruited, 12)
        self.assertLessEqual(int(a.mask.sum(1).max()), 4)
        for ids in a.ids:
            self.assertEqual(len(torch.unique(ids)), len(ids))
        self.assertGreater(float(ec.peer.weight.grad.abs().sum()), 0)
        self.assertGreater(float(ec.state_cell.weight_ih.grad.abs().sum()), 0)
        fresh = ec(self.x, latent)
        torch.testing.assert_close(a.proposal, fresh.proposal)
    def test_cells_integrated_in_recurrent_wrapper(self):
        from dataclasses import replace
        wrapper = QwenNRDFWrapper(self.donor, replace(self.cfg, ancestral_cells=True)).train()
        torch.testing.assert_close(wrapper(self.x), self.donor(self.x), atol=0, rtol=0)
        wrapper.set_influence(.2, experimental=True)
        wrapper.fabric.cell_gate.data.fill_(.2)
        result = wrapper(self.x)
        result.square().mean().backward()
        self.assertIsNotNone(wrapper.fabric.ecology.keys.grad)
        self.assertTrue(all(t.routes for t in wrapper.last_traces))
    def test_fast_overlay_bounded_and_differentiable(self):
        overlay = FastOverlay(16, 4, .1)
        overlay.gate.data.fill_(.5)
        x = torch.randn(4, 16)
        _, state = overlay(x, None)
        y, state2 = overlay(x, state)
        self.assertLessEqual(float(state2.detach().flatten(1).norm(dim=1).max()), .100001)
        y.square().sum().backward()
        self.assertGreater(float(overlay.propose.weight.grad.abs().sum()), 0)
    def test_halting_auxiliary_reaches_head(self):
        fabric = RecurrentFabric(16, self.cfg)
        _, trace = fabric(self.x)
        halting_supervision(trace.halt_probabilities, torch.tensor([1, 2, 3, 1, 2, 3])).backward()
        self.assertGreater(float(fabric.halt_head.weight.grad.abs().sum()), 0)
    def test_cache_reuses_only_exact_matching_rows_and_scope(self):
        cache = ExactDeltaCache()
        scope = CacheScope("revision", 0, "request-a", "fp32", "linear")
        with torch.inference_mode():
            cache.run(self.x, self.donor, scope)
            changed = self.x.clone()
            changed[2] += .1
            result = cache.run(changed, self.donor, scope)
            torch.testing.assert_close(result, self.donor(changed))
            self.assertEqual(cache.last_reused_rows, 5)
            cache.run(changed, self.donor, CacheScope("revision", 1, "request-a", "fp32", "linear"))
            self.assertEqual(cache.last_reused_rows, 0)
        with self.assertRaises(RuntimeError):
            cache.run(self.x, self.donor, scope)
    def test_byte_budget_eviction_and_mutation_isolation(self):
        cache = ByteLRU(32)
        cache.put((1,), torch.ones(4))
        cache.put((2,), torch.ones(4))
        cache.put((3,), torch.ones(4))
        self.assertIsNone(cache.get((1,)))
        item = cache.get((2,))
        item.zero_()
        self.assertEqual(float(cache.get((2,)).sum()), 4)
        self.assertLessEqual(cache.bytes, 32)
    def test_route_margin_certificate(self):
        keys = torch.eye(4)
        old = torch.tensor([10., 5., 1., 0.])
        ids = torch.tensor([0, 1])
        self.assertTrue(certified_topk_reuse(old, old + .01, keys, keys @ old, ids))
        self.assertFalse(certified_topk_reuse(old, torch.tensor([0., 0., 30., 20.]), keys, keys @ old, ids))
        body = torch.randn(3, 4)
        w1, w2 = torch.randn(3), torch.randn(3)
        torch.testing.assert_close(coefficient_delta((body * w1[:, None]).sum(0), body, w1, w2),
                                   (body * w2[:, None]).sum(0))
    def test_greedy_speculation_alignment_eos_and_rejection(self):
        draft = torch.tensor([1, 2])
        logits = torch.tensor([[0., 2., 1.], [3., 2., 1.], [1., 2., 3.]])
        result = verify_greedy(draft, logits)
        self.assertEqual(result.tokens, (1, 0))
        self.assertEqual(result.accepted, 1)
        self.assertEqual(verify_greedy(draft, logits, eos_token_id=1).tokens, (1,))
    def test_sampled_verification_matches_target_distribution(self):
        generator = torch.Generator().manual_seed(99)
        q = torch.tensor([[.8, .2]])
        p = torch.tensor([[.2, .8], [.2, .8]])
        counts = [0, 0]
        for _ in range(4000):
            draft = torch.multinomial(q[0], 1, generator=generator)
            result = verify_sampled(draft, q, p, generator=generator)
            counts[result.tokens[0]] += 1
        self.assertLess(abs(counts[0] / 4000 - .2), .025)
    def test_pulse_reuses_donor_head_and_alignment_trains_bridge(self):
        from leviathan.consumer.pulse import PulseBridge
        embedding = nn.Embedding(50, 16)
        head = nn.Linear(16, 50, bias=False)
        bridge = PulseBridge(embedding, head, 8, 16)
        latent = torch.randn(3, 8)
        result = bridge(latent)
        self.assertEqual(result.shape, latent.shape)
        self.assertNotIn("_embedding.weight", bridge.state_dict())
        bridge.alignment_loss(latent, torch.tensor([1, 2, 3])).backward()
        self.assertGreater(float(bridge.readout.weight.grad.abs().sum()), 0)
    def test_empty_or_invalid_evidence_cannot_pass_retention(self):
        from leviathan.consumer.training import RetentionGate
        self.assertFalse(RetentionGate().evaluate({}, {})[0])
        self.assertFalse(RetentionGate().evaluate({"wikitext_loss": 1.0}, {})[0])
        self.assertFalse(RetentionGate().evaluate({"wikitext_loss": float("nan")}, {"wikitext_loss": 1.0})[0])
        with self.assertRaises(ValueError):
            RetentionGate(accuracy_tolerance=float("nan"))
    def test_heldout_overlap_rejected(self):
        with self.assertRaises(ValueError):
            EvaluationSplit(frozenset({content_hash("x")}), frozenset({content_hash("x")}))

if __name__ == "__main__":
    unittest.main()
