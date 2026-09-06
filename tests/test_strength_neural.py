"""Small frozen random neural fixtures. These tests are NOT pretrained results."""
import unittest,importlib.util
HAS=importlib.util.find_spec('torch') is not None
if HAS:
    import torch
    from test_bedrock_neural import TinyLM
    from leviathan.bedrock.neural import FrozenExecutor
    from leviathan.strength.neural import NeuralFabric,NeuralRoute,TaskWorkspace,bounded
    from leviathan.strength.proposer import QwenProposer,render_task

@unittest.skipUnless(HAS,'PyTorch optional')
class StrengthNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(372);torch.set_num_threads(1)
        self.model=TinyLM().eval();self.engine=FrozenExecutor(self.model,model_id='one',revision='fixed')
        self.fabric=NeuralFabric(self.engine);self.ids=torch.tensor([[1,2,3,4,5]])
    def test_donor_exact_and_hooks_absent(self):
        a=self.model(self.ids).logits
        with self.fabric.use(NeuralRoute()):b=self.model(self.ids).logits
        self.assertTrue(torch.equal(a,b));self.assertTrue(all(not x._forward_hooks for x in self.model.modules()))
    def test_task_state_changes_computation_not_weights(self):
        state=TaskWorkspace('fixed',1,'support',torch.randn(2,16));before={k:v.clone() for k,v in self.model.state_dict().items()}
        a=self.model(self.ids).logits
        with self.fabric.use(NeuralRoute('task_state',layer=1),workspace=state):b=self.model(self.ids).logits
        self.assertFalse(torch.equal(a,b));self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in before.items()))
    def test_slots_are_distinct_with_same_model(self):
        state=TaskWorkspace('fixed',1,'support',torch.stack((torch.ones(16),-torch.ones(16))))
        with self.fabric.use(NeuralRoute('task_state',layer=1),workspace=state,slot=0):a=self.model(self.ids).logits
        with self.fabric.use(NeuralRoute('task_state',layer=1),workspace=state,slot=1):b=self.model(self.ids).logits
        self.assertFalse(torch.equal(a,b))
    def test_scope_mismatch_rejected(self):
        state=TaskWorkspace('wrong',1,'support',torch.ones(1,16))
        with self.assertRaises(ValueError):
            with self.fabric.use(NeuralRoute('task_state',layer=1),workspace=state):pass
    def test_damped_band_is_finite_causal_and_parameter_preserving(self):
        route=NeuralRoute('damped_band',layer=1,passes=2,radius=.25)
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        with self.fabric.use(route):full=self.model(self.ids).logits
        with self.fabric.use(route):prefix=self.model(self.ids[:,:3]).logits
        torch.testing.assert_close(full[:,:3],prefix,atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.isfinite(full).all());self.assertTrue(self.engine.unchanged())
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in before.items()))
    def test_cell_intervention_scope_and_restore(self):
        route=NeuralRoute('cell_ablation',layer=1,cell_width=8)
        original=self.model(self.ids).logits
        with self.fabric.use(route):ablated=self.model(self.ids).logits
        self.assertFalse(torch.equal(original,ablated));self.assertTrue(torch.equal(original,self.model(self.ids).logits))
    def test_hooks_removed_on_failure(self):
        with self.assertRaises(RuntimeError):
            with self.fabric.use(NeuralRoute('damped_band',layer=1)):raise RuntimeError('test')
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
    def test_contrast_extraction_no_optimization(self):
        ws=self.fabric.contrast_workspace([self.ids],[self.ids+1],layer=1,support_hash='source-demo')
        self.assertEqual(ws.vectors.shape,(1,16));self.assertFalse(ws.metadata()['learned_by_sgd'])
        self.assertTrue(self.engine.unchanged())
    def test_support_gate_has_no_benchmark_truth(self):
        route,report=self.fabric.select_on_demonstrations([(self.ids,3)])
        self.assertEqual(route.kind,'donor');self.assertFalse(report['accepted'])
    def test_zero_radius_is_identity(self):
        x=torch.randn(3,16);self.assertTrue(torch.equal(bounded(x,torch.randn_like(x),0),x))
    def test_trust_region_bound(self):
        x=torch.randn(3,16);y=bounded(x,torch.randn_like(x)*100,.1)
        self.assertTrue(((y-x).norm(dim=-1)<=.1*x.norm(dim=-1)+1e-6).all())
