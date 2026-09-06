"""Mechanism tests use frozen random tiny tensors, NOT pretrained capability scores."""
from __future__ import annotations
import importlib.util
import unittest
from dataclasses import replace
from types import SimpleNamespace
HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from leviathan.bedrock.cells import CellPolicy,FrozenCellBank,conservative_discussion
    from leviathan.bedrock.neural import FrozenPolicy,FrozenExecutor,FastAssociations
    from leviathan.bedrock.contracts import Budget,Meter,BudgetExceeded,Outcome,stable_hash
    from leviathan.consumer.quantization import Int4Linear

    class FFN(nn.Module):
        def __init__(self,d=16):
            super().__init__()
            self.gate_proj=nn.Linear(d,32,bias=True)
            self.up_proj=nn.Linear(d,32,bias=True)
            self.down_proj=nn.Linear(32,d,bias=True)
            self.act_fn=F.silu
        def forward(self,x):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))

    class Layer(nn.Module):
        def __init__(self,d=16):
            super().__init__();self.norm=nn.LayerNorm(d);self.attn=nn.MultiheadAttention(d,2,batch_first=True,dropout=0);self.mlp=FFN(d)
        def forward(self,hidden_states,attention_mask=None,past_key_values=None,use_cache=False,**kwargs):
            if past_key_values is not None:raise ValueError("No synthetic cache")
            x=self.norm(hidden_states)
            y=self.attn(x,x,x,attn_mask=attention_mask,need_weights=False)[0]
            h=hidden_states+y
            return h+self.mlp(self.norm(h))

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(model_type="bedrock_test")
            self.model=nn.Module();self.model.embed_tokens=nn.Embedding(32,16)
            self.model.layers=nn.ModuleList([Layer() for _ in range(4)])
            self.lm_head=nn.Linear(16,32,bias=False)
        def forward(self,input_ids,use_cache=False,**kwargs):
            h=self.model.embed_tokens(input_ids)
            mask=torch.ones((h.shape[1],h.shape[1]),dtype=torch.bool,device=h.device).triu(1)
            for layer in self.model.layers:h=layer(h,attention_mask=mask,use_cache=use_cache)
            return SimpleNamespace(logits=self.lm_head(h))
        def get_input_embeddings(self):return self.model.embed_tokens

@unittest.skipUnless(HAS_TORCH,"PyTorch optional")
class BedrockNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(947);torch.set_num_threads(1)
        self.model=TinyLM().eval();self.ids=torch.tensor([[1,2,3,5,8]])
        self.engine=FrozenExecutor(self.model,model_id="test",revision="fixed-test-revision")
        self.policy=FrozenPolicy(start=1,end=2,passes=3,gain=.2)
    def test_neutral_exact_no_new_parameters(self):
        snapshot={k:v.clone() for k,v in self.model.state_dict().items()}
        ids={id(p) for p in self.model.parameters()}
        expected=self.model(self.ids).logits
        actual=self.engine.run(self.ids).logits
        self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(ids,{id(p) for p in self.model.parameters()})
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in snapshot.items()))
        self.assertTrue(self.engine.last_trace["neutral_direct_path"])
    def test_frozen_recurrence_changes_computation_not_weights(self):
        snapshot={k:v.clone() for k,v in self.model.state_dict().items()}
        a=self.engine.run(self.ids).logits
        b=self.engine.run(self.ids,policy=self.policy).logits
        self.assertFalse(torch.equal(a,b))
        self.assertEqual(self.engine.last_trace["extra_layer_calls"],4)
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in snapshot.items()))
        self.assertTrue(self.engine.unchanged())
    def test_zero_gain_skips_all_replays(self):
        a=self.engine.run(self.ids).logits
        b=self.engine.run(self.ids,policy=replace(self.policy,gain=0)).logits
        self.assertTrue(torch.equal(a,b));self.assertEqual(self.engine.last_trace["extra_layer_calls"],0)
    def test_future_tokens_do_not_change_earlier_outputs(self):
        p=replace(self.policy,halt_delta=.04,halt_patience=1)
        full=self.engine.run(self.ids,policy=p).logits
        prefix=self.engine.run(self.ids[:,:3],policy=p).logits
        torch.testing.assert_close(full[:,:3],prefix,atol=1e-6,rtol=1e-6)
    def test_hooks_restored_on_budget_error(self):
        with self.assertRaises(BudgetExceeded):
            self.engine.run(self.ids,policy=self.policy,meter=Meter(Budget(layer_calls=4)))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
        self.engine.run(self.ids)
    def test_no_state_leak_or_cross_request_ffn_cache(self):
        p=replace(self.policy,cells=CellPolicy(width=8),exact_ffn_cache=True)
        a=self.engine.run(self.ids,policy=p,request_id="A").logits
        self.engine.run(self.ids+1,policy=p,request_id="B")
        b=self.engine.run(self.ids,policy=p,request_id="A").logits
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        self.engine.reset_request();self.assertFalse(self.engine._row_caches)
    def test_cache_input_is_rejected(self):
        with self.assertRaises(ValueError):self.engine.run(self.ids,past_key_values=object())
    def test_conservative_cell_messages_preserve_sum(self):
        messages=torch.randn(5,6,16)
        out,trace=conservative_discussion(messages,rounds=4,neighbors=3,mix=.4)
        torch.testing.assert_close(messages.sum(1),out.sum(1),atol=1e-6,rtol=1e-6)
        self.assertFalse(torch.equal(out,messages));self.assertLessEqual(trace[-1],trace[0])
    def test_bound_contains_actual_skipped_contribution(self):
        bank=FrozenCellBank(self.model.model.layers[0].mlp,8)
        x=torch.randn(5,16);bounds=bank.bounds(x)
        for cid in range(bank.bank.count):
            actual=bank.bank.body(x,torch.full((len(x),),cid,dtype=torch.long))
            self.assertTrue((actual.norm(dim=-1)<=bounds[:,cid]+1e-6).all())
    def test_unmet_tail_bound_falls_back_to_dense(self):
        donor=self.model.model.layers[0].mlp;bank=FrozenCellBank(donor,8)
        x=torch.randn(5,16)
        out,trace=bank.run(x,CellPolicy(width=8,seed=1,max_cells=1,mode="bounded"))
        torch.testing.assert_close(out,donor(x),atol=0,rtol=0)
        self.assertEqual(trace["dense_fallback_tokens"],5)
    def test_observe_cells_keep_donor_and_execute_recruitment(self):
        p=replace(self.policy,cells=CellPolicy(width=8,seed=1,max_cells=4,mode="observe"))
        a=self.engine.run(self.ids,policy=self.policy).logits
        b=self.engine.run(self.ids,policy=p).logits
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        self.assertGreater(sum(t.get("recruited_pairs",0) for t in self.engine.last_trace["cells"]),0)
    def test_quantized_reference_cells(self):
        donor=FFN()
        for attr in ("gate_proj","up_proj","down_proj"):
            setattr(donor,attr,Int4Linear.from_linear(getattr(donor,attr),group_size=8))
        bank=FrozenCellBank(donor,8);x=torch.randn(3,16)
        out,_=bank.run(x,CellPolicy(width=8,seed=4,max_cells=4,mode="bounded"))
        torch.testing.assert_close(out,donor(x),atol=1e-6,rtol=1e-6)
    def test_split_merge_and_zero_pruning(self):
        bank=FrozenCellBank(FFN(),8)
        self.assertEqual(bank.partitions(16),((0,16),(16,32)))
        with torch.no_grad():bank.donor.down_proj.weight[:,:8].zero_()
        self.assertIn(0,bank.zero_cells())
    def test_fast_association_bounded_and_fork_reset(self):
        state=FastAssociations(("a",self.engine.revision,2))
        state.bind(torch.randn(16),torch.randn(16),evidence="measurement")
        x=torch.randn(3,16)
        y=state.apply(x,scope=state.scope,radius=.1)
        self.assertTrue(((y-x).norm(dim=-1)<=.1*x.norm(dim=-1)+1e-6).all())
        other=state.fork();other.reset()
        self.assertTrue(torch.equal(other.apply(x,scope=other.scope,radius=.1),x))
        self.assertFalse(torch.equal(y,x))
        with self.assertRaises(ValueError):state.apply(x,scope=("b",self.engine.revision,2),radius=.1)
    def test_fast_state_inside_same_frozen_route(self):
        state=FastAssociations(("task",self.engine.revision,2))
        state.bind(torch.randn(16),torch.randn(16),evidence="external")
        a=self.engine.run(self.ids,policy=self.policy,request_id="task").logits
        b=self.engine.run(self.ids,policy=replace(self.policy,fast_gain=.3),fast=state,request_id="task").logits
        self.assertFalse(torch.equal(a,b));self.assertTrue(self.engine.unchanged())
    def test_speculative_greedy_equals_target(self):
        a,_=self.engine.generate(self.ids,policy=self.policy,max_new_tokens=6)
        b,stats=self.engine.speculative(self.ids,draft=FrozenPolicy(),target=self.policy,
                                        meter=Meter(),max_new_tokens=6,block=3)
        self.assertTrue(torch.equal(a,b));self.assertFalse(stats["speedup_claim"])
    def test_brainstorm_baseline_fallback_and_merge(self):
        result=self.engine.brainstorm(self.ids,policies=(FrozenPolicy(),self.policy,FrozenPolicy()),meter=Meter(),max_new_tokens=3)
        self.assertEqual(result["selected"],0);self.assertEqual(result["selection"],"donor_fallback")
        self.assertIn(2,result["branches"][0]["equivalent_routes"])
    def test_verifier_cannot_approve_other_tokens(self):
        def bad(tokens):return Outcome(stable_hash([100]),"test",True,"fake","test",True)
        with self.assertRaises(ValueError):
            self.engine.brainstorm(self.ids,policies=(FrozenPolicy(),),meter=Meter(),verifier=bad,max_new_tokens=2)
    def test_freeze_tripwire_catches_mutation(self):
        with torch.no_grad():next(self.model.parameters()).add_(.1)
        with self.assertRaises(RuntimeError):self.engine.run(self.ids)

if __name__=="__main__":unittest.main()
