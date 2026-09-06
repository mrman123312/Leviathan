from __future__ import annotations
import importlib.util
import unittest
from dataclasses import replace
HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from test_bedrock_neural import FFN,TinyLM
    from leviathan.bedrock.cells import CellPolicy,CellState,FrozenCellBank
    from leviathan.bedrock.neural import FrozenExecutor,FrozenPolicy

@unittest.skipUnless(HAS_TORCH,"Optional PyTorch")
class BedrockCellStateTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(728);torch.set_num_threads(1)
    def test_cell_state_updates_without_parameter_writes(self):
        donor=FFN().eval();bank=FrozenCellBank(donor,8);x=torch.randn(3,16)
        state=CellState("same")
        policy=CellPolicy(width=8,seed=2,max_cells=4,mode="observe")
        original={k:v.clone() for k,v in donor.state_dict().items()}
        with torch.inference_mode():
            _,first=bank.run(x,policy,state=state,scope="same")
            _,second=bank.run(x,policy,state=state,scope="same")
        self.assertTrue(first["local_state_updated"]);self.assertTrue(second["seed_route_reused"])
        self.assertGreater(float(state.moments[...,3].max()),1.)
        self.assertTrue(all(torch.equal(v,donor.state_dict()[k]) for k,v in original.items()))
    def test_state_scope_failure_and_reset(self):
        bank=FrozenCellBank(FFN(),8);state=CellState("first")
        with self.assertRaises(ValueError):bank.run(torch.randn(2,16),CellPolicy(width=8,mode="observe"),state=state,scope="second")
        state.reset();self.assertIsNone(state.moments)
    def test_anchored_delta_changes_frozen_computation_and_preserves_neutral(self):
        model=TinyLM();engine=FrozenExecutor(model,model_id="one",revision="test")
        ids=torch.tensor([[1,3,5,9]])
        base=engine.run(ids).logits
        policy=FrozenPolicy(passes=3,gain=.2,feedback="anchored_difference")
        changed=engine.run(ids,policy=policy).logits
        zero=engine.run(ids,policy=replace(policy,gain=0)).logits
        self.assertFalse(torch.equal(base,changed));self.assertTrue(torch.equal(base,zero))
        torch.testing.assert_close(changed[:,:2],engine.run(ids[:,:2],policy=policy).logits,atol=1e-6,rtol=1e-6)
    def test_peer_discussion_can_change_recruitment_and_finishes_after_recruits(self):
        changed = False
        for seed in range(32):
            torch.manual_seed(seed)
            bank=FrozenCellBank(FFN(),4)
            x=torch.randn(2,16)
            p=CellPolicy(width=4,seed=2,max_cells=3,mode="observe",rounds=2)
            with torch.inference_mode():
                _,a=bank.run(x,replace(p,message_mix=0))
                _,b=bank.run(x,replace(p,message_mix=.5))
            self.assertEqual(len(b["disagreement_proxy"]),2)
            changed = changed or a["routes"]!=b["routes"]
        self.assertTrue(changed)

    def test_mutable_cell_state_cannot_enter_exact_cache(self):
        with self.assertRaises(ValueError):FrozenPolicy(cells=CellPolicy(mode="observe"),exact_ffn_cache=True)

if __name__=="__main__":unittest.main()
