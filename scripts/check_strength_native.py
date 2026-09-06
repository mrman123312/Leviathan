#!/usr/bin/env python3
"""Native tiny random Qwen integration. No pretrained capability claim."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from leviathan.bedrock.stable_neural import StableFrozenExecutor
from leviathan.strength.neural import NeuralFabric,NeuralRoute,TaskWorkspace
from leviathan.strength.proposer import QwenProposer
from leviathan.strength.contracts import ArcTask,Example

def main():
    torch.manual_seed(372);torch.set_num_threads(1)
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=256,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256)
    cfg._attn_implementation='eager'
    model=Qwen3ForCausalLM(cfg).eval();engine=StableFrozenExecutor(model,model_id='one',revision='random-372')
    fabric=NeuralFabric(engine);ids=torch.tensor([[1,5,7,9,11]])
    before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.inference_mode():
        base=model(ids,use_cache=False).logits
        with fabric.use(NeuralRoute()):neutral=model(ids,use_cache=False).logits
        assert torch.equal(base,neutral)
        workspace=TaskWorkspace('random-372',1,'support-only',torch.randn(2,32))
        trials=[]
        for route in (NeuralRoute('task_state',layer=1),NeuralRoute('damped_band',layer=1),NeuralRoute('cell_ablation',layer=1,cell_width=32)):
            with fabric.use(route,workspace=workspace):full=model(ids,use_cache=False).logits
            with fabric.use(route,workspace=workspace):prefix=model(ids[:,:3],use_cache=False).logits
            torch.testing.assert_close(full[:,:3],prefix,atol=1e-5,rtol=1e-5)
            assert torch.isfinite(full).all() and not torch.equal(full,base)
            trials.append({'kind':route.kind,'prefix_max_diff':float((full[:,:3]-prefix).abs().max()),'finite':True})
        # Tokenizer fixture exercises native generate, not real text ability.
        class Tokenizer:
            eos_token_id=96
            def __call__(self,text,**kw):return type('Tokens',(),{'input_ids':ids})()
            def decode(self,tokens,**kw):return 'rot90(x)'
        proposer=QwenProposer(engine,Tokenizer(),max_new_tokens=4)
        task=ArcTask('fixture',(Example(((1,2),),((2,),(1,))),),(((3,4),),))
        proposals=proposer.propose(task,rejected=[],view='rows',count=4,round_index=0)
        assert len(proposals)==4 and proposer.total_calls>0
        # Exercise every candidate slot's demonstration gate on native layers.
        _,gate=fabric.select_on_demonstrations([(ids,3),(ids+1,3)],workspace=workspace)
        assert len([t for t in gate['trials'] if t['route']['kind']=='task_state'])==2
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    report={'scope':'native tiny random Qwen; tokenizer fixture; not pretrained ARC performance',
        'neutral_exact':True,'all_donor_tensors_unchanged':True,'training_steps':0,'new_learned_parameters':0,
        'trials':trials,'native_generate_executed':True,'actual_forward_calls_counted':proposer.total_calls,
        'hooks_restored':True,'all_slots_support_gated':True,'gpu_run':False}
    out=ROOT/'evidence/strength/native-qwen.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
