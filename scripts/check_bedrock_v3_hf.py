#!/usr/bin/env python3
"""Native tiny-Qwen v3 integration; frozen random fixtures, not language benchmarks."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    import torch,transformers
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from leviathan.bedrock.stable_neural import StableFrozenExecutor,StableFrozenPolicy
    from leviathan.bedrock.decisions import StopPolicy
    from leviathan.bedrock.activation_cells import ActivationCellBank,ActivationPolicy
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'evidence/bedrock-v3/native-hf.json')
    args=p.parse_args();torch.manual_seed(781);torch.set_num_threads(1)
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128)
    cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    engine=StableFrozenExecutor(model,model_id='native-tiny',revision='random-seed-781')
    ids=torch.tensor([[1,3,5,7,9]])
    with torch.inference_mode():baseline=model(ids,use_cache=False).logits
    neutral=engine.run(ids).logits
    torch.testing.assert_close(baseline,neutral,atol=0,rtol=0)
    routes={
        'predictive':StableFrozenPolicy(start=1,end=3,passes=4,gain=.06,prediction_stop=StopPolicy()),
        'plus':StableFrozenPolicy(start=1,end=3,passes=2,gain=.06,branch_direction='orthogonal_context',branch_mix=.6),
        'minus':StableFrozenPolicy(start=1,end=3,passes=2,gain=.06,branch_direction='orthogonal_context',branch_mix=.6,branch_sign=-1)}
    outputs={};records={}
    for name,policy in routes.items():
        out=engine.run(ids,policy=policy).logits
        outputs[name]=out.clone()
        trace=engine.last_trace
        early=engine.run(ids[:,:3],policy=policy).logits
        torch.testing.assert_close(out[:,:3],early,atol=1e-5,rtol=1e-5)
        records[name]={'finite':bool(torch.isfinite(out).all()),
            'prefix_max_error':float((out[:,:3]-early).abs().max()),
            'extra_layer_calls':trace['extra_layer_calls'],
            'prediction_head_calls':trace['prediction_head_calls']}
    assert not torch.equal(outputs['plus'],outputs['minus'])
    donor=model.model.layers[-1].mlp
    x=torch.randn(4,32)
    with torch.inference_mode():
        out,stats=ActivationCellBank(donor,16).analyze(x,ActivationPolicy(width=16,seed=2,max_cells=2))
        torch.testing.assert_close(out,donor(x),atol=0,rtol=0)
    unchanged=all(torch.equal(v,model.state_dict()[k]) for k,v in before.items());assert unchanged
    report={'scope':'random tiny native Qwen3; not a pretrained Qwen benchmark','torch':torch.__version__,
        'transformers':transformers.__version__,'training_steps':0,'new_parameters':0,
        'neutral_logit_difference':float((neutral-baseline).abs().max()),'routes':records,
        'signed_branch_outputs_differ':True,'activation_observe_parity':True,
        'activation_bound_tightening_factor':stats['median_bound_tightening_factor'],
        'weights_bytewise_unchanged':unchanged,'gpu_run':False,'full_checkpoint_loaded':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
