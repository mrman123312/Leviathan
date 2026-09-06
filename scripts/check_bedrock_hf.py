#!/usr/bin/env python3
"""Native tiny Qwen integration. Random weights, zero training, no language scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    import torch
    import transformers
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from leviathan.bedrock.neural import FrozenExecutor,FrozenPolicy
    from leviathan.bedrock.contracts import Meter
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"evidence/bedrock/native-hf.json")
    args=parser.parse_args()
    torch.manual_seed(623);torch.set_num_threads(1)
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128)
    cfg._attn_implementation="eager"
    model=Qwen3ForCausalLM(cfg).eval()
    engine=FrozenExecutor(model,model_id="qwen3-random-test",revision="random-seed623")
    before={k:v.clone() for k,v in model.state_dict().items()}
    ids=torch.tensor([[1,5,7,9,11]])
    with torch.inference_mode():raw=model(ids,use_cache=False).logits
    neutral=engine.run(ids).logits
    torch.testing.assert_close(raw,neutral,atol=0,rtol=0)
    policies=(FrozenPolicy(start=1,end=2,passes=3,gain=.15),
        FrozenPolicy(start=1,end=2,passes=3,gain=.15,feedback="anchored_difference"))
    results=[]
    for policy in policies:
        out=engine.run(ids,policy=policy).logits
        prefix=engine.run(ids[:,:3],policy=policy).logits
        torch.testing.assert_close(out[:,:3],prefix,atol=1e-5,rtol=1e-5)
        assert not torch.equal(neutral,out)
        results.append({"feedback":policy.feedback,"prefix_max_diff":float((out[:,:3]-prefix).abs().max()),
                        "output_changed":True})
    direct,_=engine.generate(ids,policy=policies[0],max_new_tokens=5)
    spec,counters=engine.speculative(ids,draft=FrozenPolicy(),target=policies[0],meter=Meter(),max_new_tokens=5)
    assert torch.equal(direct,spec)
    unchanged=all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert unchanged
    report={"scope":"native tiny randomly initialized Qwen3; no pretrained checkpoint",
            "torch":torch.__version__,"transformers":transformers.__version__,"training_steps":0,
            "neutral_max_logit_difference":float((neutral-raw).abs().max()),"routes":results,
            "same_model_greedy_speculation_equal":True,"donor_tensors_bytewise_equal":unchanged,
            "new_learned_parameters":0,"full_1_7b_run":False,"gpu_run":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=="__main__":main()
