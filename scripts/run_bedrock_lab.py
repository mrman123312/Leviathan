#!/usr/bin/env python3
"""One-click frozen-weight feature lab. Reuses existing environment/cache; no installs."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def report_html(report):
    rows=[]
    for mode,records in report.get("gpu_routes",{}).items():
        for r in records:
            rows.append(f"<tr><td>{html.escape(mode)}</td><td>{html.escape(r['prompt'])}</td>"
                f"<td>{r['max_logit_change']:.6g}</td><td>{r['seconds']:.3f}</td>"
                f"<td>{r['extra_layer_calls']}</td><td>{html.escape(r.get('route_status',''))}</td></tr>")
    return """<!doctype html><html><head><meta charset='utf-8'><title>Leviathan Frozen Bedrock</title>
<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}td,th{padding:10px;border-bottom:1px solid #bbb;text-align:left}pre{white-space:pre-wrap}table{width:100%}</style></head><body>
<h1>Leviathan: frozen-weight feature lab</h1>
<p><b>No training, no new neural parameters, no cloud model.</b> An altered output proves different computation, not improved intelligence.</p>
<p>These are small mechanism/integration tests. Public ARC/WikiText accuracy is not measured by this run. All transformed routes are experimental and the donor remains the fallback.</p>
<h2>GPU route experiments</h2><table><tr><th>Mode</th><th>Prompt</th><th>Max logit change</th><th>Seconds (full prefix)</th><th>Extra layer calls</th><th>Status</th></tr>"""+"".join(rows)+"</table><h2>Execution record</h2><pre>"+html.escape(json.dumps(report,indent=2))+"</pre></body></html>"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--mechanisms-only",action="store_true",help="Run pure algorithm tests without loading any model")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={"training_steps":0,"status":"running","new_parameters":0,"gpu_routes":{},
            "public_language_benchmarks_run":False,"protocol":"small feature smoke; not a capability promotion"}
    def save():
        (args.output/"RESULTS.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        (args.output/"RESULTS.html").write_text(report_html(report),encoding="utf-8")
    try:
        print("[1/4] Running no-training mechanism tests...",flush=True)
        command=[sys.executable,"-m","unittest","discover","-s",str(ROOT/"tests"),"-p","test_bedrock*.py","-v"]
        env=__import__('os').environ.copy();env["PYTHONPATH"]=str(ROOT/"src")
        result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        print(result.stdout,flush=True)
        (args.output/"TESTS.txt").write_text(result.stdout,encoding="utf-8")
        report["unit_tests_passed"]=result.returncode==0
        if result.returncode:raise RuntimeError("Mechanism tests failed; not proceeding to a model")
        print("[2/4] Discovering, saving and reusing rules in finite test worlds...",flush=True)
        subprocess.run([sys.executable,str(ROOT/"scripts/benchmark_bedrock_mechanisms.py"),
            "--output",str(args.output/"finite-worlds.json")],check=True,timeout=180)
        report["finite_worlds"]=json.loads((args.output/"finite-worlds.json").read_text())
        save()
        if not args.mechanisms_only:
            print("[3/4] Loading the already-cached 1.7B base onto CUDA. No download or installation...",flush=True)
            import torch
            from transformers import AutoTokenizer,Qwen3ForCausalLM
            from leviathan.consumer.profiles import get_profile
            from leviathan.bedrock.stable_neural import StableFrozenExecutor,StableFrozenPolicy
            from leviathan.bedrock.cells import CellPolicy
            profile=get_profile("rtx3060")
            common=dict(revision=profile.revision,local_files_only=True,trust_remote_code=False)
            tokenizer=AutoTokenizer.from_pretrained(profile.repo_id,**common)
            model=Qwen3ForCausalLM.from_pretrained(profile.repo_id,**common,torch_dtype=torch.float16,
                        device_map={"":"cuda"},low_cpu_mem_usage=True,attn_implementation="eager").eval()
            engine=StableFrozenExecutor(model,model_id=profile.id,revision=profile.revision)
            report["model"]={"repository":profile.repo_id,"revision":profile.revision,"dtype":"FP16",
                             "gpu":torch.cuda.get_device_name(0),"stage":profile.stage}
            modes={"donor":StableFrozenPolicy(),
                "transported_band_2":StableFrozenPolicy(passes=2,gain=.08,reentry_radius=.06),
                "transported_band_4":StableFrozenPolicy(passes=4,gain=.06,reentry_radius=.05),
                "adaptive_transport_4":StableFrozenPolicy(passes=4,gain=.06,reentry_radius=.05,halt_delta=.01,halt_patience=1),
                "anchored_guarded_4":StableFrozenPolicy(passes=4,gain=.06,reentry_radius=.05,feedback="anchored_difference"),
                "cell_discussion_observe":StableFrozenPolicy(cells=CellPolicy(mode="observe",seed=2,max_cells=4))}
            prompts=("The capital of France is","A prime number is","Water freezes when")
            originals={}
            for name,policy in modes.items():
                report["gpu_routes"][name]=[]
                for text in prompts:
                    print(f"  {name}: {text}",flush=True)
                    ids=tokenizer(text,return_tensors="pt").input_ids.to("cuda")
                    try:
                        _=engine.run(ids,policy=policy,request_id=name+":warmup")
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                        out=engine.run(ids,policy=policy,request_id=name).logits.detach().float()
                        torch.cuda.synchronize();elapsed=time.perf_counter()-started
                        if name=="donor":originals[text]=out.cpu()
                        difference=float((out.cpu()-originals[text]).abs().max())
                        trace=dict(engine.last_trace)
                        report["gpu_routes"][name].append({"prompt":text,"seconds":elapsed,
                            "max_logit_change":difference,"argmax_token":int(out[0,-1].argmax()),
                            "extra_layer_calls":trace["extra_layer_calls"],
                            "route_status":trace.get("route_status","unknown"),
                            "nonfinite_replay_fallbacks":trace.get("nonfinite_replay_fallbacks",0),
                            "max_reentry_relative_l2":max(trace.get("reentry_relative_l2_max",[]) or [0.0]),
                            "peak_vram_gib":torch.cuda.max_memory_allocated()/2**30,
                            "cell_trace":trace["cells"]})
                        if name=="cell_discussion_observe" and difference!=0:
                            report.setdefault("warnings",[]).append("Observe-mode numerical parity differed; investigate before using")
                    except Exception as route_exc:
                        report["gpu_routes"][name].append({"prompt":text,"route_status":"error",
                            "error":f"{type(route_exc).__name__}: {route_exc}","seconds":0.0,
                            "max_logit_change":0.0,"extra_layer_calls":0})
                        report.setdefault("route_errors",[]).append({"mode":name,"prompt":text,
                            "error":f"{type(route_exc).__name__}: {route_exc}"})
                    save()
            print("  Running one raw completion with the frozen band (experimental, unverified)...",flush=True)
            ids=tokenizer(prompts[0],return_tensors="pt").input_ids.to("cuda")
            generated,_=engine.generate(ids,policy=modes["transported_band_2"],max_new_tokens=12,
                                         eos_token_id=tokenizer.eos_token_id)
            report["experimental_completion"]=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True)
            report["frozen_version_tripwire_passed"]=engine.unchanged()
            report["tripwire_is_full_weight_hash"]=False
        else:
            report["gpu_run"]=False
        report["status"]="completed";save()
        print("[4/4] Finished. RESULTS.html and RESULTS.json saved.",flush=True)
    except Exception as exc:
        report["status"]="failed";report["error"]=f"{type(exc).__name__}: {exc}"
        report["traceback"]=traceback.format_exc();save()
        print(report["traceback"],flush=True)
        return 1
    return 0

if __name__=="__main__":raise SystemExit(main())
