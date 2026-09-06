#!/usr/bin/env python3
"""No-training finite-world experiments. Not a language/AGI benchmark."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from leviathan.bedrock.runtime import BedrockRuntime
from leviathan.bedrock.world import catalogue,Rule,VersionSpace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"evidence/bedrock/finite-worlds.json")
    parser.add_argument("--instances",type=int,default=12)
    args=parser.parse_args()
    if not 1<=args.instances<=48:parser.error("instances must be in [1,48]")
    rng=random.Random(602)
    report={"scope":"declared finite deterministic grammars; not general novel-world competence",
            "seed":602,"training_steps":0,"neural_model_calls":0,"families":{},
            "language_benchmarks":{"ARC-Easy":None,"WikiText":None,"GSM8K":None},
            "evaluation_protocol":"explore then fresh-validation then unseen-action transfer; no transfer labels enter runtime"}
    started=time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        journal=str(Path(tmp)/"memory.jsonl")
        core=BedrockRuntime(model_id="frozen-semantic-identity-control",memory_journal=journal)
        for family in ("affine_mod11","bit_permutation","boolean_circuit"):
            pool=catalogue(family);unique=VersionSpace(pool).rules
            selected=rng.sample(range(len(unique)),min(args.instances,len(unique)))
            rows=[]
            for index in selected:
                truth=unique[index]
                scope=f"{family}:instance:{index}"
                t=time.perf_counter()
                out=core.discover(problem="Infer a device rule from chosen interventions",scope=scope,
                                  rules=pool,observe=truth.predict)
                row={"instance":index,"status":out["status"],"seconds":time.perf_counter()-t,
                     "experiments":out["budget"]["used"].get("environment_steps",0)}
                if out["status"]=="validated_in_declared_domain":
                    used={p["action"] for p in out["transcript"]}|{p["action"] for p in out["validation"]}
                    unseen=[a for a in truth.domain if a not in used]
                    # New process-like runtime reload: no old task/prompt/state is passed.
                    reloaded=BedrockRuntime(model_id=core.model_id,memory_journal=journal)
                    predictions=[reloaded.transfer(scope=scope,action=a)["prediction"] for a in unseen]
                    correct=sum(p==truth.predict(a) for p,a in zip(predictions,unseen))
                    row.update({"unseen_queries":len(unseen),"transfer_correct":correct,
                                "discovery_queries":len(out["transcript"]),
                                "graph_complete":out["graph_complete"]})
                else:row["reason"]=out["reason"]
                # Same grammar, fixed action order comparator. No learned neural prior.
                sequential=VersionSpace(pool);count=0
                for action in sequential.domain:
                    if len(sequential.rules)==1:break
                    sequential.observe(action,truth.predict(action),evidence_id=f"seq-{action}");count+=1
                row["fixed_order_discovery_queries"]=count
                rows.append(row)
                print(f"{family}: {len(rows)}/{len(selected)}; {row['status']}; observations={row['experiments']}",flush=True)
            successful=[r for r in rows if r["status"]=="validated_in_declared_domain"]
            report["families"][family]={"instances":len(rows),"validated":len(successful),
                "unseen_queries":sum(r.get("unseen_queries",0) for r in rows),
                "transfer_correct":sum(r.get("transfer_correct",0) for r in rows),
                "mean_discovery_queries":statistics.mean(r["discovery_queries"] for r in successful) if successful else None,
                "mean_total_observations":statistics.mean(r["experiments"] for r in rows),
                "mean_fixed_order_queries":statistics.mean(r["fixed_order_discovery_queries"] for r in rows),
                "records":rows}
    report["elapsed_seconds"]=time.perf_counter()-started
    report["claim"]="No-weight-update hypothesis elimination, experiment selection, persistence and executable skill transfer"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!="families"},indent=2))
    for name,data in report["families"].items():print(name,json.dumps({k:v for k,v in data.items() if k!="records"}))

if __name__=="__main__":main()
