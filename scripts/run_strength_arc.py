#!/usr/bin/env python3
"""Strength-first ARC-AGI-1 inference. No training, no hidden-answer search feedback."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from leviathan.strength.contracts import ArcTask,SearchConfig,digest
from leviathan.strength.runtime import StrengthRuntime
from leviathan.strength.evaluation import select_ids,score
from leviathan.strength.programs import parse


def page(report):
    summaries=[]
    for mode,value in report.get('scores',{}).items():
        summaries.append(f'<tr><td>{html.escape(mode)}</td><td>{value["task_pass1"]}/{value["tasks"]}</td>'
          f'<td>{value["task_pass2"]}/{value["tasks"]}</td><td>{100*value["task_accuracy2"]:.1f}%</td></tr>')
    status=html.escape(report.get('status','running'))
    body=f'''<!doctype html><html><meta charset="utf-8"><title>Leviathan ARC-AGI-1 strength</title>
<style>body{{font:17px system-ui;max-width:1150px;margin:32px auto;padding:0 24px}}td,th{{padding:10px;border-bottom:1px solid #ccc;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}section{{padding:18px;border:1px solid #bbb;margin:16px 0}}</style>
<h1>Leviathan: ARC-AGI-1 strength research</h1><p>Status: <b>{status}</b></p>
<p>Exact grid transformations, not ARC-Easy. No neural training. One frozen Qwen in hybrid mode; symbolic mode makes zero neural calls.</p>
<p>A fit to the visible examples is not a solved test. Labels are opened only after the batch predictions are saved.</p>
<table><tr><th>Mode</th><th>Tasks pass@1</th><th>Tasks pass@2</th><th>Pass@2</th></tr>{''.join(summaries)}</table>
<p>Public results are not a private ARC leaderboard result. Two attempts per query; every query in a task must match exactly.</p>'''
    for identifier,result in report.get('predictions',{}).get('hybrid',report.get('predictions',{}).get('symbolic',{})).items():
        body+=f'<section><h2>{html.escape(identifier)}</h2><p>{html.escape(result.get("status","error"))}</p><pre>'+html.escape(json.dumps({k:result.get(k) for k in ('selected_programs','selected_sources','support_solutions','budget','activation_trials','error')},indent=2))+'</pre></section>'
    metadata={k:v for k,v in report.items() if k not in ('predictions',)}
    return body+'<h2>Protocol and evidence</h2><pre>'+html.escape(json.dumps(metadata,indent=2))+'</pre><p>See RESULTS.json and PREDICTIONS_SEALED.json for raw evidence.</p></html>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=('training','evaluation'),default='evaluation')
    p.add_argument('--limit',type=int,default=50)
    p.add_argument('--mode',choices=('symbolic','hybrid'),default='hybrid')
    p.add_argument('--max-candidates',type=int,default=60000)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--beam',type=int,default=24)
    p.add_argument('--task-seconds',type=float,default=240.)
    p.add_argument('--neural-rounds',type=int,default=3)
    p.add_argument('--activation-trials',action='store_true')
    p.add_argument('--symbolic-control',action='store_true')
    p.add_argument('--donor-control',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    output=args.output or ROOT/'results_strength'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    config=SearchConfig(max_depth=args.depth,beam_per_view=args.beam,max_candidates=args.max_candidates,
        max_seconds=args.task_seconds,neural_rounds=args.neural_rounds if args.mode=='hybrid' else 0,
        activation_trials=args.activation_trials)
    datadir=ROOT/'data/arc_agi_1';manifest=json.loads((datadir/'MANIFEST.json').read_text())
    task_file=datadir/(args.split+'_tasks.json');raw=task_file.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['files'][task_file.name]['sha256']:raise RuntimeError('Task file hash mismatch')
    tasks=json.loads(raw);ids=select_ids(tasks,args.limit)
    report={'status':'running','scope':f'ARC-AGI-1 public {args.split}, {args.limit} fixed hash-selected tasks',
        'dataset':manifest,'task_ids':ids,'configuration':asdict(config),'policy_hash':config.fingerprint,
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'src/leviathan/strength').glob('*.py'))},
        'training_steps':0,'new_learned_parameters':0,'goal':'ARC-AGI-1 exact grid generalization',
        'labels_opened':False,'neural_full_model_run':False,'predictions':{},'scores':{},'errors':[]}
    def save():
        temporary=output/'RESULTS.json.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(output/'RESULTS.json');(output/'RESULTS.html').write_text(page(report),encoding='utf-8')
    save()
    try:
        executor=tokenizer=None
        if args.mode=='hybrid':
            print('Loading your existing Qwen3-1.7B-Base cache on CUDA. No install, download or training.',flush=True)
            import torch
            from transformers import AutoTokenizer,Qwen3ForCausalLM
            from leviathan.bedrock.stable_neural import StableFrozenExecutor
            repo='Qwen/Qwen3-1.7B-Base';revision='ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
            common=dict(revision=revision,local_files_only=True,trust_remote_code=False)
            tokenizer=AutoTokenizer.from_pretrained(repo,**common)
            model=Qwen3ForCausalLM.from_pretrained(repo,**common,dtype=torch.float16,device_map={'':'cuda'},
                                                  attn_implementation='eager').eval()
            executor=StableFrozenExecutor(model,model_id=repo,revision=revision)
            report['neural_full_model_run']=True
            report['model']={'repository':repo,'revision':revision,'gpu':torch.cuda.get_device_name(0),'precision':'FP16'}
        modes=['symbolic',args.mode] if args.mode=='hybrid' and args.symbolic_control else [args.mode]
        if executor and args.donor_control:modes=['donor_grid']+modes
        for mode in modes:
            report['predictions'][mode]={}
            for i,identifier in enumerate(ids):
                print(f'ARC-AGI-1 {i+1}/{len(ids)} | {mode} | {identifier}',flush=True)
                task=ArcTask.from_public(identifier,tasks[identifier],args.split)
                if mode=='donor_grid':
                    from leviathan.strength.proposer import direct_grid_control
                    try:
                        report['predictions'][mode][identifier]=direct_grid_control(executor,tokenizer,task,progress=lambda s:print(s,flush=True))
                    except Exception as exc:
                        report['predictions'][mode][identifier]={'status':'error','attempts':[],'error':str(exc)}
                        report['errors'].append({'mode':mode,'task':identifier,'error':str(exc),'traceback':traceback.format_exc()})
                    save();continue
                # Fresh task state: no earlier evaluation answers/skills are carried forward.
                runtime=StrengthRuntime(model_id=executor.model_id if executor and mode=='hybrid' else 'symbolic-control',
                    neural=executor if mode=='hybrid' else None,tokenizer=tokenizer if mode=='hybrid' else None,
                    config=config,progress=lambda s:print(s,flush=True))
                try:
                    result=runtime.solve_arc(task)
                    if executor and mode=='hybrid':result['frozen_version_tripwire_passed']=executor.unchanged()
                    report['predictions'][mode][identifier]=result
                    print(f'  {result["status"]}; exact support fits={result["support_solutions"]}; candidates={result["budget"]["candidates"]}',flush=True)
                except Exception as e:
                    report['predictions'][mode][identifier]={'status':'error','error':f'{type(e).__name__}: {e}','attempts':[]}
                    report['errors'].append({'mode':mode,'task':identifier,'error':str(e),'traceback':traceback.format_exc()})
                    print(traceback.format_exc(),flush=True)
                finally:
                    if executor:executor.reset_request()
                save()
        # Raw-program-proposal control reuses the exact proposals already paid for,
        # selecting the first two distinct syntactically valid neural programs, with
        # NO demonstration-fit filtering. This is not a direct-grid Qwen baseline.
        if 'hybrid' in report['predictions']:
            raw_control={}
            for identifier,result in report['predictions']['hybrid'].items():
                task=ArcTask.from_public(identifier,tasks[identifier],args.split);preds=[];seen=set()
                for item in result.get('neural_proposals',[]):
                    try:
                        program=parse(item['text']);values=tuple(program.run(x) for x in task.queries)
                        if values not in seen:preds.append(values);seen.add(values)
                    except (ValueError,SyntaxError):continue
                    if len(preds)==2:break
                raw_control[identifier]={'attempts':[[s[q] for s in preds] for q in range(len(task.queries))],
                    'protocol':'first_two_parseable_neural_programs; no support filtering; shared proposals, not a timing control'}
            report['predictions']['raw_neural_programs']=raw_control
        seal=output/'PREDICTIONS_SEALED.json';seal.write_text(json.dumps(report['predictions'],sort_keys=True),encoding='utf-8')
        report['sealed_predictions_sha256']=hashlib.sha256(seal.read_bytes()).hexdigest()
        print('Predictions sealed. Opening evaluator labels now; no further search or adaptation.',flush=True)
        label_file=datadir/(args.split+'_labels.json');label_raw=label_file.read_bytes()
        if hashlib.sha256(label_raw).hexdigest()!=manifest['files'][label_file.name]['sha256']:raise RuntimeError('Label file hash mismatch')
        labels=json.loads(label_raw);report['labels_opened']=True
        for mode,predictions in report['predictions'].items():report['scores'][mode]=score(predictions,labels,ids)
        report['status']='completed_with_errors' if report['errors'] else 'completed';save()
        print(json.dumps(report['scores'],indent=2),flush=True)
        print('RESULTS:',output/'RESULTS.html',flush=True)
        if os.name=='nt':os.startfile(output/'RESULTS.html')
        return 0
    except Exception as e:
        report['status']='failed';report['errors'].append({'error':str(e),'traceback':traceback.format_exc()});save()
        print(traceback.format_exc(),flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
