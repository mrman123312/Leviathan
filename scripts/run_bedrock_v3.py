#!/usr/bin/env python3
"""Frozen Bedrock v3: ARC-Easy + retention + expressive mechanisms, offline and no training."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse,hashlib,html,json,os,random,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def fmt(value,places=3):return 'not measured' if value is None else f'{value:.{places}f}'
def pct(value):return 'not measured' if value is None else f'{100*value:.1f}%'


def render(report):
    rows=[]
    for mode,s in report.get('arc_easy',{}).get('summary',{}).items():
        paired=s.get('paired',{})
        delta=str(paired.get('net_correct_change','—'))
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (
            mode,f'{s["correct"]}/{s["evaluated"]}',pct(s['accuracy']),delta,
            fmt(s['seconds_per_correct_answer']),fmt(s['total_model_seconds']),s['errors']))+'</tr>')
    wiki=[]
    for mode,r in report.get('wikitext',{}).get('modes',{}).items():
        wiki.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in
            (mode,fmt(r.get('mean_token_nll'),5),pct(r.get('relative_nll_change')),
             r.get('loss_gate_pass','not applicable'),r.get('passages',0)))+'</tr>')
    status=html.escape(report.get('status','running'))
    error=html.escape(report.get('error',''))
    summary={k:v for k,v in report.items() if k not in {'arc_easy','wikitext','features'}}
    return '''<!doctype html><html><head><meta charset="utf-8"><title>Leviathan v3 ARC-Easy</title>
<style>body{font:17px system-ui;background:#11151b;color:#e4eaf1;max-width:1180px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #46505b;text-align:left}section{background:#1b222b;padding:22px;margin:18px 0;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1,h2{color:white}a{color:#80c7ff}</style></head><body>
<h1>Leviathan Frozen Bedrock v3</h1><p>Status: <b>'''+status+'''</b></p><p>'''+error+'''</p>
<p>No training. One cached Qwen3-1.7B-Base. One GPU. No automatic promotion.</p>
<section><h2>ARC-Easy: paired 50-question canary</h2><p>Raw completion likelihood is the primary score, matching the earlier canary prompt. This is not full ARC-Easy, nor a claim of an official harness result. Counts and changed-question IDs are preserved in RESULTS.json.</p>
<table><tr><th>Mode</th><th>Correct</th><th>Accuracy</th><th>Net correct change</th><th>Seconds/correct</th><th>Model seconds</th><th>Errors</th></tr>'''+''.join(rows)+'''</table>
<p>Adaptive runs execute their own donor/refinement/exploration calls; all stages are included in their time. Confidence and stability are heuristics, not truth certificates. A net gain can still hide individual regressions: inspect fixed_ids and broken_ids.</p></section>
<section><h2>WikiText retention</h2><p>64 fixed qualifying passages, maximum 256 tokens each. This differs from the earlier 32-passage canary. No thresholds are tuned against these passages. Adaptive MCQ routing is not assigned a made-up language-model loss.</p>
<table><tr><th>Mode</th><th>Token NLL</th><th>Change</th><th>2% loss gate</th><th>Passages</th></tr>'''+''.join(wiki)+'''</table></section>
<section><h2>Depth, branches, cells, neural rule discovery</h2><pre>'''+html.escape(json.dumps(report.get('features',{}),indent=2))+'''</pre></section>
<section><h2>Execution and limitations</h2><pre>'''+html.escape(json.dumps(summary,indent=2))+'''</pre></section>
<p>Raw records: <a href="RESULTS.json">RESULTS.json</a>. Earlier runs are not overwritten.</p></body></html>'''


def cache_wikitext():
    from datasets import Dataset
    cache=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'datasets'
    paths=sorted(cache.glob('Salesforce___wikitext/wikitext-2-raw-v1/*/*/*test.arrow'))
    paths+=sorted(cache.glob('wikitext/wikitext-2-raw-v1/*/*/*test.arrow'))
    for path in paths:
        ds=Dataset.from_file(str(path))
        if 'text' in ds.column_names:return ds,str(path)
    raise RuntimeError('WikiText test cache missing; ARC still runs, retention marked not measured')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--skip-wikitext',action='store_true')
    p.add_argument('--skip-neural-world',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'format':3,'status':'running','training_steps':0,'new_parameters':0,
        'automatic_promotion':False,'errors':[],'features':{},
        'scope':'RTX3060 frozen-weight ARC canary, retention and mechanism tests',
        'limits':['50 ARC test examples, not the full benchmark',
          'No claim that latent branches represent meaningful hypotheses',
          'Donor-input proximity does not prove manifold membership or quality',
          'Logical per-position halting does not compact causal attention rows',
          'Neural discovery has no supplied true-rule catalogue but uses a bounded expression grammar',
          'Unit-test results do not establish GPU accuracy or speed']}
    def save():
        encoded=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)
        temp=args.output/'RESULTS.json.tmp';temp.write_text(encoded,encoding='utf-8')
        temp.replace(args.output/'RESULTS.json')
        (args.output/'RESULTS.html').write_text(render(report),encoding='utf-8')
    def err(stage,exc):
        report['errors'].append({'stage':stage,'error':f'{type(exc).__name__}: {exc}'})
        print(f'{stage}: ERROR: {exc}',flush=True);save()
    save()
    try:
        print('[1/5] Loading cached Qwen on your existing CUDA environment. No installation or model download.',flush=True)
        import torch,transformers
        from transformers import AutoTokenizer,Qwen3ForCausalLM
        from leviathan.consumer.profiles import get_profile
        from leviathan.bedrock.stable_neural import StableFrozenExecutor,StableFrozenPolicy
        from leviathan.bedrock.decisions import StopPolicy,ChoicePolicy
        from leviathan.bedrock.evaluation import default_routes,score_choices,adaptive_choice,grade,summarize_arc,sync
        from leviathan.bedrock.contracts import stable_hash
        profile=get_profile('rtx3060');torch.manual_seed(9307)
        common=dict(revision=profile.revision,local_files_only=True,trust_remote_code=False)
        tokenizer=AutoTokenizer.from_pretrained(profile.repo_id,**common)
        model=Qwen3ForCausalLM.from_pretrained(profile.repo_id,**common,torch_dtype=torch.float16,
            device_map={'':'cuda'},low_cpu_mem_usage=True,attn_implementation='eager').eval()
        engine=StableFrozenExecutor(model,model_id=profile.id,revision=profile.revision)
        device=model.get_input_embeddings().weight.device
        report['model']={'repo':profile.repo_id,'revision':profile.revision,'stage':profile.stage,
                         'precision':'FP16','torch':torch.__version__,'transformers':transformers.__version__,
                         'gpu':torch.cuda.get_device_name(0)}
        routes=default_routes();choice_policy=ChoicePolicy()
        report['fixed_policies']={k:asdict(v) for k,v in routes.items()}
        report['choice_policy']=asdict(choice_policy)
        report['policy_sha256']=stable_hash([report['fixed_policies'],report['choice_policy']])
        report['policy_fit_steps']=0
        modes=[*routes,'adaptive']
        warm=tokenizer('The capital of France is',return_tensors='pt').input_ids.to(device)
        for name,policy in routes.items():
            print(f'  Warming {name}',flush=True);engine.run(warm,policy=policy,request_id='warmup');sync(device)
        print('[2/5] ARC-Easy: 50 questions, all modes, paired and no answer-key routing.',flush=True)
        path=ROOT/'data/arc_easy_50.json'
        if path.is_file():arc=json.loads(path.read_text(encoding='utf-8'))
        else:
            from export_arc_canary import export
            arc=export(args.output/'arc_easy_50.json')
        encoded=json.dumps(arc['examples'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        if hashlib.sha256(encoded).hexdigest()!=arc['examples_sha256']:raise ValueError('ARC dataset hash mismatch')
        examples=arc['examples']
        report['arc_easy']={'protocol':'Question: {question}\\nAnswer: + full choice text; no chat template; sum log probabilities',
            'selection':arc['selection'],'dataset_sha256':arc['examples_sha256'],
            'source':arc.get('source',{}),'records':{k:[] for k in modes},'summary':{},
            'previous_canary_score_is_reference_only':.72}
        records=report['arc_easy']['records']
        # Deterministic interleaving reduces fixed-mode-order temperature effects.
        schedule=random.Random(9307)
        for i,item in enumerate(examples):
            ordered=modes.copy();schedule.shuffle(ordered)
            for mode in ordered:
                print(f'ARC-Easy {i+1}/{len(examples)} | {mode}',flush=True)
                start=time.perf_counter()
                try:
                    def evaluate(regime):
                        return score_choices(engine,tokenizer,item['question'],item['choices']['text'],
                            policy=routes[regime],request_id=f'arc:{i}:{mode}:{regime}')
                    scored=adaptive_choice(evaluate,policy=choice_policy) if mode=='adaptive' else evaluate(mode)
                    # This is the first place answerKey enters the pipeline.
                    result=grade(scored,item['choices']['label'],item['answerKey'])
                    result.update(id=item['id'],status='completed',wall_seconds=time.perf_counter()-start)
                    records[mode].append(result)
                except Exception as exc:
                    records[mode].append({'id':item['id'],'status':'error','error':str(exc)})
                    err(f'ARC {item["id"]} {mode}',exc)
                report['arc_easy']['summary']={k:summarize_arc(v,records['donor'] if k!='donor' else None)
                                                for k,v in records.items()}
                save()
            scores=' | '.join(f'{m}: {sum(r.get("correct",False) for r in records[m])}/{sum(r["status"]=="completed" for r in records[m])}' for m in modes)
            print('  '+scores,flush=True)
        report['arc_easy']['completed']=all(len(r)==50 and all(e['status']=='completed' for e in r) for r in records.values())
        save()
        print('[3/5] WikiText: fixed language-retention sample; no training.',flush=True)
        report['wikitext']={'modes':{},'adaptive_mcq_loss':'not defined; not claimed','standard_corpus_perplexity':False}
        if not args.skip_wikitext:
            try:
                ds,cache=cache_wikitext();passages=[]
                for row in ds:
                    text=row['text']
                    if len(text.strip())<160 or text.strip().startswith('='):continue
                    ids=tokenizer.encode(text,add_special_tokens=False)[:256]
                    if len(ids)>=32:passages.append(ids)
                    if len(passages)==64:break
                if len(passages)!=64:raise ValueError('Insufficient WikiText passages')
                report['wikitext'].update(cache=cache,token_manifest_sha256=stable_hash(passages),passages=64,
                    max_tokens=256,excluded_from_optimizer=True,thresholds_fitted=False)
                for mode,policy in routes.items():
                    loss_sum=0.;count=0;losses=[];seconds=0.;nonfinite=0
                    for index,ids in enumerate(passages):
                        print(f'WikiText {index+1}/64 | {mode}',flush=True)
                        batch=torch.tensor([ids],device=device,dtype=torch.long)
                        with torch.inference_mode():
                            sync(device);t=time.perf_counter()
                            result=engine.run(batch,policy=policy,request_id=f'wiki:{mode}:{index}')
                            loss=torch.nn.functional.cross_entropy(result.logits[0,:-1].float(),batch[0,1:],reduction='sum')
                            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite WikiText loss')
                            sync(device);seconds+=time.perf_counter()-t
                            value=float(loss);n=len(ids)-1;loss_sum+=value;count+=n;losses.append(value/n)
                            nonfinite+=engine.last_trace.get('nonfinite_replay_fallbacks',0)
                            del result,loss
                    measured=loss_sum/count
                    baseline=report['wikitext']['modes'].get('donor',{}).get('mean_token_nll',measured)
                    relative=measured/baseline-1
                    report['wikitext']['modes'][mode]={'mean_token_nll':measured,'relative_nll_change':relative,
                        'loss_gate_pass':relative<=.02,'passages':64,'predicted_tokens':count,
                        'model_seconds':seconds,'nonfinite_replay_fallbacks':nonfinite,'passage_losses':losses}
                    save()
            except Exception as exc:err('WikiText retention',exc)
        else:report['wikitext']['status']='skipped_explicitly'
        print('[4/5] Prediction stopping, signed latent branches and activation-conditioned cells.',flush=True)
        feature_policies={
            'prediction_stop_4':StableFrozenPolicy(passes=4,gain=.06,reentry_radius=.05,prediction_stop=StopPolicy()),
            'latent_context_plus':StableFrozenPolicy(passes=2,gain=.06,reentry_radius=.05,
                branch_direction='orthogonal_context',branch_mix=.6,branch_sign=1),
            'latent_context_minus':StableFrozenPolicy(passes=2,gain=.06,reentry_radius=.05,
                branch_direction='orthogonal_context',branch_mix=.6,branch_sign=-1)}
        feature_rows=[]
        for text in ('The capital of France is','A prime number is','Water freezes when'):
            ids=tokenizer(text,return_tensors='pt').input_ids.to(device)
            baseline=engine.run(ids).logits.detach().float().cpu()
            for name,policy in feature_policies.items():
                print(f'Feature | {name} | {text}',flush=True)
                try:
                    with torch.inference_mode():
                        out=engine.run(ids,policy=policy,request_id=name).logits.detach().float().cpu()
                    feature_rows.append({'mode':name,'prompt':text,'max_logit_change':float((out-baseline).abs().max()),
                        'argmax_token':int(out[0,-1].argmax()),'trace':engine.last_trace})
                except Exception as exc:err('feature '+name,exc)
        report['features']['routes']=feature_rows;save()
        from leviathan.bedrock.activation_cells import ActivationCellBank,ActivationPolicy
        donor_ffn=model.model.layers[-1].mlp
        bank=ActivationCellBank(donor_ffn,128)
        captured=[]
        def capture(module,arguments):captured.append(arguments[0].detach())
        handle=donor_ffn.register_forward_pre_hook(capture)
        try:engine.run(warm)
        finally:handle.remove()
        with torch.inference_mode():
            out,cell_stats=bank.analyze(captured[-1],ActivationPolicy(width=128,seed=4,max_cells=16))
            original=donor_ffn(captured[-1])
            cell_stats['observe_max_output_difference']=float((out.float()-original.float()).abs().max())
        report['features']['activation_cell_relevance']=cell_stats
        del captured,out,original,bank;save()
        print('[5/5] Same-Qwen rule proposals and evidence-driven revision (no hidden catalogue).',flush=True)
        if not args.skip_neural_world:
            try:
                from leviathan.bedrock.runtime import BedrockRuntime
                from leviathan.bedrock.neural_discovery import discover_neural
                core=BedrockRuntime(model_id=profile.id,neural=engine,memory_journal=str(args.output/'neural-memory.jsonl'))
                # The host owns truth; the proposer only gets domain and chosen observations.
                # No selection/tuning is performed on the outcome of this fixed demo.
                truth=lambda x:(3*x+4)%11
                world=discover_neural(core,scope='v3-demo-device',problem='Infer an unfamiliar input/output device',
                    domain=tuple(range(11)),observe=truth,tokenizer=tokenizer,progress=lambda s:print(s,flush=True))
                report['features']['neural_world']=world
                if world['status']=='empirically_validated_candidate':
                    used={r['x'] for r in world['observations']};unseen=[a for a in range(11) if a not in used]
                    reloaded=BedrockRuntime(model_id=profile.id,memory_journal=str(args.output/'neural-memory.jsonl'))
                    pred=[reloaded.transfer(scope='v3-demo-device',action=a)['prediction'] for a in unseen]
                    world['external_transfer_evaluation']={'unseen_queries':len(unseen),
                        'correct':sum(v==truth(a) for v,a in zip(pred,unseen)),
                        'outcomes_were_not_supplied_to_proposer':True}
            except Exception as exc:err('neural world',exc)
        else:report['features']['neural_world']={'status':'skipped_explicitly'}
        report['frozen_version_tripwire_passed']=engine.unchanged()
        report['tripwire_is_full_weight_hash']=False
        report['status']='completed_with_errors' if report['errors'] else 'completed'
        save();print(f'DONE: {args.output / "RESULTS.html"}',flush=True)
        return 0 if report['arc_easy']['completed'] and not report['errors'] else 2
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}'
        report['traceback']=traceback.format_exc();save();print(report['traceback'],flush=True)
        return 1

if __name__=='__main__':raise SystemExit(main())
