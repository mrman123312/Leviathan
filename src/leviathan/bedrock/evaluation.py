"""Paired, no-training evaluation. Gold answers never enter routing or scoring.

ARC uses the inherited Question/Answer raw completion protocol for continuity.
We report raw, token-normalized and character-normalized option scores separately;
only raw canary accuracy is compared with the user's earlier 72%/50 result.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from collections import Counter
import math
import time
from typing import Callable,Sequence
import torch
from .contracts import stable_hash
from .decisions import ChoicePolicy,choose_next
from .stable_neural import StableFrozenPolicy


def default_routes():
    return {
        'donor':StableFrozenPolicy(),
        'refine_2':StableFrozenPolicy(passes=2,gain=.06,reentry_radius=.05,feedback='anchored_difference'),
        'explore_2':StableFrozenPolicy(passes=2,gain=.06,reentry_radius=.05),
        'explore_4':StableFrozenPolicy(passes=4,gain=.06,reentry_radius=.05),
    }


def encode_pair(tokenizer,context:str,answer:str):
    prefix=tokenizer.encode(context,add_special_tokens=False)
    full=tokenizer.encode(context+' '+answer,add_special_tokens=False)
    if not prefix or full[:len(prefix)]!=prefix or len(full)==len(prefix):
        raise ValueError('Tokenizer boundary mismatch; refusing silently altered ARC scoring')
    return prefix,full


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def score_choices(engine,tokenizer,question:str,choices:Sequence[str],*,policy:StableFrozenPolicy,
                  request_id:str,max_tokens:int=1024):
    """No labels argument. Full-prefix causal likelihood, no instruction template."""
    if len(choices)<2 or not question.strip():raise ValueError('Invalid multiple-choice task')
    device=engine.model.get_input_embeddings().weight.device
    scores=[];counts=[];extra=0;fallbacks=0;timed=0.
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    context=f'Question: {question}\nAnswer:'
    try:
        with torch.inference_mode():
            for i,choice in enumerate(choices):
                prefix,full=encode_pair(tokenizer,context,choice)
                if len(full)>max_tokens:raise ValueError('ARC sequence exceeds explicit token limit, no silent truncation')
                ids=torch.tensor([full],device=device,dtype=torch.long)
                sync(device);start=time.perf_counter()
                output=engine.run(ids,policy=policy,request_id=f'{request_id}:{i}')
                logits=output.logits[0,len(prefix)-1:-1].float()
                labels=ids[0,len(prefix):]
                loss=torch.nn.functional.cross_entropy(logits,labels,reduction='sum')
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite choice likelihood')
                sync(device);timed+=time.perf_counter()-start
                scores.append(-float(loss));counts.append(len(labels))
                extra+=engine.last_trace.get('extra_layer_calls',0)
                fallbacks+=engine.last_trace.get('nonfinite_replay_fallbacks',0)
                del output,logits,loss
        return {'scores':scores,'token_counts':counts,'character_counts':[len(c) for c in choices],
                'model_seconds':timed,'model_calls':len(choices),'extra_layer_calls':extra,
                'nonfinite_replay_fallbacks':fallbacks,
                'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                'policy':asdict(policy)}
    finally:engine.reset_request()


def adaptive_choice(evaluate:Callable[[str],dict],*,policy:ChoicePolicy=ChoicePolicy()):
    """Every stage is executed and charged, not reconstructed from precomputed scores.

    evaluate(mode) receives no gold answer. Initial confident answers can remain
    donor-only; unchanged low-novelty predictions can stop after refinement.
    """
    stages=[];events=[]
    donor=evaluate('donor');stages.append(donor)
    choice=choose_next(donor['scores'],policy=policy);events.append(choice)
    selected='donor';result=donor
    if choice['action']!='DIRECT':
        refined=evaluate('refine_2');stages.append(refined);result=refined;selected='refine_2'
        choice=choose_next(refined['scores'],previous=donor['scores'],policy=policy);events.append(choice)
        if choice['action']=='EXPLORE':
            explored=evaluate('explore_4');stages.append(explored);result=explored;selected='explore_4'
    return {**result,'selected_regime':selected,'decisions':events,
            'model_seconds':sum(s['model_seconds'] for s in stages),
            'model_calls':sum(s['model_calls'] for s in stages),
            'extra_layer_calls':sum(s['extra_layer_calls'] for s in stages),
            'nonfinite_replay_fallbacks':sum(s['nonfinite_replay_fallbacks'] for s in stages),
            'peak_allocated_gib':max((s['peak_allocated_gib'] for s in stages
                                    if s.get('peak_allocated_gib') is not None),default=None),
            'standalone_compute_fully_charged':True,'stages_executed':len(stages)}


def grade(scored:dict,labels:Sequence[str],gold:str):
    """Evaluation boundary: answer key becomes visible only AFTER scoring/selection."""
    if gold not in labels or len(scored['scores'])!=len(labels):raise ValueError('Invalid gold/choice labels')
    vals=scored['scores'];tokens=scored['token_counts'];chars=scored['character_counts']
    pick=lambda scores:max(range(len(scores)),key=lambda i:scores[i])
    raw=pick(vals);token=pick([v/max(1,n) for v,n in zip(vals,tokens)])
    char=pick([v/max(1,n) for v,n in zip(vals,chars)])
    return {**scored,'prediction':labels[raw],'token_normalized_prediction':labels[token],
        'character_normalized_prediction':labels[char],'gold':gold,
        'correct':labels[raw]==gold,'token_normalized_correct':labels[token]==gold,
        'character_normalized_correct':labels[char]==gold}


def wilson(correct,total,z=1.96):
    if not total:return None
    p=correct/total;d=1+z*z/total
    midpoint=(p+z*z/(2*total))/d
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0.,midpoint-half),min(1.,midpoint+half)]


def summarize_arc(records:Sequence[dict],baseline:Sequence[dict]|None=None):
    good=[r for r in records if r.get('status')=='completed']
    n=len(good);correct=sum(r['correct'] for r in good)
    seconds=sum(r['model_seconds'] for r in good)
    out={'attempted':len(records),'evaluated':n,'errors':len(records)-n,'correct':correct,
        'accuracy':correct/n if n else None,'wilson_95_interval':wilson(correct,n),
        'token_normalized_accuracy':sum(r['token_normalized_correct'] for r in good)/n if n else None,
        'character_normalized_accuracy':sum(r['character_normalized_correct'] for r in good)/n if n else None,
        'total_model_seconds':seconds,'seconds_per_correct_answer':seconds/correct if correct else None,
        'extra_layer_calls':sum(r['extra_layer_calls'] for r in good),
        'nonfinite_replay_fallbacks':sum(r['nonfinite_replay_fallbacks'] for r in good),
        'selected_regimes':dict(Counter(r.get('selected_regime','fixed') for r in good)),
        'automatic_promotion':False}
    if baseline is not None:
        ref={r['id']:r for r in baseline if r.get('status')=='completed'}
        paired=[r for r in good if r['id'] in ref]
        improved=[r['id'] for r in paired if r['correct'] and not ref[r['id']]['correct']]
        harmed=[r['id'] for r in paired if not r['correct'] and ref[r['id']]['correct']]
        total_ref=sum(ref[r['id']]['model_seconds'] for r in paired)
        total_cand=sum(r['model_seconds'] for r in paired)
        out['paired']={'n':len(paired),'fixed_ids':improved,'broken_ids':harmed,
                       'net_correct_change':len(improved)-len(harmed),
                       'runtime_factor':total_cand/total_ref if total_ref else None,
                       'missing_examples_do_not_count_as_success':True}
    return out
