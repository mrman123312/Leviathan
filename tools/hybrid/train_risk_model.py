#!/usr/bin/env python3
"""Leakage-resistant trainer for Leviathan P18 hybrid advisor.

Key differences from the first prototype:
- grouped train/validation split (position/session groups never cross folds)
- optional untouched prospective datasets
- train-only feature normalization persisted in the checkpoint
- masked class-balanced losses for reply/risk heads
- third head predicts log1p centipawn regret
- PR-AUC, ROC-AUC, Brier, top-risk regret capture and reply top-1 metrics
- deterministic seed and predeclared promotion gates
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from pathlib import Path
from typing import Iterable
try:
    from gpu_risk_model import FEATURE_NAMES
except ImportError:
    from .gpu_risk_model import FEATURE_NAMES


def load(paths: Iterable[str]):
    out=[]
    for path in paths:
        p=Path(path)
        with p.open(encoding='utf-8') as f:
            for lineno,line in enumerate(f,1):
                try:r=json.loads(line)
                except Exception:continue
                if not isinstance(r.get('features'),list) or len(r['features'])!=len(FEATURE_NAMES):continue
                r=dict(r);r['_source_file']=str(p);r['_line']=lineno
                r['_group']=group_key(r)
                out.append(r)
    return out


def group_key(r):
    if r.get('group_id'):
        raw='group:'+str(r['group_id'])
    elif r.get('position'):
        raw='position:'+str(r['position'])
    elif r.get('session_key'):
        raw='session:'+str(r['session_key'])
    elif r.get('generation') is not None:
        raw='generation:'+str(r.get('_source_file',''))+':'+str(r['generation'])
    else:
        raw='row:'+str(r.get('_source_file',''))+':'+str(r.get('_line',0))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def grouped_split(rows, holdout, seed):
    groups={}
    for r in rows:groups.setdefault(r['_group'],[]).append(r)
    keys=list(groups);random.Random(seed).shuffle(keys)
    target=max(1,int(round(len(rows)*holdout)))
    valid_keys=set();count=0
    for k in keys:
        if count>=target and valid_keys:break
        valid_keys.add(k);count+=len(groups[k])
    train=[r for r in rows if r['_group'] not in valid_keys]
    valid=[r for r in rows if r['_group'] in valid_keys]
    if not train or not valid:raise SystemExit('grouped split produced empty train/validation set')
    return train,valid


def roc_auc(scores,labels):
    if not scores:return None
    pairs=sorted(zip(scores,labels));pos=sum(y>.5 for _,y in pairs);neg=len(pairs)-pos
    if not pos or not neg:return None
    rs=0.;i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        avg=(i+1+j)/2.;rs+=avg*sum(y>.5 for _,y in pairs[i:j]);i=j
    return (rs-pos*(pos+1)/2.)/(pos*neg)


def pr_auc(scores,labels):
    pos=sum(y>.5 for y in labels)
    if not scores or not pos:return None
    order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
    tp=fp=0;last_recall=0.;area=0.
    for i in order:
        if labels[i]>.5:tp+=1
        else:fp+=1
        recall=tp/pos;precision=tp/max(1,tp+fp)
        area+=(recall-last_recall)*precision;last_recall=recall
    return area


def brier(scores,labels):
    return sum((s-y)**2 for s,y in zip(scores,labels))/len(scores) if scores else None


def top_fraction_capture(scores, values, frac=.2):
    if not scores or not values:return None
    total=sum(max(0.,float(v)) for v in values)
    if total<=0:return None
    n=max(1,int(math.ceil(len(scores)*frac)))
    idx=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:n]
    return sum(max(0.,float(values[i])) for i in idx)/total


def reply_top1(rows, probs):
    by={}
    for r,p in zip(rows,probs):
        if r.get('reply_label') is None:continue
        by.setdefault(r['_group'],[]).append((p,float(r['reply_label']),float(r.get('rank',999))))
    if not by:return None,None
    good=base=0
    for vals in by.values():
        good+=int(max(vals,key=lambda x:x[0])[1]>.5)
        base+=int(min(vals,key=lambda x:x[2])[1]>.5)
    return good/len(by),base/len(by)


def class_weights(rows,key):
    ys=[float(r[key]) for r in rows if r.get(key) is not None]
    pos=sum(y>.5 for y in ys);neg=len(ys)-pos
    return max(1.,neg/max(1,pos)) if ys else 1.


def metrics_for(rows, reply_probs, risks, regrets):
    ri=[i for i,r in enumerate(rows) if r.get('reply_label') is not None]
    ki=[i for i,r in enumerate(rows) if r.get('risk_label') is not None]
    rl=[float(rows[i]['reply_label']) for i in ri];rp=[reply_probs[i] for i in ri]
    kl=[float(rows[i]['risk_label']) for i in ki];kp=[risks[i] for i in ki]
    rv=[float(rows[i].get('regret_cp') or 0.) for i in ki]
    top1,base=reply_top1([rows[i] for i in ri],rp)
    mae=None
    reg_idx=[i for i,r in enumerate(rows) if r.get('regret_cp') is not None]
    if reg_idx:mae=sum(abs(regrets[i]-float(rows[i].get('regret_cp') or 0.)) for i in reg_idx)/len(reg_idx)
    return {
        'rows':len(rows),'groups':len({r['_group'] for r in rows}),
        'reply_positive_rate':sum(rl)/len(rl) if rl else None,
        'reply_auc':roc_auc(rp,rl),'reply_pr_auc':pr_auc(rp,rl),'reply_brier':brier(rp,rl),
        'reply_top1_accuracy':top1,'reply_rank1_baseline':base,
        'risk_positive_rate':sum(kl)/len(kl) if kl else None,
        'risk_auc':roc_auc(kp,kl),'risk_pr_auc':pr_auc(kp,kl),'risk_brier':brier(kp,kl),
        'risk_top20_regret_capture':top_fraction_capture(kp,rv,.2) if kp else None,
        'risk_top10_regret_capture':top_fraction_capture(kp,rv,.1) if kp else None,
        'regret_mae_cp':mae,
    }


def passes(m, min_risk_auc, min_capture, min_reply_gain):
    checks={}
    if m.get('risk_auc') is not None:checks['risk_auc']=m['risk_auc']>=min_risk_auc
    if m.get('risk_top20_regret_capture') is not None:checks['top20_regret_capture']=m['risk_top20_regret_capture']>=min_capture
    if m.get('reply_top1_accuracy') is not None and m.get('reply_rank1_baseline') is not None:
        checks['reply_top1_vs_rank1']=m['reply_top1_accuracy']+1e-9>=m['reply_rank1_baseline']+min_reply_gain
    return checks, bool(checks) and all(checks.values())


def main():
    a=argparse.ArgumentParser()
    a.add_argument('datasets',nargs='+');a.add_argument('--prospective',nargs='*',default=[])
    a.add_argument('--output',required=True);a.add_argument('--metrics-output',default=None)
    a.add_argument('--device',default='auto',choices=('auto','cuda','cpu'));a.add_argument('--hidden',type=int,default=48)
    a.add_argument('--epochs',type=int,default=160);a.add_argument('--patience',type=int,default=20);a.add_argument('--batch-size',type=int,default=128);a.add_argument('--lr',type=float,default=2e-3)
    a.add_argument('--seed',type=int,default=20260817);a.add_argument('--holdout',type=float,default=.2)
    a.add_argument('--min-risk-auc',type=float,default=.58);a.add_argument('--min-top20-regret-capture',type=float,default=.30);a.add_argument('--min-reply-top1-gain',type=float,default=0.0)
    x=a.parse_args()
    import torch
    random.seed(x.seed);torch.manual_seed(x.seed)
    device='cuda' if x.device in ('auto','cuda') and torch.cuda.is_available() else 'cpu'
    if x.device=='cuda' and device!='cuda':raise SystemExit('CUDA requested but unavailable')
    rows=load(x.datasets)
    if len(rows)<40:raise SystemExit(f'need at least 40 labeled rows; found {len(rows)}')
    train,valid=grouped_split(rows,x.holdout,x.seed)
    prospective=load(x.prospective) if x.prospective else []
    if prospective and ({r['_group'] for r in rows}&{r['_group'] for r in prospective}):raise SystemExit('prospective data overlaps training groups')
    Xtr=torch.tensor([r['features'] for r in train],dtype=torch.float32)
    mean=Xtr.mean(0);std=Xtr.std(0,unbiased=False).clamp_min(1e-6)
    mean_d=mean.to(device);std_d=std.to(device)
    model=torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES),x.hidden),torch.nn.SiLU(),torch.nn.Linear(x.hidden,x.hidden),torch.nn.SiLU(),torch.nn.Linear(x.hidden,3)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=x.lr,weight_decay=1e-4);bce=torch.nn.BCEWithLogitsLoss(reduction='none');smooth=torch.nn.SmoothL1Loss(reduction='none')
    wr=class_weights(train,'reply_label');wk=class_weights(train,'risk_label')
    def tensors(batch):
        X=torch.tensor([r['features'] for r in batch],dtype=torch.float32,device=device);X=(X-mean_d)/std_d
        yr=torch.tensor([float(r.get('reply_label') or 0) for r in batch],device=device);mr=torch.tensor([1. if r.get('reply_label') is not None else 0. for r in batch],device=device)
        yk=torch.tensor([float(r.get('risk_label') or 0) for r in batch],device=device);mk=torch.tensor([1. if r.get('risk_label') is not None else 0. for r in batch],device=device)
        yg=torch.tensor([math.log1p(max(0.,float(r.get('regret_cp') or 0.))) for r in batch],device=device);mg=torch.tensor([1. if r.get('regret_cp') is not None else 0. for r in batch],device=device)
        return X,yr,mr,yk,mk,yg,mg
    def infer(batch):
        model.eval();X,_,_,_,_,_,_=tensors(batch)
        with torch.inference_mode():z=model(X);rp=torch.sigmoid(z[:,0]).cpu().tolist();rk=torch.sigmoid(z[:,1]).cpu().tolist();rg=torch.expm1(torch.clamp(z[:,2],min=0,max=math.log1p(1000))).cpu().tolist()
        return rp,rk,rg
    best=None;best_state=None;stale=0
    for epoch in range(x.epochs):
        random.shuffle(train);model.train()
        for i in range(0,len(train),x.batch_size):
            batch=train[i:i+x.batch_size];X,yr,mr,yk,mk,yg,mg=tensors(batch);z=model(X)
            rw=torch.where(yr>.5,torch.full_like(yr,wr),torch.ones_like(yr));kw=torch.where(yk>.5,torch.full_like(yk,wk),torch.ones_like(yk))
            lr=(bce(z[:,0],yr)*rw*mr).sum()/mr.sum().clamp_min(1)
            lk=(bce(z[:,1],yk)*kw*mk).sum()/mk.sum().clamp_min(1)
            lg=(smooth(z[:,2],yg)*mg).sum()/mg.sum().clamp_min(1)
            loss=lr+lk+0.35*lg;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
        rp,rk,rg=infer(valid);vm=metrics_for(valid,rp,rk,rg)
        score=(vm.get('risk_auc') or .5)+(vm.get('risk_top20_regret_capture') or .2)+0.25*(vm.get('reply_top1_accuracy') or 0.)
        if best is None or score>best+1e-6:
            best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=x.patience:break
    if best_state is not None:model.load_state_dict(best_state)
    vrp,vrk,vrg=infer(valid);valid_metrics=metrics_for(valid,vrp,vrk,vrg)
    pros_metrics=None
    if prospective:
        prp,prk,prg=infer(prospective);pros_metrics=metrics_for(prospective,prp,prk,prg)
    gate_metrics=pros_metrics or valid_metrics
    checks,promote=passes(gate_metrics,x.min_risk_auc,x.min_top20_regret_capture,x.min_reply_top1_gain)
    metrics={'version':'p18.2','device':device,'train_rows':len(train),'valid':valid_metrics,'prospective':pros_metrics,'promotion_checks':checks,'promote':promote,'seed':x.seed,'reply_pos_weight':wr,'risk_pos_weight':wk}
    payload={'state_dict':model.state_dict(),'hidden':x.hidden,'heads':3,'feature_names':FEATURE_NAMES,'normalizer':{'mean':mean.tolist(),'std':std.tolist()},'metrics':metrics}
    out=Path(x.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,out)
    mo=Path(x.metrics_output) if x.metrics_output else out.with_suffix('.metrics.json');mo.write_text(json.dumps(metrics,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(metrics,indent=2,sort_keys=True))
    return 0 if promote else 4
if __name__=='__main__':raise SystemExit(main())
