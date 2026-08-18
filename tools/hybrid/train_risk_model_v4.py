#!/usr/bin/env python3
"""P18.4 advisor trainer.

Separates two concepts that must never be conflated:
- leakage group: whole generated game/session, used only for train/validation splitting
- decision group: one exact chess position, used for reply-pool softmax/ranking metrics

The reply head is trained groupwise (cross-entropy inside each covered candidate pool),
while risk/regret heads are trained only where expensive oracle labels exist.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from pathlib import Path
from typing import Iterable
try:
    from gpu_risk_model import FEATURE_NAMES
except ImportError:
    from .gpu_risk_model import FEATURE_NAMES

def h24(s): return hashlib.sha256(str(s).encode()).hexdigest()[:24]

def leakage_key(r):
    if r.get('group_id'): return h24('group:'+str(r['group_id']))
    if r.get('session_key'): return h24('session:'+str(r['session_key']))
    if r.get('generation') is not None: return h24('generation:'+str(r.get('_source_file',''))+':'+str(r['generation']))
    if r.get('position'): return h24('position:'+str(r['position']))
    return h24('row:'+str(r.get('_source_file',''))+':'+str(r.get('_line',0)))

def decision_key(r):
    if r.get('decision_id'): return str(r['decision_id'])
    if r.get('position'): return h24('decision:'+str(r['position']))
    if r.get('key'): return h24('key:'+str(r['key']))
    return h24('row:'+str(r.get('_source_file',''))+':'+str(r.get('_line',0)))

def rank_of(r):
    if r.get('rank') is not None:
        try: return float(r['rank'])
        except Exception: pass
    f=r.get('features') or []
    return float(f[0]) if f else 999.0

def load(paths: Iterable[str]):
    raw=[]
    for path in paths:
        p=Path(path)
        if not p.exists(): continue
        with p.open(encoding='utf-8') as f:
            for lineno,line in enumerate(f,1):
                try:r=json.loads(line)
                except Exception:continue
                if not isinstance(r.get('features'),list) or len(r['features'])!=len(FEATURE_NAMES):continue
                r=dict(r);r['_source_file']=str(p);r['_line']=lineno
                r['_group']=leakage_key(r);r['_decision']=decision_key(r);r['_rank']=rank_of(r)
                raw.append(r)
    merged={};order=[]
    for r in raw:
        k=(r['_decision'],str(r.get('reply') or r.get('move') or r.get('key') or r['_line']))
        if k not in merged:
            merged[k]=r;order.append(k);continue
        a=merged[k]
        for field in ('reply_label','risk_label','regret_cp','fast_move','deep_move','deep_best_cp','deep_fast_move_cp'):
            if a.get(field) is None and r.get(field) is not None:a[field]=r[field]
        if a.get('rank') is None and r.get('rank') is not None:a['rank']=r['rank'];a['_rank']=rank_of(r)
    return [merged[k] for k in order]

def grouped_split(rows,holdout,seed):
    groups={}
    for r in rows:groups.setdefault(r['_group'],[]).append(r)
    keys=list(groups);random.Random(seed).shuffle(keys)
    target=max(1,int(round(len(rows)*holdout)));valid_keys=set();count=0
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
    order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True);tp=fp=0;last=0.;area=0.
    for i in order:
        if labels[i]>.5:tp+=1
        else:fp+=1
        rec=tp/pos;prec=tp/max(1,tp+fp);area+=(rec-last)*prec;last=rec
    return area

def brier(scores,labels):
    return sum((s-y)**2 for s,y in zip(scores,labels))/len(scores) if scores else None

def top_fraction_capture(scores,values,frac=.2):
    if not scores:return None
    total=sum(max(0.,float(v)) for v in values)
    if total<=0:return None
    n=max(1,int(math.ceil(len(scores)*frac)))
    idx=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:n]
    return sum(max(0.,float(values[i])) for i in idx)/total

def decision_groups(rows):
    by={}
    for i,r in enumerate(rows):by.setdefault(r['_decision'],[]).append(i)
    return by

def reply_pool_metrics(rows,probs):
    by=decision_groups(rows);total=len(by);covered=good=base=0
    for idxs in by.values():
        labeled=[i for i in idxs if rows[i].get('reply_label') is not None]
        if not labeled:continue
        pos=[i for i in labeled if float(rows[i]['reply_label'])>.5]
        if not pos:continue
        covered+=1
        good+=int(max(labeled,key=lambda i:probs[i]) in pos)
        base+=int(min(labeled,key=lambda i:rows[i]['_rank']) in pos)
    return {'reply_decisions':total,'reply_covered_decisions':covered,'reply_candidate_coverage':covered/total if total else None,'reply_top1_accuracy':good/covered if covered else None,'reply_rank1_baseline':base/covered if covered else None}

def metrics_for(rows,reply_probs,risks,regrets):
    ri=[i for i,r in enumerate(rows) if r.get('reply_label') is not None]
    ki=[i for i,r in enumerate(rows) if r.get('risk_label') is not None]
    rl=[float(rows[i]['reply_label']) for i in ri];rp=[reply_probs[i] for i in ri]
    kl=[float(rows[i]['risk_label']) for i in ki];kp=[risks[i] for i in ki]
    rv=[float(rows[i].get('regret_cp') or 0.) for i in ki]
    reg_idx=[i for i,r in enumerate(rows) if r.get('regret_cp') is not None]
    mae=sum(abs(regrets[i]-float(rows[i].get('regret_cp') or 0.)) for i in reg_idx)/len(reg_idx) if reg_idx else None
    out={'rows':len(rows),'leakage_groups':len({r['_group'] for r in rows}),'decision_groups':len({r['_decision'] for r in rows}),'reply_positive_rate':sum(rl)/len(rl) if rl else None,'reply_auc':roc_auc(rp,rl),'reply_pr_auc':pr_auc(rp,rl),'reply_brier':brier(rp,rl),'risk_labeled_rows':len(ki),'risk_positive_rate':sum(kl)/len(kl) if kl else None,'risk_auc':roc_auc(kp,kl),'risk_pr_auc':pr_auc(kp,kl),'risk_brier':brier(kp,kl),'risk_top20_regret_capture':top_fraction_capture(kp,rv,.2) if kp else None,'risk_top10_regret_capture':top_fraction_capture(kp,rv,.1) if kp else None,'regret_mae_cp':mae}
    out.update(reply_pool_metrics(rows,reply_probs));return out

def passes(m,min_risk_auc,min_capture,min_reply_gain,min_coverage):
    checks={}
    if m.get('risk_auc') is not None:checks['risk_auc']=m['risk_auc']>=min_risk_auc
    if m.get('risk_top20_regret_capture') is not None:checks['top20_regret_capture']=m['risk_top20_regret_capture']>=min_capture
    if m.get('reply_candidate_coverage') is not None:checks['reply_candidate_coverage']=m['reply_candidate_coverage']>=min_coverage
    if m.get('reply_top1_accuracy') is not None and m.get('reply_rank1_baseline') is not None:checks['reply_top1_vs_rank1']=m['reply_top1_accuracy']+1e-9>=m['reply_rank1_baseline']+min_reply_gain
    return checks,bool(checks) and all(checks.values())

def select_device(torch,requested):
    if requested in ('cuda','auto') and torch.cuda.is_available():return 'cuda',torch.device('cuda')
    if requested in ('dml','auto'):
        try:
            import torch_directml
            return 'dml',torch_directml.device()
        except Exception:
            if requested=='dml':raise SystemExit('DirectML requested but torch-directml unavailable')
    if requested=='cuda':raise SystemExit('CUDA requested but unavailable')
    return 'cpu',torch.device('cpu')

def pack_decision_batches(rows,batch_rows,rng):
    by={}
    for r in rows:by.setdefault(r['_decision'],[]).append(r)
    keys=list(by);rng.shuffle(keys);batch=[]
    for k in keys:
        g=by[k]
        if batch and len(batch)+len(g)>batch_rows:
            yield batch;batch=[]
        batch.extend(g)
    if batch:yield batch

def main():
    a=argparse.ArgumentParser();a.add_argument('datasets',nargs='+');a.add_argument('--prospective',nargs='*',default=[]);a.add_argument('--output',required=True);a.add_argument('--metrics-output',default=None);a.add_argument('--device',default='auto',choices=('auto','cuda','dml','cpu'));a.add_argument('--hidden',type=int,default=48);a.add_argument('--epochs',type=int,default=160);a.add_argument('--patience',type=int,default=20);a.add_argument('--batch-size',type=int,default=128);a.add_argument('--lr',type=float,default=2e-3);a.add_argument('--seed',type=int,default=20260817);a.add_argument('--holdout',type=float,default=.2);a.add_argument('--min-risk-auc',type=float,default=.58);a.add_argument('--min-top20-regret-capture',type=float,default=.30);a.add_argument('--min-reply-top1-gain',type=float,default=0.0);a.add_argument('--min-reply-coverage',type=float,default=.90);x=a.parse_args()
    import torch
    random.seed(x.seed);torch.manual_seed(x.seed);device_name,device=select_device(torch,x.device)
    rows=load(x.datasets)
    if len(rows)<40:raise SystemExit(f'need at least 40 labeled rows; found {len(rows)}')
    train,valid=grouped_split(rows,x.holdout,x.seed);prospective=load(x.prospective) if x.prospective else []
    if prospective and ({r['_group'] for r in rows}&{r['_group'] for r in prospective}):raise SystemExit('prospective leakage groups overlap training')
    Xtr=torch.tensor([r['features'] for r in train],dtype=torch.float32);mean=Xtr.mean(0);std=Xtr.std(0,unbiased=False).clamp_min(1e-6);mean_d=mean.to(device);std_d=std.to(device)
    model=torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES),x.hidden),torch.nn.SiLU(),torch.nn.Linear(x.hidden,x.hidden),torch.nn.SiLU(),torch.nn.Linear(x.hidden,3)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=x.lr,weight_decay=1e-4);bce=torch.nn.BCEWithLogitsLoss(reduction='none');smooth=torch.nn.SmoothL1Loss(reduction='none')
    risk_y=[float(r['risk_label']) for r in train if r.get('risk_label') is not None];pos=sum(y>.5 for y in risk_y);neg=len(risk_y)-pos;risk_pos_weight=max(1.,neg/max(1,pos)) if risk_y else 1.
    def td(v):return torch.tensor(v,dtype=torch.float32).to(device)
    def tensors(batch):
        X=(td([r['features'] for r in batch])-mean_d)/std_d;yk=td([float(r.get('risk_label') or 0) for r in batch]);mk=td([1. if r.get('risk_label') is not None else 0. for r in batch]);yg=td([math.log1p(max(0.,float(r.get('regret_cp') or 0.))) for r in batch]);mg=td([1. if r.get('regret_cp') is not None else 0. for r in batch]);return X,yk,mk,yg,mg
    def reply_loss_for_batch(batch,logits):
        by={}
        for i,r in enumerate(batch):by.setdefault(r['_decision'],[]).append(i)
        losses=[]
        for idxs in by.values():
            labeled=[i for i in idxs if batch[i].get('reply_label') is not None];posidx=[i for i in labeled if float(batch[i]['reply_label'])>.5]
            if not posidx:continue
            ls=logits[labeled];target=labeled.index(posidx[0]);losses.append(-torch.log_softmax(ls,dim=0)[target])
        return torch.stack(losses).mean() if losses else logits.sum()*0.
    def infer(batch):
        model.eval();X,_,_,_,_=tensors(batch)
        with torch.inference_mode():z=model(X)
        raw=z[:,0].detach().cpu().tolist();probs=[0.0]*len(batch);by=decision_groups(batch)
        for idxs in by.values():
            vals=[raw[i] for i in idxs];mx=max(vals);ex=[math.exp(v-mx) for v in vals];s=sum(ex) or 1.
            for i,e in zip(idxs,ex):probs[i]=e/s
        risks=torch.sigmoid(z[:,1]).detach().cpu().tolist();regs=torch.expm1(torch.clamp(z[:,2],min=0,max=math.log1p(1000))).detach().cpu().tolist();return probs,risks,regs
    best=None;best_state=None;stale=0;epochs_ran=0;rng=random.Random(x.seed)
    for epoch in range(x.epochs):
        epochs_ran=epoch+1;model.train()
        for batch in pack_decision_batches(train,x.batch_size,rng):
            X,yk,mk,yg,mg=tensors(batch);z=model(X);lr=reply_loss_for_batch(batch,z[:,0]);kw=torch.where(yk>.5,torch.full_like(yk,risk_pos_weight),torch.ones_like(yk));lk=(bce(z[:,1],yk)*kw*mk).sum()/mk.sum().clamp_min(1);lg=(smooth(z[:,2],yg)*mg).sum()/mg.sum().clamp_min(1);loss=lr+lk+0.35*lg;opt.zero_grad(set_to_none=True);loss.backward()
            try:torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,foreach=False)
            except TypeError:torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
        rp,rk,rg=infer(valid);vm=metrics_for(valid,rp,rk,rg);score=(vm.get('risk_auc') or .5)+(vm.get('risk_top20_regret_capture') or .2)+0.25*(vm.get('reply_top1_accuracy') or 0.)+0.25*(vm.get('reply_candidate_coverage') or 0.)
        if best is None or score>best+1e-6:best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=x.patience:break
    if best_state is not None:model.load_state_dict(best_state)
    vrp,vrk,vrg=infer(valid);valid_metrics=metrics_for(valid,vrp,vrk,vrg);pros_metrics=None
    if prospective:prp,prk,prg=infer(prospective);pros_metrics=metrics_for(prospective,prp,prk,prg)
    gate=pros_metrics or valid_metrics;checks,promote=passes(gate,x.min_risk_auc,x.min_top20_regret_capture,x.min_reply_top1_gain,x.min_reply_coverage)
    metrics={'version':'p18.4','device':device_name,'epochs_ran':epochs_ran,'train_rows':len(train),'valid':valid_metrics,'prospective':pros_metrics,'promotion_checks':checks,'promote':promote,'seed':x.seed,'risk_pos_weight':risk_pos_weight}
    state_cpu={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};payload={'state_dict':state_cpu,'hidden':x.hidden,'heads':3,'feature_names':FEATURE_NAMES,'normalizer':{'mean':mean.tolist(),'std':std.tolist()},'metrics':metrics}
    out=Path(x.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,out);mo=Path(x.metrics_output) if x.metrics_output else out.with_suffix('.metrics.json');mo.write_text(json.dumps(metrics,indent=2,sort_keys=True),encoding='utf-8');print(json.dumps(metrics,indent=2,sort_keys=True));return 0 if promote else 4
if __name__=='__main__':raise SystemExit(main())
