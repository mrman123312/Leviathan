#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
try:from gpu_risk_model import FEATURE_NAMES
except ImportError:from .gpu_risk_model import FEATURE_NAMES

def load(paths):
    out=[]
    for p in paths:
        with open(p,encoding='utf-8') as f:
            for line in f:
                try:r=json.loads(line)
                except Exception:continue
                if isinstance(r.get('features'),list) and len(r['features'])==len(FEATURE_NAMES):out.append(r)
    return out
def auc(scores,labels):
    pairs=sorted(zip(scores,labels));pos=sum(y>.5 for _,y in pairs);neg=len(pairs)-pos
    if not pos or not neg:return None
    rs=0.;i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        rs+=((i+1+j)/2.)*sum(y>.5 for _,y in pairs[i:j]);i=j
    return (rs-pos*(pos+1)/2.)/(pos*neg)
def main():
    a=argparse.ArgumentParser();a.add_argument('datasets',nargs='+');a.add_argument('--output',required=True);a.add_argument('--device',default='auto',choices=('auto','cuda','cpu'));a.add_argument('--hidden',type=int,default=32);a.add_argument('--epochs',type=int,default=80);a.add_argument('--batch-size',type=int,default=128);a.add_argument('--lr',type=float,default=2e-3);a.add_argument('--seed',type=int,default=20260817);a.add_argument('--holdout',type=float,default=.2);x=a.parse_args()
    import torch
    random.seed(x.seed);torch.manual_seed(x.seed);device='cuda' if x.device in ('auto','cuda') and torch.cuda.is_available() else 'cpu'
    if x.device=='cuda' and device!='cuda':raise SystemExit('CUDA requested but unavailable')
    rows=load(x.datasets)
    if len(rows)<20:raise SystemExit(f'need at least 20 labeled rows; found {len(rows)}')
    random.shuffle(rows);cut=max(1,min(len(rows)-1,int(len(rows)*(1-x.holdout))));train,valid=rows[:cut],rows[cut:]
    m=torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES),x.hidden),torch.nn.SiLU(),torch.nn.Linear(x.hidden,x.hidden),torch.nn.SiLU(),torch.nn.Linear(x.hidden,2)).to(device);opt=torch.optim.AdamW(m.parameters(),lr=x.lr,weight_decay=1e-4);bce=torch.nn.BCEWithLogitsLoss(reduction='none')
    def ts(batch):
        X=torch.tensor([r['features'] for r in batch],dtype=torch.float32,device=device);yr=torch.tensor([float(r.get('reply_label') or 0) for r in batch],device=device);mr=torch.tensor([1. if r.get('reply_label') is not None else 0. for r in batch],device=device);yk=torch.tensor([float(r.get('risk_label') or 0) for r in batch],device=device);mk=torch.tensor([1. if r.get('risk_label') is not None else 0. for r in batch],device=device);return X,yr,mr,yk,mk
    for _ in range(x.epochs):
        random.shuffle(train);m.train()
        for i in range(0,len(train),x.batch_size):
            X,yr,mr,yk,mk=ts(train[i:i+x.batch_size]);z=m(X);lr=(bce(z[:,0],yr)*mr).sum()/mr.sum().clamp_min(1);lk=(bce(z[:,1],yk)*mk).sum()/mk.sum().clamp_min(1);loss=lr+lk;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    m.eval();X,yr,mr,yk,mk=ts(valid)
    with torch.inference_mode():z=m(X);pr=torch.sigmoid(z[:,0]).cpu().tolist();pk=torch.sigmoid(z[:,1]).cpu().tolist()
    yrl=yr.cpu().tolist();mrl=mr.cpu().tolist();ykl=yk.cpu().tolist();mkl=mk.cpu().tolist();rs=[s for s,q in zip(pr,mrl) if q];rl=[y for y,q in zip(yrl,mrl) if q];ks=[s for s,q in zip(pk,mkl) if q];kl=[y for y,q in zip(ykl,mkl) if q]
    metrics={'device':device,'rows':len(rows),'train_rows':len(train),'valid_rows':len(valid),'reply_auc':auc(rs,rl) if rs else None,'risk_auc':auc(ks,kl) if ks else None,'seed':x.seed};o=Path(x.output);o.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':m.state_dict(),'hidden':x.hidden,'feature_names':FEATURE_NAMES,'metrics':metrics},o);print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
