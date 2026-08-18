#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--train',required=True);p.add_argument('--holdout',required=True);p.add_argument('--holdout-frac',type=float,default=.2);p.add_argument('--seed',default='20260817');a=p.parse_args();rows=[x.strip() for x in Path(a.input).read_text(encoding='utf-8').splitlines() if x.strip()];tr=[];ho=[]
for r in rows:
    u=int(hashlib.sha256((a.seed+'\n'+r).encode()).hexdigest()[:16],16)/(16**16-1);(ho if u<a.holdout_frac else tr).append(r)
Path(a.train).write_text('\n'.join(tr)+'\n',encoding='utf-8');Path(a.holdout).write_text('\n'.join(ho)+'\n',encoding='utf-8');print(json.dumps({'train':len(tr),'holdout':len(ho)},indent=2))
