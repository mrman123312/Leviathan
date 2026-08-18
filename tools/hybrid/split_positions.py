#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path

def group_key(line):
    try:
        r=json.loads(line)
        if isinstance(r,dict) and r.get('group_id'):return 'group:'+str(r['group_id'])
    except Exception:pass
    return 'row:'+line

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--train',required=True);p.add_argument('--holdout',required=True);p.add_argument('--holdout-frac',type=float,default=.2);p.add_argument('--seed',default='20260817');a=p.parse_args();rows=[x.strip() for x in Path(a.input).read_text(encoding='utf-8').splitlines() if x.strip()];tr=[];ho=[];groups={}
for r in rows:groups.setdefault(group_key(r),[]).append(r)
for g,items in groups.items():
    u=int(hashlib.sha256((a.seed+'\n'+g).encode()).hexdigest()[:16],16)/(16**16-1);(ho if u<a.holdout_frac else tr).extend(items)
Path(a.train).write_text('\n'.join(tr)+'\n',encoding='utf-8');Path(a.holdout).write_text('\n'.join(ho)+'\n',encoding='utf-8');print(json.dumps({'train':len(tr),'holdout':len(ho),'groups':len(groups),'train_groups':len({group_key(x) for x in tr}),'holdout_groups':len({group_key(x) for x in ho})},indent=2))