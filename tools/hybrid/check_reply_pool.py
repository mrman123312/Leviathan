#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('datasets',nargs='+');p.add_argument('--min-coverage',type=float,default=.90);a=p.parse_args()
by={}
for path in a.datasets:
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try:r=json.loads(line)
        except Exception:continue
        pos=r.get('position')
        d=str(r.get('decision_id') or hashlib.sha256(('decision:'+str(pos)).encode()).hexdigest()[:24])
        by.setdefault(d,[]).append(r)
covered=sum(any(float(r.get('reply_label') or 0)>.5 for r in rs) for rs in by.values())
total=len(by);coverage=covered/total if total else 0.
out={'decisions':total,'covered':covered,'misses':total-covered,'coverage':coverage,'minimum':a.min_coverage,'pass':coverage>=a.min_coverage}
print(json.dumps(out,indent=2,sort_keys=True))
raise SystemExit(0 if out['pass'] else 4)
