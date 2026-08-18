#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
try:from gpu_risk_model import ReplyFeatures
except ImportError:from .gpu_risk_model import ReplyFeatures

def events(paths):
    for p in paths:
        with open(p,encoding="utf-8") as f:
            for line in f:
                try:yield json.loads(line)
                except Exception:pass
def feature(c,best):
    return ReplyFeatures(float(c.get('rank',1)),float(c.get('score_cp',0)),max(0.,float(best)-float(c.get('score_cp',0))),float(c.get('depth',0)),float(c.get('seldepth',0)),float(c.get('nodes',0)),float(c.get('nps',0)),float(c.get('hashfull',0)),float(c.get('pv_len',0)),1. if c.get('predicted') else 0.,float(c.get('mate_flag',0))).vector()
def build(es):
    pools={};actual={}
    for e in es:
        g=e.get('generation')
        if g is None:continue
        if e.get('event')=='ponder_pool_ready':pools[g]=e
        elif e.get('event') in ('warm_promote','ponder_hit_branch') and e.get('reply'):actual[g]=e['reply']
    rows=[];sessions=0
    for g,p in pools.items():
        move=actual.get(g);cs=p.get('candidates') or []
        if not move or not cs or not any(c.get('move')==move for c in cs):continue
        sessions+=1;best=max(float(c.get('score_cp',0)) for c in cs)
        for c in cs:rows.append({'generation':g,'move':c.get('move'),'features':feature(c,best),'reply_label':1. if c.get('move')==move else 0.,'risk_label':None,'source':'hybrid_opponent_observation'})
    return rows,sessions
def main():
    a=argparse.ArgumentParser();a.add_argument('logs',nargs='+');a.add_argument('--output',required=True);x=a.parse_args();rows,s=build(events(x.logs));o=Path(x.output);o.parent.mkdir(parents=True,exist_ok=True)
    with o.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    print(json.dumps({'rows':len(rows),'covered_sessions':s,'output':str(o)},indent=2))
if __name__=='__main__':main()
