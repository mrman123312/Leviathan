#!/usr/bin/env python3
"""Measure the value of a correct opponent-clock ponder hit.

For each prospective row whose reply_label==1, compare:
- cold P09: own_ms only
- warm P09: ponder_ms pre-search on the opponent clock, then the same own_ms
The deep_move already mined for that branch is used as the frozen oracle target.
"""
from __future__ import annotations
import argparse,json,queue,time
from pathlib import Path
try:from leviathan_hybrid_uci import EngineProcess,append_move_to_position,parse_info
except ImportError:from .leviathan_hybrid_uci import EngineProcess,append_move_to_position,parse_info

def init(path,label,threads,h):
    e=EngineProcess(path,label);e.send('uci');e.wait_for('uciok',20);e.initialized=True;e.send(f'setoption name Threads value {threads}');e.send(f'setoption name Hash value {h}');e.send('isready');e.wait_for('readyok',20);return e
def clear(e):
    e.drain();e.send('stop');e.drain();e.send('ucinewgame');e.send('setoption name Clear Hash');e.send('isready');e.wait_for('readyok',20)
def go(e,pos,ms):
    e.drain();e.send(pos);e.send(f'go movetime {ms}');best=None;last={};end=time.monotonic()+max(10,ms/1000+10)
    while time.monotonic()<end:
        try:l=e.read(1)
        except queue.Empty:continue
        i=parse_info(l)
        if i:last=i
        if l.startswith('bestmove'):best=l.split()[1];break
    return best,last
def rows(path,limit):
    seen=set();out=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try:r=json.loads(line)
        except Exception:continue
        if float(r.get('reply_label') or 0)<.5 or not r.get('deep_move'):continue
        k=r.get('group_id') or r.get('position')
        if k in seen:continue
        seen.add(k);out.append(r)
        if limit and len(out)>=limit:break
    return out
def main():
    a=argparse.ArgumentParser();a.add_argument('--engine',required=True);a.add_argument('--dataset',required=True);a.add_argument('--output',required=True);a.add_argument('--ponder-ms',type=int,default=2000);a.add_argument('--own-ms',type=int,default=250);a.add_argument('--threads',type=int,default=1);a.add_argument('--hash',type=int,default=64);a.add_argument('--limit',type=int,default=48);x=a.parse_args();cold=init(x.engine,'cold',x.threads,x.hash);warm=init(x.engine,'warm',x.threads,x.hash);data=rows(x.dataset,x.limit);res=[]
    try:
        for n,r in enumerate(data):
            branch=append_move_to_position(r['position'],r['reply']);oracle=r['deep_move'];clear(cold);clear(warm)
            cb,ci=go(cold,branch,x.own_ms);go(warm,branch,x.ponder_ms);wb,wi=go(warm,branch,x.own_ms)
            q={'index':n,'reply':r['reply'],'oracle':oracle,'cold_move':cb,'warm_move':wb,'cold_hit':cb==oracle,'warm_hit':wb==oracle,'cold_depth':int(ci.get('depth',0)),'warm_depth':int(wi.get('depth',0)),'cold_seldepth':int(ci.get('seldepth',0)),'warm_seldepth':int(wi.get('seldepth',0)),'cold_nodes':int(ci.get('nodes',0)),'warm_nodes':int(wi.get('nodes',0))};res.append(q);print(json.dumps(q),flush=True)
    finally:cold.close();warm.close()
    n=len(res);summary={'positions':n,'ponder_ms':x.ponder_ms,'own_ms':x.own_ms,'cold_oracle_hits':sum(q['cold_hit'] for q in res),'warm_oracle_hits':sum(q['warm_hit'] for q in res),'oracle_hit_delta':sum(q['warm_hit'] for q in res)-sum(q['cold_hit'] for q in res),'mean_depth_delta':sum(q['warm_depth']-q['cold_depth'] for q in res)/n if n else 0,'mean_seldepth_delta':sum(q['warm_seldepth']-q['cold_seldepth'] for q in res)/n if n else 0,'pass':bool(n and sum(q['warm_hit'] for q in res)>=sum(q['cold_hit'] for q in res) and sum(q['warm_depth']-q['cold_depth'] for q in res)>=0)}
    Path(x.output).write_text(json.dumps({'summary':summary,'rows':res},indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return 0 if summary['pass'] else 5
if __name__=='__main__':raise SystemExit(main())
