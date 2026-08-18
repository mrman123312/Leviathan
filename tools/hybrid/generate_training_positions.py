#!/usr/bin/env python3
"""Generate diverse engine-relevant positions without python-chess.

With --grouped-jsonl every position carries its source game id so train,
validation and prospective splits can keep whole trajectories isolated.
"""
from __future__ import annotations
import argparse,json,math,queue,random,time
from pathlib import Path
try:from leviathan_hybrid_uci import EngineProcess,parse_info
except ImportError:from .leviathan_hybrid_uci import EngineProcess,parse_info

def init(path,threads,h):
    e=EngineProcess(path,'position-generator');e.send('uci');e.wait_for('uciok',20);e.initialized=True;e.send(f'setoption name Threads value {threads}');e.send(f'setoption name Hash value {h}');e.send('isready');e.wait_for('readyok',20);return e

def choose(e,pos,nodes,multipv,rng,temp):
    e.drain();e.send('ucinewgame');e.send(f'setoption name MultiPV value {multipv}');e.send('isready');e.wait_for('readyok',20);e.send(pos);e.send(f'go nodes {nodes}');latest={};end=time.monotonic()+120
    while time.monotonic()<end:
        try:l=e.read(1)
        except queue.Empty:continue
        i=parse_info(l)
        if i and i.get('pv'):latest[int(i.get('multipv',1))]=i
        if l.startswith('bestmove'):break
    rows=[]
    for k in sorted(latest):
        i=latest[k];pv=i.get('pv') or []
        if pv:rows.append((pv[0],float(i.get('score_cp',0))))
    if not rows:return None
    best=max(s for _,s in rows);weights=[math.exp(max(-20,min(0,(s-best)/max(1,temp)))) for _,s in rows]
    return rng.choices([m for m,_ in rows],weights=weights,k=1)[0]

def main():
    a=argparse.ArgumentParser();a.add_argument('--engine',required=True);a.add_argument('--output',required=True);a.add_argument('--games',type=int,default=80);a.add_argument('--min-ply',type=int,default=8);a.add_argument('--max-ply',type=int,default=70);a.add_argument('--nodes',type=int,default=3000);a.add_argument('--multipv',type=int,default=4);a.add_argument('--temperature-cp',type=float,default=55);a.add_argument('--threads',type=int,default=1);a.add_argument('--hash',type=int,default=32);a.add_argument('--seed',type=int,default=20260817);a.add_argument('--grouped-jsonl',action='store_true');x=a.parse_args();rng=random.Random(x.seed);e=init(x.engine,x.threads,x.hash);seen=set();out=[]
    try:
        for game in range(x.games):
            moves=[];gid=f'game-{x.seed}-{game:05d}'
            for ply in range(x.max_ply):
                pos='position startpos'+((' moves '+' '.join(moves)) if moves else '')
                m=choose(e,pos,x.nodes,x.multipv,rng,x.temperature_cp if ply<24 else max(18,x.temperature_cp*.55))
                if not m or m=='0000':break
                moves.append(m)
                if ply+1>=x.min_ply:
                    q='position startpos moves '+' '.join(moves)
                    if q in seen:continue
                    seen.add(q);out.append({'group_id':gid,'position':q,'ply':ply+1} if x.grouped_jsonl else q)
    finally:e.close()
    p=Path(x.output);p.parent.mkdir(parents=True,exist_ok=True)
    if x.grouped_jsonl:p.write_text('\n'.join(json.dumps(r,sort_keys=True) for r in out)+'\n',encoding='utf-8')
    else:p.write_text('\n'.join(out)+'\n',encoding='utf-8')
    print(json.dumps({'positions':len(out),'games':x.games,'grouped_jsonl':x.grouped_jsonl,'output':str(p),'seed':x.seed},indent=2))
if __name__=='__main__':main()