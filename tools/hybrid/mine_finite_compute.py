#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,queue,time
from pathlib import Path
try:from gpu_risk_model import ReplyFeatures
except ImportError:from .gpu_risk_model import ReplyFeatures
try:from leviathan_hybrid_uci import EngineProcess,append_move_to_position,parse_info
except ImportError:from .leviathan_hybrid_uci import EngineProcess,append_move_to_position,parse_info

def init(path,label,threads,h):
    e=EngineProcess(path,label);e.send('uci');e.wait_for('uciok',20);e.initialized=True;e.send(f'setoption name Threads value {threads}');e.send(f'setoption name Hash value {h}');e.send('isready');e.wait_for('readyok',20);e.options['Threads']=str(threads);e.options['Hash']=str(h);return e
def reset(e):e.drain();e.send('ucinewgame');e.send('isready');e.wait_for('readyok',20)
def search(e,pos,nodes,multipv=1,searchmove=None):
    reset(e);e.send(f'setoption name MultiPV value {multipv}');e.send('isready');e.wait_for('readyok',20);e.drain();e.send(pos);e.send(('go '+(f'searchmoves {searchmove} ' if searchmove else '')+f'nodes {nodes}'));latest={};best=None;end=time.monotonic()+180
    while time.monotonic()<end:
        try:l=e.read(1)
        except queue.Empty:continue
        i=parse_info(l)
        if i and i.get('pv'):
            m=int(i.get('multipv',1));q=latest.get(m)
            if q is None or (i.get('depth',0),i.get('nodes',0))>=(q.get('depth',0),q.get('nodes',0)):latest[m]=i
        if l.startswith('bestmove'):best=l.split()[1];break
    return {'bestmove':best,'lines':latest}
def normalize(line):
    line=line.strip()
    if not line or line.startswith('#'):return None
    if line.startswith('position '):return ' '.join(line.split())
    return 'position fen '+line if '/' in line else None
def key(pos,reply):return hashlib.sha256((pos+'\n'+reply).encode()).hexdigest()[:24]
def group(pos):return hashlib.sha256(pos.encode()).hexdigest()[:24]
def main():
    a=argparse.ArgumentParser();a.add_argument('--engine',required=True);a.add_argument('--opponent-engine',default=None);a.add_argument('--positions',required=True);a.add_argument('--output',required=True);a.add_argument('--reply-nodes',type=int,default=12000);a.add_argument('--fast-nodes',type=int,default=50000);a.add_argument('--deep-nodes',type=int,default=800000);a.add_argument('--multipv',type=int,default=4);a.add_argument('--regret-threshold',type=int,default=25);a.add_argument('--threads',type=int,default=1);a.add_argument('--hash',type=int,default=64);a.add_argument('--limit',type=int,default=0);x=a.parse_args();o=Path(x.output);o.parent.mkdir(parents=True,exist_ok=True);done=set()
    if o.exists():
        for line in o.read_text(encoding='utf-8').splitlines():
            try:done.add(json.loads(line).get('key'))
            except Exception:pass
    opp=init(x.opponent_engine or x.engine,'miner-opponent',1,min(x.hash,32));fast=init(x.engine,'miner-fast',x.threads,x.hash);deep=init(x.engine,'miner-deep',x.threads,x.hash);verify=init(x.engine,'miner-verify',x.threads,x.hash);positions=[p for p in map(normalize,Path(x.positions).read_text(encoding='utf-8').splitlines()) if p];positions=positions[:x.limit] if x.limit else positions;rows=posn=skip=0
    try:
        with o.open('a',encoding='utf-8') as f:
            for pi,pos in enumerate(positions):
                rs=search(opp,pos,x.reply_nodes,x.multipv);lines=rs['lines']
                if not lines:continue
                bestcp=max(int(v.get('score_cp',0)) for v in lines.values())
                for m in sorted(lines):
                    info=lines[m];pv=info.get('pv') or []
                    if not pv:continue
                    reply=pv[0];k=key(pos,reply)
                    if k in done:skip+=1;continue
                    branch=append_move_to_position(pos,reply);fr=search(fast,branch,x.fast_nodes);dr=search(deep,branch,x.deep_nodes);fm,dm=fr['bestmove'],dr['bestmove'];db=int(dr['lines'].get(1,{}).get('score_cp',0));fc=db
                    if fm and dm and fm!=dm:fc=int(search(verify,branch,x.deep_nodes,1,fm)['lines'].get(1,{}).get('score_cp',db))
                    regret=max(0,db-fc);risk=1. if regret>=x.regret_threshold else 0.;posn+=int(risk);feat=ReplyFeatures(float(m),float(info.get('score_cp',0)),max(0.,float(bestcp-int(info.get('score_cp',0)))),float(info.get('depth',0)),float(info.get('seldepth',0)),float(info.get('nodes',0)),float(info.get('nps',0)),float(info.get('hashfull',0)),float(len(pv)),1. if reply==rs.get('bestmove') else 0.,1. if info.get('score_mate') is not None else 0.).vector();row={'key':k,'group_id':group(pos),'position':pos,'reply':reply,'features':feat,'reply_label':1. if reply==rs.get('bestmove') else 0.,'risk_label':risk,'regret_cp':regret,'fast_move':fm,'deep_move':dm,'deep_best_cp':db,'deep_fast_move_cp':fc,'fast_nodes':x.fast_nodes,'deep_nodes':x.deep_nodes,'reply_nodes':x.reply_nodes,'regret_threshold':x.regret_threshold,'source':'finite_compute_miner'};f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();done.add(k);rows+=1;print(json.dumps({'position':pi,'reply':reply,'reply_label':int(row['reply_label']),'regret_cp':regret,'risk':int(risk)}),flush=True)
    finally:
        for e in (opp,fast,deep,verify):e.close()
    print(json.dumps({'new_rows':rows,'positive_risk':posn,'skipped':skip,'output':str(o)},indent=2))
if __name__=='__main__':main()
