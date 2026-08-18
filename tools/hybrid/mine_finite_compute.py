#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,queue,time
from pathlib import Path
try:from gpu_risk_model import ReplyFeatures
except ImportError:from .gpu_risk_model import ReplyFeatures
try:from leviathan_hybrid_uci import EngineProcess,append_move_to_position,parse_info
except ImportError:from .leviathan_hybrid_uci import EngineProcess,append_move_to_position,parse_info

class SearchWorker:
    def __init__(self,path,label,threads,h,ready_timeout=30.0,search_timeout=180.0,retries=2):
        self.path=path;self.label=label;self.threads=threads;self.hash=h;self.ready_timeout=float(ready_timeout);self.search_timeout=float(search_timeout);self.retries=max(0,int(retries));self.e=None;self.restart('initial')
    def restart(self,reason):
        if self.e is not None:
            try:self.e.close()
            except Exception:pass
        self.e=EngineProcess(self.path,self.label);self.e.send('uci');self.e.wait_for('uciok',self.ready_timeout);self.e.initialized=True;self.e.send(f'setoption name Threads value {self.threads}');self.e.send(f'setoption name Hash value {self.hash}');self.e.send('isready');self.e.wait_for('readyok',self.ready_timeout);self.e.options['Threads']=str(self.threads);self.e.options['Hash']=str(self.hash)
        if reason!='initial':print(json.dumps({'event':'worker_restarted','worker':self.label,'reason':reason}),flush=True)
    def close(self):
        if self.e is not None:self.e.close()
    def _search_once(self,pos,nodes,multipv=1,searchmove=None):
        e=self.e;assert e is not None;e.drain()
        try:e.send('stop')
        except Exception:pass
        e.send('ucinewgame');e.send('isready');e.wait_for('readyok',self.ready_timeout);e.send(f'setoption name MultiPV value {multipv}');e.send('isready');e.wait_for('readyok',self.ready_timeout);e.drain();e.send(pos);e.send('go '+(f'searchmoves {searchmove} ' if searchmove else '')+f'nodes {nodes}');latest={};best=None;end=time.monotonic()+self.search_timeout
        while time.monotonic()<end:
            try:l=e.read(1)
            except queue.Empty:continue
            if l=='__LEV_ENGINE_EOF__':raise RuntimeError(f'{self.label}: engine exited during search')
            i=parse_info(l)
            if i and i.get('pv'):
                m=int(i.get('multipv',1));q=latest.get(m)
                if q is None or (i.get('depth',0),i.get('nodes',0))>=(q.get('depth',0),q.get('nodes',0)):latest[m]=i
            if l.startswith('bestmove'):
                p=l.split();best=p[1] if len(p)>1 else None;break
        if best is None:
            try:e.send('stop')
            except Exception:pass
            grace=time.monotonic()+15.0
            while time.monotonic()<grace:
                try:l=e.read(.25)
                except queue.Empty:continue
                if l=='__LEV_ENGINE_EOF__':raise RuntimeError(f'{self.label}: engine exited after stop')
                i=parse_info(l)
                if i and i.get('pv'):
                    m=int(i.get('multipv',1));q=latest.get(m)
                    if q is None or (i.get('depth',0),i.get('nodes',0))>=(q.get('depth',0),q.get('nodes',0)):latest[m]=i
                if l.startswith('bestmove'):
                    p=l.split();best=p[1] if len(p)>1 else None;break
            if best is None:raise TimeoutError(f'{self.label}: no bestmove after timeout and stop grace')
        return {'bestmove':best,'lines':latest}
    def search(self,pos,nodes,multipv=1,searchmove=None):
        last=None
        for attempt in range(self.retries+1):
            try:return self._search_once(pos,nodes,multipv,searchmove)
            except (queue.Empty,TimeoutError,RuntimeError,BrokenPipeError,OSError) as exc:
                last=exc
                if attempt>=self.retries:break
                self.restart(f'{type(exc).__name__}: {exc}')
        raise RuntimeError(f'{self.label}: search failed after {self.retries+1} attempts: {last}')

def normalize(line):
    line=line.strip()
    if not line or line.startswith('#'):return None
    try:
        r=json.loads(line)
        if isinstance(r,dict) and r.get('position'):
            p=' '.join(str(r['position']).split());return (p,str(r.get('group_id') or fallback_group(p)))
    except Exception:pass
    if line.startswith('position '):p=' '.join(line.split());return (p,fallback_group(p))
    if '/' in line:p='position fen '+line;return (p,fallback_group(p))
    return None

def key(pos,reply):return hashlib.sha256((pos+'\n'+reply).encode()).hexdigest()[:24]
def fallback_group(pos):return hashlib.sha256(pos.encode()).hexdigest()[:24]

def main():
    a=argparse.ArgumentParser();a.add_argument('--engine',required=True);a.add_argument('--opponent-engine',default=None);a.add_argument('--positions',required=True);a.add_argument('--output',required=True);a.add_argument('--reply-nodes',type=int,default=12000);a.add_argument('--opponent-label-nodes',type=int,default=50000);a.add_argument('--fast-nodes',type=int,default=50000);a.add_argument('--deep-nodes',type=int,default=800000);a.add_argument('--multipv',type=int,default=4);a.add_argument('--regret-threshold',type=int,default=25);a.add_argument('--threads',type=int,default=1);a.add_argument('--hash',type=int,default=64);a.add_argument('--limit',type=int,default=0);a.add_argument('--ready-timeout',type=float,default=30.0);a.add_argument('--search-timeout',type=float,default=180.0);a.add_argument('--worker-retries',type=int,default=2);a.add_argument('--max-consecutive-skips',type=int,default=12);x=a.parse_args();o=Path(x.output);o.parent.mkdir(parents=True,exist_ok=True);done=set()
    if o.exists():
        for line in o.read_text(encoding='utf-8').splitlines():
            try:
                k=json.loads(line).get('key')
                if k:done.add(k)
            except Exception:pass
    kw=dict(ready_timeout=x.ready_timeout,search_timeout=x.search_timeout,retries=x.worker_retries);opp=SearchWorker(x.opponent_engine or x.engine,'miner-opponent-probe',1,min(x.hash,32),**kw);opp_label=SearchWorker(x.opponent_engine or x.engine,'miner-opponent-label',1,min(x.hash,32),**kw);fast=SearchWorker(x.engine,'miner-fast',x.threads,x.hash,**kw);deep=SearchWorker(x.engine,'miner-deep',x.threads,x.hash,**kw);verify=SearchWorker(x.engine,'miner-verify',x.threads,x.hash,**kw);positions=[p for p in map(normalize,Path(x.positions).read_text(encoding='utf-8').splitlines()) if p];positions=positions[:x.limit] if x.limit else positions;rows=posn=skip=errors=attempted=consecutive_skips=0
    try:
        with o.open('a',encoding='utf-8') as f:
            for pi,(pos,gid) in enumerate(positions):
                try:
                    rs=opp.search(pos,x.reply_nodes,x.multipv);lines=rs['lines']
                    if not lines:continue
                    label_move=opp_label.search(pos,x.opponent_label_nodes,1).get('bestmove')
                except Exception as exc:
                    errors+=1;consecutive_skips+=1;print(json.dumps({'event':'position_skip','position':pi,'group_id':gid,'error':repr(exc)}),flush=True)
                    if consecutive_skips>=x.max_consecutive_skips:raise RuntimeError(f'aborting after {consecutive_skips} consecutive position failures')
                    continue
                bestcp=max(int(v.get('score_cp',0)) for v in lines.values())
                for m in sorted(lines):
                    info=lines[m];pv=info.get('pv') or []
                    if not pv:continue
                    reply=pv[0];k=key(pos,reply)
                    if k in done:skip+=1;continue
                    attempted+=1
                    try:
                        branch=append_move_to_position(pos,reply);fr=fast.search(branch,x.fast_nodes);dr=deep.search(branch,x.deep_nodes);fm,dm=fr['bestmove'],dr['bestmove'];db=int(dr['lines'].get(1,{}).get('score_cp',0));fc=db
                        if fm and dm and fm!=dm:
                            vr=verify.search(branch,x.deep_nodes,1,fm);vline=vr['lines'].get(1,{})
                            if not vline:raise RuntimeError('verification search returned no principal variation')
                            fc=int(vline.get('score_cp',db))
                    except Exception as exc:
                        errors+=1;consecutive_skips+=1;print(json.dumps({'event':'candidate_skip','position':pi,'group_id':gid,'reply':reply,'error':repr(exc)}),flush=True)
                        if consecutive_skips>=x.max_consecutive_skips:raise RuntimeError(f'aborting after {consecutive_skips} consecutive candidate failures')
                        continue
                    consecutive_skips=0;regret=max(0,db-fc);risk=1. if regret>=x.regret_threshold else 0.;posn+=int(risk);feat=ReplyFeatures(float(m),float(info.get('score_cp',0)),max(0.,float(bestcp-int(info.get('score_cp',0)))),float(info.get('depth',0)),float(info.get('seldepth',0)),float(info.get('nodes',0)),float(info.get('nps',0)),float(info.get('hashfull',0)),float(len(pv)),1. if reply==rs.get('bestmove') else 0.,1. if info.get('score_mate') is not None else 0.).vector();row={'key':k,'group_id':gid,'position':pos,'reply':reply,'features':feat,'reply_label':1. if reply==label_move else 0.,'risk_label':risk,'regret_cp':regret,'fast_move':fm,'deep_move':dm,'deep_best_cp':db,'deep_fast_move_cp':fc,'fast_nodes':x.fast_nodes,'deep_nodes':x.deep_nodes,'reply_nodes':x.reply_nodes,'opponent_label_nodes':x.opponent_label_nodes,'opponent_label_move':label_move,'regret_threshold':x.regret_threshold,'source':'finite_compute_miner_v3_resilient'};f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();done.add(k);rows+=1;print(json.dumps({'position':pi,'group_id':gid,'reply':reply,'reply_label':int(row['reply_label']),'regret_cp':regret,'risk':int(risk)}),flush=True)
    finally:
        for w in (opp,opp_label,fast,deep,verify):
            try:w.close()
            except Exception:pass
    print(json.dumps({'new_rows':rows,'positive_risk':posn,'skipped_existing':skip,'recoverable_errors':errors,'attempted_new_candidates':attempted,'output':str(o)},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
