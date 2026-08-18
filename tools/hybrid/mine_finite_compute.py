#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, queue, time
from pathlib import Path
try:
    from gpu_risk_model import ReplyFeatures
except ImportError:
    from .gpu_risk_model import ReplyFeatures
try:
    from leviathan_hybrid_uci import EngineProcess, append_move_to_position, parse_info
except ImportError:
    from .leviathan_hybrid_uci import EngineProcess, append_move_to_position, parse_info


class SearchWorker:
    def __init__(self, path, label, threads, h, ready_timeout=30.0, search_timeout=180.0, retries=2):
        self.path=path;self.label=label;self.threads=threads;self.hash=h
        self.ready_timeout=float(ready_timeout);self.search_timeout=float(search_timeout);self.retries=max(0,int(retries))
        self.e=None;self.restart('initial')
    def restart(self,reason):
        if self.e is not None:
            try:self.e.close()
            except Exception:pass
        self.e=EngineProcess(self.path,self.label);self.e.send('uci');self.e.wait_for('uciok',self.ready_timeout);self.e.initialized=True
        self.e.send(f'setoption name Threads value {self.threads}');self.e.send(f'setoption name Hash value {self.hash}')
        self.e.send('isready');self.e.wait_for('readyok',self.ready_timeout);self.e.options['Threads']=str(self.threads);self.e.options['Hash']=str(self.hash)
        if reason!='initial':print(json.dumps({'event':'worker_restarted','worker':self.label,'reason':reason}),flush=True)
    def close(self):
        if self.e is not None:self.e.close()
    def _search_once(self,pos,nodes,multipv=1,searchmove=None):
        e=self.e;assert e is not None;e.drain()
        try:e.send('stop')
        except Exception:pass
        e.send('ucinewgame');e.send('isready');e.wait_for('readyok',self.ready_timeout)
        e.send(f'setoption name MultiPV value {multipv}');e.send('isready');e.wait_for('readyok',self.ready_timeout);e.drain();e.send(pos)
        e.send('go '+(f'searchmoves {searchmove} ' if searchmove else '')+f'nodes {nodes}')
        latest={};best=None;end=time.monotonic()+self.search_timeout
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


def fallback_group(pos):return hashlib.sha256(pos.encode()).hexdigest()[:24]
def decision(pos):return hashlib.sha256(('decision:'+pos).encode()).hexdigest()[:24]
def key(pos,reply):return hashlib.sha256((pos+'\n'+reply).encode()).hexdigest()[:24]

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


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--engine',required=True);a.add_argument('--opponent-engine',default=None);a.add_argument('--positions',required=True);a.add_argument('--output',required=True)
    a.add_argument('--reply-nodes',type=int,default=12000);a.add_argument('--opponent-label-nodes',type=int,default=50000)
    a.add_argument('--fast-nodes',type=int,default=50000);a.add_argument('--deep-nodes',type=int,default=800000);a.add_argument('--multipv',type=int,default=4)
    a.add_argument('--regret-threshold',type=int,default=25);a.add_argument('--threads',type=int,default=1);a.add_argument('--hash',type=int,default=64);a.add_argument('--limit',type=int,default=0)
    a.add_argument('--ready-timeout',type=float,default=30.0);a.add_argument('--search-timeout',type=float,default=180.0);a.add_argument('--worker-retries',type=int,default=2);a.add_argument('--max-consecutive-skips',type=int,default=12)
    x=a.parse_args()
    o=Path(x.output);o.parent.mkdir(parents=True,exist_ok=True)

    done=set();completed=set();existing={}
    if o.exists():
        for line in o.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line)
            except Exception:continue
            if r.get('event')=='finite_compute_decision_complete':
                config_ok=(int(r.get('multipv',-1))==x.multipv and int(r.get('reply_nodes',-1))==x.reply_nodes and int(r.get('opponent_label_nodes',-1))==x.opponent_label_nodes and int(r.get('fast_nodes',-1))==x.fast_nodes and int(r.get('deep_nodes',-1))==x.deep_nodes and int(r.get('regret_threshold',-1))==x.regret_threshold)
                if config_ok and r.get('decision_id'):completed.add(str(r['decision_id']))
                continue
            k=r.get('key')
            if k:done.add(k)
            pos=r.get('position')
            if not pos:continue
            did=str(r.get('decision_id') or decision(str(pos)))
            st=existing.setdefault(did,{'ranks':set(),'rows':0})
            rank=r.get('rank')
            if rank is None:
                feat=r.get('features') or []
                rank=feat[0] if feat else None
            try:st['ranks'].add(int(round(float(rank))))
            except Exception:pass
            st['rows']+=1
    full_ranks=set(range(1,x.multipv+1))
    for did,st in existing.items():
        if full_ranks.issubset(st['ranks']):completed.add(did)

    positions=[p for p in map(normalize,Path(x.positions).read_text(encoding='utf-8').splitlines()) if p]
    positions=positions[:x.limit] if x.limit else positions
    relevant={decision(pos) for pos,_ in positions};completed &= relevant
    print(json.dumps({'event':'finite_compute_resume','completed_decisions':len(completed),'total_positions':len(positions),'existing_candidate_rows':len(done),'output':str(o)}),flush=True)

    kw=dict(ready_timeout=x.ready_timeout,search_timeout=x.search_timeout,retries=x.worker_retries)
    opp=opp_label=fast=deep=verify=None
    rows=posn=skip_rows=skip_decisions=errors=attempted=consecutive_skips=0
    try:
        with o.open('a',encoding='utf-8') as f:
            for pi,(pos,gid) in enumerate(positions):
                did=decision(pos)
                if did in completed:
                    skip_decisions+=1
                    continue
                if opp is None:
                    opp=SearchWorker(x.opponent_engine or x.engine,'miner-opponent-probe',1,min(x.hash,32),**kw)
                    opp_label=SearchWorker(x.opponent_engine or x.engine,'miner-opponent-label',1,min(x.hash,32),**kw)
                    fast=SearchWorker(x.engine,'miner-fast',x.threads,x.hash,**kw)
                    deep=SearchWorker(x.engine,'miner-deep',x.threads,x.hash,**kw)
                    verify=SearchWorker(x.engine,'miner-verify',x.threads,x.hash,**kw)
                try:
                    rs=opp.search(pos,x.reply_nodes,x.multipv);lines=rs['lines']
                    if not lines:continue
                    label_move=opp_label.search(pos,x.opponent_label_nodes,1).get('bestmove')
                except Exception as exc:
                    errors+=1;consecutive_skips+=1;print(json.dumps({'event':'position_skip','position':pi,'group_id':gid,'error':repr(exc)}),flush=True)
                    if consecutive_skips>=x.max_consecutive_skips:raise RuntimeError(f'aborting after {consecutive_skips} consecutive position failures')
                    continue
                bestcp=max(int(v.get('score_cp',0)) for v in lines.values())
                candidate_keys=[];decision_failed=False;candidate_count=0
                for m in sorted(lines):
                    info=lines[m];pv=info.get('pv') or []
                    if not pv:continue
                    candidate_count+=1;reply=pv[0];k=key(pos,reply);candidate_keys.append(k)
                    if k in done:skip_rows+=1;continue
                    attempted+=1
                    try:
                        branch=append_move_to_position(pos,reply);fr=fast.search(branch,x.fast_nodes);dr=deep.search(branch,x.deep_nodes)
                        fm,dm=fr['bestmove'],dr['bestmove'];db=int(dr['lines'].get(1,{}).get('score_cp',0));fc=db
                        if fm and dm and fm!=dm:
                            vr=verify.search(branch,x.deep_nodes,1,fm);vline=vr['lines'].get(1,{})
                            if not vline:raise RuntimeError('verification search returned no principal variation')
                            fc=int(vline.get('score_cp',db))
                    except Exception as exc:
                        decision_failed=True;errors+=1;consecutive_skips+=1
                        print(json.dumps({'event':'candidate_skip','position':pi,'group_id':gid,'reply':reply,'error':repr(exc)}),flush=True)
                        if consecutive_skips>=x.max_consecutive_skips:raise RuntimeError(f'aborting after {consecutive_skips} consecutive candidate failures')
                        continue
                    consecutive_skips=0;regret=max(0,db-fc);risk=1. if regret>=x.regret_threshold else 0.;posn+=int(risk)
                    feat=ReplyFeatures(float(m),float(info.get('score_cp',0)),max(0.,float(bestcp-int(info.get('score_cp',0)))),float(info.get('depth',0)),float(info.get('seldepth',0)),float(info.get('nodes',0)),float(info.get('nps',0)),float(info.get('hashfull',0)),float(len(pv)),1. if reply==rs.get('bestmove') else 0.,1. if info.get('score_mate') is not None else 0.).vector()
                    row={'key':k,'group_id':gid,'decision_id':did,'rank':int(m),'position':pos,'reply':reply,'features':feat,'reply_label':1. if reply==label_move else 0.,'risk_label':risk,'regret_cp':regret,'fast_move':fm,'deep_move':dm,'deep_best_cp':db,'deep_fast_move_cp':fc,'fast_nodes':x.fast_nodes,'deep_nodes':x.deep_nodes,'reply_nodes':x.reply_nodes,'opponent_label_nodes':x.opponent_label_nodes,'opponent_label_move':label_move,'regret_threshold':x.regret_threshold,'source':'finite_compute_miner_v4_resume'}
                    f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();done.add(k);rows+=1
                    print(json.dumps({'position':pi,'group_id':gid,'reply':reply,'reply_label':int(row['reply_label']),'regret_cp':regret,'risk':int(risk)}),flush=True)
                if candidate_count and not decision_failed and all(k in done for k in candidate_keys):
                    marker={'event':'finite_compute_decision_complete','decision_id':did,'position':pos,'group_id':gid,'candidate_count':candidate_count,'multipv':x.multipv,'reply_nodes':x.reply_nodes,'opponent_label_nodes':x.opponent_label_nodes,'fast_nodes':x.fast_nodes,'deep_nodes':x.deep_nodes,'regret_threshold':x.regret_threshold}
                    f.write(json.dumps(marker,sort_keys=True)+'\n');f.flush();completed.add(did)
    finally:
        for w in (opp,opp_label,fast,deep,verify):
            if w is None:continue
            try:w.close()
            except Exception:pass
    print(json.dumps({'new_rows':rows,'positive_risk_new_rows':posn,'skipped_existing_rows':skip_rows,'skipped_completed_decisions':skip_decisions,'recoverable_errors':errors,'attempted_new_candidates':attempted,'output':str(o)},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
