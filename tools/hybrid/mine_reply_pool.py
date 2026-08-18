#!/usr/bin/env python3
"""Mine the full live reply candidate pool cheaply.

This deliberately does NOT deep-label finite-compute risk. It exists so the reply
head sees the same width (default MultiPV 8) that the live P18 controller ranks.
The expensive risk miner can stay at MultiPV 4.

Resume semantics are position-level, not merely row-level: completed decisions are
skipped before launching either Stockfish search. New runs append explicit completion
markers. Older files without markers are recognized as complete when ranks 1..MultiPV
are already present, so existing full-width work is not recomputed.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
try:
    from gpu_risk_model import ReplyFeatures
    from mine_finite_compute import SearchWorker, normalize
except ImportError:
    from .gpu_risk_model import ReplyFeatures
    from .mine_finite_compute import SearchWorker, normalize

def key(pos,reply): return hashlib.sha256((pos+'\n'+reply).encode()).hexdigest()[:24]
def decision(pos): return hashlib.sha256(('decision:'+pos).encode()).hexdigest()[:24]

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--opponent-engine',required=True);a.add_argument('--positions',required=True);a.add_argument('--output',required=True)
    a.add_argument('--reply-nodes',type=int,default=12000);a.add_argument('--opponent-label-nodes',type=int,default=50000)
    a.add_argument('--multipv',type=int,default=8);a.add_argument('--hash',type=int,default=32);a.add_argument('--limit',type=int,default=0)
    a.add_argument('--ready-timeout',type=float,default=30.0);a.add_argument('--search-timeout',type=float,default=180.0);a.add_argument('--worker-retries',type=int,default=2)
    x=a.parse_args()
    out=Path(x.output);out.parent.mkdir(parents=True,exist_ok=True)
    done=set();completed=set();existing={}
    if out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line)
            except Exception:continue
            if r.get('event')=='reply_pool_decision_complete':
                if int(r.get('multipv',-1))==x.multipv and int(r.get('reply_nodes',-1))==x.reply_nodes and int(r.get('opponent_label_nodes',-1))==x.opponent_label_nodes:
                    did=r.get('decision_id')
                    if did:completed.add(str(did))
                continue
            k=r.get('key')
            if k:done.add(k)
            did=r.get('decision_id')
            if not did:continue
            st=existing.setdefault(str(did),{'ranks':set(),'covered':None,'rows':0})
            try:st['ranks'].add(int(r.get('rank')))
            except Exception:pass
            if r.get('reply_pool_covered') is not None:st['covered']=bool(r.get('reply_pool_covered'))
            st['rows']+=1
    full_ranks=set(range(1,x.multipv+1))
    for did,st in existing.items():
        if full_ranks.issubset(st['ranks']):completed.add(did)

    positions=[p for p in map(normalize,Path(x.positions).read_text(encoding='utf-8').splitlines()) if p]
    if x.limit:positions=positions[:x.limit]
    relevant={decision(pos) for pos,_ in positions}
    completed &= relevant
    print(json.dumps({'event':'reply_pool_resume','completed_decisions':len(completed),'total_positions':len(positions),'output':str(out)}),flush=True)

    kw=dict(ready_timeout=x.ready_timeout,search_timeout=x.search_timeout,retries=x.worker_retries)
    probe=truth=None
    new=skipped_rows=skipped_decisions=misses=covered=decisions=0
    try:
        with out.open('a',encoding='utf-8') as f:
            for pi,(pos,gid) in enumerate(positions):
                did=decision(pos)
                if did in completed:
                    skipped_decisions+=1;decisions+=1
                    cov=existing.get(did,{}).get('covered')
                    if cov is True:covered+=1
                    elif cov is False:misses+=1
                    continue
                if probe is None:
                    probe=SearchWorker(x.opponent_engine,'reply-pool-probe',1,x.hash,**kw)
                    truth=SearchWorker(x.opponent_engine,'reply-pool-truth',1,x.hash,**kw)
                try:
                    rs=probe.search(pos,x.reply_nodes,x.multipv);lines=rs['lines']
                    if not lines:continue
                    actual=truth.search(pos,x.opponent_label_nodes,1).get('bestmove')
                except Exception as exc:
                    print(json.dumps({'event':'reply_pool_position_skip','position':pi,'group_id':gid,'error':repr(exc)}),flush=True);continue
                candidates=[]
                for m in sorted(lines):
                    info=lines[m];pv=info.get('pv') or []
                    if pv:candidates.append((m,info,pv[0],pv))
                if not candidates:continue
                decisions+=1
                is_covered=any(move==actual for _,_,move,_ in candidates)
                covered+=int(is_covered);misses+=int(not is_covered)
                bestcp=max(int(info.get('score_cp',0)) for _,info,_,_ in candidates)
                for m,info,move,pv in candidates:
                    k=key(pos,move)
                    if k in done:skipped_rows+=1;continue
                    feat=ReplyFeatures(float(m),float(info.get('score_cp',0)),max(0.,float(bestcp-int(info.get('score_cp',0)))),float(info.get('depth',0)),float(info.get('seldepth',0)),float(info.get('nodes',0)),float(info.get('nps',0)),float(info.get('hashfull',0)),float(len(pv)),1. if move==rs.get('bestmove') else 0.,1. if info.get('score_mate') is not None else 0.).vector()
                    row={'key':k,'group_id':gid,'decision_id':did,'position':pos,'reply':move,'rank':int(m),'features':feat,'reply_label':1. if move==actual else 0.,'risk_label':None,'regret_cp':None,'reply_pool_covered':is_covered,'opponent_label_move':actual,'reply_nodes':x.reply_nodes,'opponent_label_nodes':x.opponent_label_nodes,'source':'reply_pool_v4'}
                    f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();done.add(k);new+=1
                marker={'event':'reply_pool_decision_complete','decision_id':did,'position':pos,'group_id':gid,'candidate_count':len(candidates),'covered':is_covered,'multipv':x.multipv,'reply_nodes':x.reply_nodes,'opponent_label_nodes':x.opponent_label_nodes}
                f.write(json.dumps(marker,sort_keys=True)+'\n');f.flush();completed.add(did)
                print(json.dumps({'event':'reply_pool','position':pi,'group_id':gid,'covered':is_covered,'actual':actual,'candidates':len(candidates)}),flush=True)
    finally:
        if probe is not None:probe.close()
        if truth is not None:truth.close()
    print(json.dumps({'new_rows':new,'skipped_existing_rows':skipped_rows,'skipped_completed_decisions':skipped_decisions,'decisions':decisions,'covered':covered,'misses':misses,'coverage':covered/decisions if decisions else None,'output':str(out)},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
