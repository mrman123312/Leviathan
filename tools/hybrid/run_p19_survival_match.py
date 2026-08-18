#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine

MATE_SCORE = 100000

LOSS_SENTINELS = [
    {"name":"p18.6-game1","fen":"rnbqkb1r/ppp2ppp/5n2/8/3p4/2NQ2P1/PPP1PP1P/R1B1KBNR w KQkq - 0 6","leviathan_white":True},
    {"name":"p18.7-game24","fen":"rnbqk2r/pp3ppp/2pbpn2/3p4/2PP4/4PN1P/PP3PP1/RNBQKB1R w KQkq - 1 6","leviathan_white":False},
    {"name":"p18.7-game27","fen":"rnbqkb1r/pppnp2p/4p1p1/3p4/8/P1N5/1PPP1PPP/R1BQKBNR w KQkq - 0 6","leviathan_white":True},
]

def jprint(obj: Any) -> None:
    print(json.dumps(obj, sort_keys=True), flush=True)

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def safe_quit(e):
    if e is None:return
    try:e.quit()
    except Exception:
        try:e.close()
        except Exception:pass

def configure(e,threads,hash_mb):
    opts={}
    if 'Threads' in e.options:opts['Threads']=threads
    if 'Hash' in e.options:opts['Hash']=hash_mb
    if 'UCI_ShowWDL' in e.options:opts['UCI_ShowWDL']=True
    if opts:e.configure(opts)

def cp(info,turn):
    s=info.get('score')
    if s is None:return -MATE_SCORE
    v=s.pov(turn).score(mate_score=MATE_SCORE)
    return int(v if v is not None else -MATE_SCORE)

def infos_list(x):return x if isinstance(x,list) else [x]

def analyse(engine,board,seconds,multipv,root_moves=None,game=None):
    kwargs={'multipv':max(1,multipv)}
    if root_moves:kwargs['root_moves']=root_moves
    if game is not None:kwargs['game']=game
    return infos_list(engine.analyse(board,chess.engine.Limit(time=seconds),**kwargs))

def move_scores(infos,board):
    out={}
    for info in infos:
        pv=info.get('pv') or []
        if not pv:continue
        m=pv[0];v=cp(info,board.turn)
        if m not in out or v>out[m]:out[m]=v
    return out

def immediate_claim_draw(board,move):
    if move not in board.legal_moves:return False
    b=board.copy(stack=True);b.push(move)
    return b.can_claim_threefold_repetition() or b.can_claim_fifty_moves() or b.is_stalemate()

@dataclass
class Candidate:
    move: chess.Move
    primary:int
    guardian:int
    lower:float
    disagreement:int
    support:int
    immediate_draw:bool
    verifier:int|None=None
    final:float=-1e9

class SurvivalFunnel:
    def __init__(self,engine_path,total_threads,hash_mb,broad_ms,verify_ms,missing_penalty,disagreement_weight,draw_lock_cp):
        self.total_threads=max(2,total_threads)
        self.primary_threads=max(1,self.total_threads*2//3)
        self.guardian_threads=max(1,self.total_threads-self.primary_threads)
        self.broad_s=broad_ms/1000.0;self.verify_s=verify_ms/1000.0
        self.missing_penalty=missing_penalty;self.disagreement_weight=disagreement_weight;self.draw_lock_cp=draw_lock_cp
        self.primary=chess.engine.SimpleEngine.popen_uci(engine_path,timeout=30.0)
        self.guardian=chess.engine.SimpleEngine.popen_uci(engine_path,timeout=30.0)
        self.verifier=chess.engine.SimpleEngine.popen_uci(engine_path,timeout=30.0)
        configure(self.primary,self.primary_threads,max(16,hash_mb*2//3))
        configure(self.guardian,self.guardian_threads,max(16,hash_mb//3))
        configure(self.verifier,self.total_threads,hash_mb)
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=2)
    def close(self):
        self.executor.shutdown(wait=False,cancel_futures=True)
        safe_quit(self.primary);safe_quit(self.guardian);safe_quit(self.verifier)
    def choose(self,board,token):
        t0=time.perf_counter()
        f1=self.executor.submit(analyse,self.primary,board.copy(stack=True),self.broad_s,4,None,token)
        f2=self.executor.submit(analyse,self.guardian,board.copy(stack=True),self.broad_s,6,None,token)
        p_infos=f1.result(timeout=max(5.0,self.broad_s+4.0));g_infos=f2.result(timeout=max(5.0,self.broad_s+4.0))
        p=move_scores(p_infos,board);g=move_scores(g_infos,board)
        if not p and not g:raise RuntimeError('survival broad phase produced no candidate moves')
        p_best=max(p.values()) if p else -MATE_SCORE;g_best=max(g.values()) if g else -MATE_SCORE
        union=list(dict.fromkeys(list(p.keys())+list(g.keys())))
        candidates=[]
        for m in union:
            ps=p.get(m,p_best-self.missing_penalty);gs=g.get(m,g_best-self.missing_penalty)
            support=int(m in p)+int(m in g);disagreement=abs(ps-gs)
            lower=min(ps,gs)-self.disagreement_weight*disagreement
            candidates.append(Candidate(m,ps,gs,lower,disagreement,support,immediate_claim_draw(board,m)))
        viable=[c for c in candidates if c.primary>-MATE_SCORE//2 and c.guardian>-MATE_SCORE//2]
        if viable:candidates=viable
        candidates.sort(key=lambda c:(c.lower,c.support,max(c.primary,c.guardian)),reverse=True)
        finalists=candidates[:min(2,len(candidates))]
        draw_candidates=[c for c in candidates if c.immediate_draw]
        if draw_candidates and candidates[0].lower<=self.draw_lock_cp:
            chosen=max(draw_candidates,key=lambda c:c.lower);chosen.verifier=0;chosen.final=max(chosen.lower,0.0)
            return chosen.move,self._telemetry(t0,p_best,g_best,candidates,finalists,chosen,'draw_lock')
        roots=[c.move for c in finalists]
        v_infos=analyse(self.verifier,board,self.verify_s,len(roots),roots,token);v=move_scores(v_infos,board)
        v_best=max(v.values()) if v else -MATE_SCORE
        for c in finalists:
            c.verifier=v.get(c.move,v_best-self.missing_penalty)
            spread=max(c.primary,c.guardian,c.verifier)-min(c.primary,c.guardian,c.verifier)
            c.final=min(c.primary,c.guardian,c.verifier)-0.25*spread
        non_mated=[c for c in finalists if c.verifier is not None and c.verifier>-MATE_SCORE//2]
        pool=non_mated if non_mated else finalists
        chosen=max(pool,key=lambda c:(c.final,c.support,c.lower))
        return chosen.move,self._telemetry(t0,p_best,g_best,candidates,finalists,chosen,'verified')
    def _telemetry(self,t0,p_best,g_best,candidates,finalists,chosen,reason):
        def row(c):
            return {'move':c.move.uci(),'primary_cp':c.primary,'guardian_cp':c.guardian,'lower_cp':round(c.lower,2),'disagreement_cp':c.disagreement,'support':c.support,'immediate_draw':c.immediate_draw,'verifier_cp':c.verifier,'final_cp':None if c.final<-1e8 else round(c.final,2)}
        return {'reason':reason,'elapsed_ms':int((time.perf_counter()-t0)*1000),'thread_split':[self.primary_threads,self.guardian_threads,self.total_threads],'primary_best_cp':p_best,'guardian_best_cp':g_best,'chosen':chosen.move.uci(),'candidates':[row(c) for c in candidates[:8]],'finalists':[row(c) for c in finalists]}

def weighted_choice(infos,board,rng,temp_cp):
    rows=[]
    for info in infos:
        pv=info.get('pv') or []
        if pv:rows.append((pv[0],float(cp(info,board.turn))))
    if not rows:return next(iter(board.legal_moves))
    best=max(v for _,v in rows);weights=[math.exp(max(-20.0,min(0.0,(v-best)/max(1.0,temp_cp)))) for _,v in rows]
    return rng.choices([m for m,_ in rows],weights=weights,k=1)[0]

def generate_openings(sf_path,out,count,plies,seed,nodes):
    if out.exists():
        fens=[x.strip() for x in out.read_text(encoding='utf-8').splitlines() if x.strip()]
        if len(fens)>=count:return fens[:count]
    rng=random.Random(seed);sf=chess.engine.SimpleEngine.popen_uci(sf_path,timeout=30.0);configure(sf,1,32);fens=[]
    try:
        for i in range(count):
            b=chess.Board()
            for _ in range(plies):
                if b.is_game_over(claim_draw=True):break
                n=max(1,min(4,b.legal_moves.count()))
                infos=infos_list(sf.analyse(b,chess.engine.Limit(nodes=nodes),multipv=n))
                m=weighted_choice(infos,b,rng,55.0)
                if m not in b.legal_moves:break
                b.push(m)
            if b.is_game_over(claim_draw=True):b=chess.Board()
            fens.append(b.fen());jprint({'event':'opening','index':i+1,'fen':b.fen()})
    finally:safe_quit(sf)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(fens)+'\n',encoding='utf-8');return fens

def score_from_result(result,leviathan_white):
    if result=='1/2-1/2':return 0.5
    return 1.0 if ((result=='1-0')==leviathan_white) else 0.0

def play_game(funnel,sf,fen,leviathan_white,token,movetime_ms,max_plies,decision_log):
    board=chess.Board(fen);moves=[];sf_limit=chess.engine.Limit(time=movetime_ms/1000.0)
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):break
        lev_turn=board.turn==(chess.WHITE if leviathan_white else chess.BLACK)
        if lev_turn:
            move,telem=funnel.choose(board,token);telem.update({'game':token,'ply':ply+1,'fen':board.fen()})
            with decision_log.open('a',encoding='utf-8') as f:f.write(json.dumps(telem,sort_keys=True)+'\n')
        else:
            r=sf.play(board,sf_limit,game=token,ponder=False);move=r.move
        if move is None or move not in board.legal_moves:raise RuntimeError(f'illegal/no move at ply {ply+1}: {move} fen={board.fen()}')
        moves.append(move.uci());board.push(move)
    outcome=board.outcome(claim_draw=True);result='1/2-1/2' if outcome is None else outcome.result();term='MAX_PLIES' if outcome is None else outcome.termination.name
    return {'opening_fen':fen,'leviathan_white':leviathan_white,'result':result,'score_leviathan':score_from_result(result,leviathan_white),'termination':term,'plies':len(moves),'moves':moves}

def summary(rows):
    w=sum(r['score_leviathan']==1.0 for r in rows);d=sum(r['score_leviathan']==0.5 for r in rows);l=sum(r['score_leviathan']==0.0 for r in rows)
    return {'games':len(rows),'wins':w,'draws':d,'losses':l,'score':sum(r['score_leviathan'] for r in rows)/len(rows) if rows else 0.0}

def append_jsonl(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()

def main():
    ap=argparse.ArgumentParser(description='P19 survival-funnel zero-loss experiment')
    ap.add_argument('--engine',required=True);ap.add_argument('--opponent-engine',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--games',type=int,default=100);ap.add_argument('--threads',type=int,default=6);ap.add_argument('--hash',type=int,default=128);ap.add_argument('--movetime-ms',type=int,default=500)
    ap.add_argument('--broad-ms',type=int,default=320);ap.add_argument('--verify-ms',type=int,default=180);ap.add_argument('--max-plies',type=int,default=240);ap.add_argument('--opening-plies',type=int,default=10);ap.add_argument('--opening-nodes',type=int,default=1500);ap.add_argument('--seed',type=int,default=20260818);ap.add_argument('--sentinel-repeats',type=int,default=3)
    ap.add_argument('--missing-penalty',type=int,default=120);ap.add_argument('--disagreement-weight',type=float,default=0.35);ap.add_argument('--draw-lock-cp',type=int,default=80)
    a=ap.parse_args()
    if a.broad_ms+a.verify_ms!=a.movetime_ms:raise SystemExit('broad-ms + verify-ms must equal movetime-ms for fixed compute budget')
    if a.games<=0 or a.games%2:raise SystemExit('games must be positive and even')
    identity={'version':'p19-survival-funnel-v1','engine_sha256':sha256_file(Path(a.engine)),'stockfish_sha256':sha256_file(Path(a.opponent_engine)),'games':a.games,'threads':a.threads,'hash':a.hash,'movetime_ms':a.movetime_ms,'broad_ms':a.broad_ms,'verify_ms':a.verify_ms,'seed':a.seed,'sentinel_repeats':a.sentinel_repeats}
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12];out=Path(a.out_dir)/run_id;out.mkdir(parents=True,exist_ok=True);(out/'manifest.json').write_text(json.dumps(identity,indent=2,sort_keys=True),encoding='utf-8');jprint({'event':'P19_CONFIG','run_id':run_id,**identity})
    funnel=SurvivalFunnel(a.engine,a.threads,a.hash,a.broad_ms,a.verify_ms,a.missing_penalty,a.disagreement_weight,a.draw_lock_cp);sf=chess.engine.SimpleEngine.popen_uci(a.opponent_engine,timeout=30.0);configure(sf,a.threads,a.hash);rows=[];decision_log=out/'survival-decisions.jsonl'
    try:
        jprint({'event':'P19_SENTINEL_GATE_START','cases':len(LOSS_SENTINELS),'repeats':a.sentinel_repeats});sidx=0
        for case in LOSS_SENTINELS:
            for rep in range(1,a.sentinel_repeats+1):
                sidx+=1;r=play_game(funnel,sf,case['fen'],bool(case['leviathan_white']),f'sentinel-{sidx}',a.movetime_ms,a.max_plies,decision_log);r.update({'sentinel':case['name'],'repeat':rep});append_jsonl(out/'sentinels.jsonl',r);jprint({'event':'P19_SENTINEL_COMPLETE',**r})
                if r['score_leviathan']==0.0:jprint({'event':'P19_SENTINEL_FAILURE',**r});return 21
        jprint({'event':'P19_SENTINEL_GATE_PASSED','games':sidx,'losses':0})
        openings=generate_openings(a.opponent_engine,out/'openings.fen',a.games//2,a.opening_plies,a.seed,a.opening_nodes)
        for game in range(1,a.games+1):
            fen=openings[(game-1)//2];lev_white=game%2==1;r=play_game(funnel,sf,fen,lev_white,f'p19-{game}',a.movetime_ms,a.max_plies,decision_log);r['game']=game;append_jsonl(out/'games.jsonl',r);rows.append(r);jprint({'event':'P19_GAME_COMPLETE',**r,'cumulative':summary(rows)})
            if r['score_leviathan']==0.0:
                failure={'event':'P19_ZERO_LOSS_GATE_FAILED','game':game,'opening_fen':fen,'leviathan_white':lev_white,'result':r['result'],'moves':r['moves'],'summary':summary(rows)};(out/'failure.json').write_text(json.dumps(failure,indent=2,sort_keys=True),encoding='utf-8');jprint(failure);return 31
        final={'event':'P19_ZERO_LOSS_GATE_PASSED','run_id':run_id,'summary':summary(rows)};(out/'summary.json').write_text(json.dumps(final,indent=2,sort_keys=True),encoding='utf-8');jprint(final);return 0
    finally:funnel.close();safe_quit(sf)

if __name__=='__main__':raise SystemExit(main())
