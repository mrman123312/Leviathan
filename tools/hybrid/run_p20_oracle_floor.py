#!/usr/bin/env python3
"""P20 Oracle Floor survival prototype.

Goal: Leviathan may improve on a frozen-Stockfish defensive baseline, but may
never knowingly undercut it. This replaces absolute cp/WDL safety thresholds
with a monotonic baseline-dominance rule.

For every Leviathan move:
  * frozen Stockfish (1 thread, fixed nodes) proposes the defensive floor move;
  * P09 + Stockfish propose challenger moves;
  * every challenger and the floor move are evaluated by a deeper frozen-SF
    hostile verifier, including multiple opponent replies;
  * a challenger is eligible only if its verified worst-case score is no worse
    than the floor move within a tiny epsilon and has no mate-loss evidence;
  * otherwise Leviathan plays the floor move;
  * immediate legal draw claims are locked whenever the floor is not clearly winning.

This is still empirical, not a proof that chess is solved. The key invariant is
monotonic: experimental Leviathan logic cannot select a move that the defensive
oracle judges worse than its own baseline move.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.engine

MATE_SCORE = 100000

LOSS_SENTINELS = [
    {"name":"p18.6-game1","fen":"rnbqkb1r/ppp2ppp/5n2/8/3p4/2NQ2P1/PPP1PP1P/R1B1KBNR w KQkq - 0 6","leviathan_white":True},
    {"name":"p18.7-game24","fen":"rnbqk2r/pp3ppp/2pbpn2/3p4/2PP4/4PN1P/PP3PP1/RNBQKB1R w KQkq - 1 6","leviathan_white":False},
    {"name":"p18.7-game27","fen":"rnbqkb1r/pppnp2p/4p1p1/3p4/8/P1N5/1PPP1PPP/R1BQKBNR w KQkq - 0 6","leviathan_white":True},
    {"name":"p19-v1-ne4-nc6-micro","fen":"r1bqkb1r/ppp2ppp/2n2n2/8/3pN3/3Q2P1/PPP1PP1P/R1B1KBNR w KQkq - 2 7","leviathan_white":True},
]


def jprint(x: Any) -> None:
    print(json.dumps(x, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def safe_quit(e):
    if e is None: return
    try: e.quit()
    except Exception:
        try: e.close()
        except Exception: pass


def configure(e, threads: int, hash_mb: int):
    opts={}
    if 'Threads' in e.options: opts['Threads']=threads
    if 'Hash' in e.options: opts['Hash']=hash_mb
    if 'UCI_ShowWDL' in e.options: opts['UCI_ShowWDL']=True
    if opts: e.configure(opts)


def infos_list(x): return x if isinstance(x,list) else [x]


def score_cp(info: dict[str,Any], color: chess.Color) -> int:
    s=info.get('score')
    if s is None: return -MATE_SCORE
    v=s.pov(color).score(mate_score=MATE_SCORE)
    return int(v if v is not None else -MATE_SCORE)


def mate_loss(info: dict[str,Any], color: chess.Color) -> bool:
    s=info.get('score')
    if s is None: return False
    try:
        p=s.pov(color)
        return p.is_mate() and p.mate() is not None and p.mate() <= 0
    except Exception:
        return score_cp(info,color) <= -MATE_SCORE//2


def analyse(engine, board: chess.Board, nodes: int, multipv: int=1, root_moves=None, game=None):
    kwargs={'multipv':max(1,min(multipv,board.legal_moves.count()))}
    if root_moves is not None: kwargs['root_moves']=root_moves
    if game is not None: kwargs['game']=game
    return infos_list(engine.analyse(board,chess.engine.Limit(nodes=max(1,nodes)),**kwargs))


def root_map(engine, board, nodes, multipv, color, token):
    out={}
    for info in analyse(engine,board,nodes,multipv,None,token):
        pv=info.get('pv') or []
        if not pv: continue
        m=pv[0]
        row={'cp':score_cp(info,color),'mate_loss':mate_loss(info,color),'depth':int(info.get('depth') or 0),'nodes':int(info.get('nodes') or 0)}
        old=out.get(m)
        if old is None or row['cp']>old['cp']: out[m]=row
    return out


def immediate_draw(board,move):
    if move not in board.legal_moves:return False
    b=board.copy(stack=True);b.push(move)
    return b.can_claim_threefold_repetition() or b.can_claim_fifty_moves() or b.is_stalemate()


@dataclass
class VerifiedMove:
    move: chess.Move
    seed_cp: int=-MATE_SCORE
    worst_cp: int=-MATE_SCORE
    direct_cp: int=-MATE_SCORE
    mate_refuted: bool=False
    replies: list[dict[str,Any]]=field(default_factory=list)
    immediate_draw: bool=False


class OracleFloor:
    def __init__(self,p09_path,sf_path,hash_mb,floor_nodes,candidate_nodes,verify_nodes,reply_nodes,continuation_nodes,candidate_mpv,reply_mpv,epsilon_cp,panic_nodes):
        self.p09=chess.engine.SimpleEngine.popen_uci(p09_path,timeout=120.0)
        self.floor=chess.engine.SimpleEngine.popen_uci(sf_path,timeout=120.0)
        self.verify=chess.engine.SimpleEngine.popen_uci(sf_path,timeout=120.0)
        configure(self.p09,1,max(16,hash_mb//3));configure(self.floor,1,max(16,hash_mb//3));configure(self.verify,1,max(16,hash_mb//3))
        self.floor_nodes=floor_nodes;self.candidate_nodes=candidate_nodes;self.verify_nodes=verify_nodes
        self.reply_nodes=reply_nodes;self.continuation_nodes=continuation_nodes;self.candidate_mpv=candidate_mpv;self.reply_mpv=reply_mpv
        self.epsilon_cp=epsilon_cp;self.panic_nodes=panic_nodes;self.serial=0
    def close(self):
        safe_quit(self.p09);safe_quit(self.floor);safe_quit(self.verify)
    def tok(self,p): self.serial+=1;return f'{p}-{self.serial}'
    def _direct(self,board,move,color,nodes,token):
        infos=analyse(self.verify,board,nodes,1,[move],self.tok(token+'-direct'))
        info=infos[0] if infos else {}
        return score_cp(info,color),mate_loss(info,color)
    def _hostile(self,board,move,color,token,deep=False):
        b=board.copy(stack=True);b.push(move)
        if b.is_game_over(claim_draw=True):
            out=b.outcome(claim_draw=True)
            if out is None or out.winner is None:return 0,False,[]
            lost=out.winner != color
            return (-MATE_SCORE if lost else MATE_SCORE),lost,[]
        rn=self.panic_nodes if deep else self.reply_nodes
        cn=self.panic_nodes if deep else self.continuation_nodes
        opp=~color
        reply_infos=analyse(self.floor,b,rn,min(self.reply_mpv,b.legal_moves.count()),None,self.tok(token+'-replies'))
        reply_moves=[]
        for info in reply_infos:
            pv=info.get('pv') or []
            if pv and pv[0] not in reply_moves: reply_moves.append(pv[0])
        # Missing replies never get silently ignored in deep/panic mode.
        if deep:
            for r in b.legal_moves:
                if r not in reply_moves: reply_moves.append(r)
        rows=[];worst=MATE_SCORE;hard=False
        for i,r in enumerate(reply_moves):
            bb=b.copy(stack=True);bb.push(r)
            infos=analyse(self.verify,bb,cn,1,None,self.tok(f'{token}-cont-{i}'))
            info=infos[0] if infos else {}
            cp=score_cp(info,color);ml=mate_loss(info,color)
            worst=min(worst,cp);hard=hard or ml or cp<=-MATE_SCORE//2
            rows.append({'reply':r.uci(),'cp':cp,'mate_loss':ml})
        if not rows: return -MATE_SCORE,True,[]
        rows.sort(key=lambda x:x['cp'])
        return worst,hard,rows
    def verify_move(self,board,move,color,seed_cp,token,deep=False):
        dcp,dml=self._direct(board,move,color,self.panic_nodes if deep else self.verify_nodes,token)
        hcp,hml,replies=self._hostile(board,move,color,token,deep)
        return VerifiedMove(move,seed_cp,min(dcp,hcp),dcp,dml or hml,replies,immediate_draw(board,move))
    def choose(self,board,token):
        color=board.turn
        sf_roots=root_map(self.floor,board,self.floor_nodes,max(4,self.candidate_mpv),color,self.tok(token+'-sf-floor'))
        if not sf_roots: raise RuntimeError('oracle floor produced no root move')
        floor_move=max(sf_roots,key=lambda m:sf_roots[m]['cp'])
        p09_roots=root_map(self.p09,board,self.candidate_nodes,self.candidate_mpv,color,self.tok(token+'-p09'))
        pool=[]
        for m in [floor_move]+list(p09_roots.keys())+list(sf_roots.keys()):
            if m not in pool:pool.append(m)
        draw_moves=[m for m in board.legal_moves if immediate_draw(board,m)]
        for m in draw_moves:
            if m not in pool:pool.append(m)
        seed=lambda m:max(p09_roots.get(m,{'cp':-MATE_SCORE})['cp'],sf_roots.get(m,{'cp':-MATE_SCORE})['cp'])
        floor_v=self.verify_move(board,floor_move,color,seed(floor_move),token+'-floor',False)
        # If even the oracle's own move looks tactically catastrophic, escalate it first.
        if floor_v.mate_refuted or floor_v.worst_cp <= -900:
            floor_v=self.verify_move(board,floor_move,color,seed(floor_move),token+'-floor-panic',True)
        # A legal claimable draw dominates all speculative defense whenever the oracle floor is not clearly winning.
        if draw_moves and floor_v.worst_cp < 150:
            m=draw_moves[0]
            return m,{'reason':'draw_lock','floor_move':floor_move.uci(),'floor_worst_cp':floor_v.worst_cp,'chosen':m.uci()}
        verified=[]
        for i,m in enumerate(pool):
            v=self.verify_move(board,m,color,seed(m),f'{token}-cand-{i}',False)
            verified.append(v)
        eligible=[v for v in verified if not v.mate_refuted and v.worst_cp >= floor_v.worst_cp-self.epsilon_cp]
        if not eligible:
            # Experimental layer cannot beat its floor: use the floor, unless it is itself deeply refuted.
            if not floor_v.mate_refuted:
                chosen=floor_v
                return chosen.move,self._telemetry('floor_fallback',floor_v,chosen,verified)
            # Panic: every legal move, deep hostile reply enumeration.
            panic=[]
            for i,m in enumerate(board.legal_moves):
                panic.append(self.verify_move(board,m,color,seed(m),f'{token}-panic-{i}',True))
            safe=[v for v in panic if not v.mate_refuted]
            if not safe:
                raise RuntimeError('P20_UNCERTIFIED_ALL_MOVES_REFUTED')
            chosen=max(safe,key=lambda v:(v.worst_cp,v.seed_cp))
            return chosen.move,self._telemetry('panic_escape',floor_v,chosen,panic)
        # Primary objective is survival floor. Only after that maximize seed strength.
        chosen=max(eligible,key=lambda v:(v.worst_cp,v.seed_cp))
        return chosen.move,self._telemetry('oracle_dominance',floor_v,chosen,verified)
    def _telemetry(self,reason,floor_v,chosen,rows):
        def r(v):return {'move':v.move.uci(),'seed_cp':v.seed_cp,'direct_cp':v.direct_cp,'worst_cp':v.worst_cp,'mate_refuted':v.mate_refuted,'immediate_draw':v.immediate_draw,'replies':v.replies[:6]}
        return {'reason':reason,'floor':r(floor_v),'chosen':r(chosen),'candidates':[r(v) for v in sorted(rows,key=lambda x:x.worst_cp,reverse=True)[:10]]}


def weighted_choice(infos,board,rng,temp_cp):
    rows=[]
    for info in infos:
        pv=info.get('pv') or []
        if pv: rows.append((pv[0],float(score_cp(info,board.turn))))
    if not rows:return next(iter(board.legal_moves))
    best=max(v for _,v in rows);ws=[math.exp(max(-20.0,min(0.0,(v-best)/max(1.0,temp_cp)))) for _,v in rows]
    return rng.choices([m for m,_ in rows],weights=ws,k=1)[0]


def generate_openings(sf_path,out,count,plies,seed,nodes):
    if out.exists():
        f=[x.strip() for x in out.read_text(encoding='utf-8').splitlines() if x.strip()]
        if len(f)>=count:return f[:count]
    rng=random.Random(seed);e=chess.engine.SimpleEngine.popen_uci(sf_path,timeout=60.0);configure(e,1,32);fens=[]
    try:
        for i in range(count):
            b=chess.Board()
            for _ in range(plies):
                if b.is_game_over(claim_draw=True):break
                infos=analyse(e,b,nodes,min(4,b.legal_moves.count()),None,f'open-{i}')
                m=weighted_choice(infos,b,rng,55.0)
                if m not in b.legal_moves:break
                b.push(m)
            if b.is_game_over(claim_draw=True):b=chess.Board()
            fens.append(b.fen());jprint({'event':'opening','index':i+1,'fen':b.fen()})
    finally:safe_quit(e)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(fens)+'\n',encoding='utf-8');return fens


def score_from_result(result,lev_white):
    if result=='1/2-1/2':return .5
    return 1.0 if ((result=='1-0')==lev_white) else 0.0


def append_jsonl(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()


def play_game(funnel,sf,fen,lev_white,token,opp_ms,max_plies,decisions):
    b=chess.Board(fen);moves=[];limit=chess.engine.Limit(time=opp_ms/1000.0)
    for ply in range(max_plies):
        if b.is_game_over(claim_draw=True):break
        lev_turn=b.turn==(chess.WHITE if lev_white else chess.BLACK)
        if lev_turn:
            m,t=funnel.choose(b,token);t.update({'game':token,'ply':ply+1,'fen':b.fen()});append_jsonl(decisions,t)
        else:
            r=sf.play(b,limit,game=token,ponder=False);m=r.move
        if m is None or m not in b.legal_moves:raise RuntimeError(f'illegal move {m} at {b.fen()}')
        moves.append(m.uci());b.push(m)
    o=b.outcome(claim_draw=True);res='1/2-1/2' if o is None else o.result();term='MAX_PLIES' if o is None else o.termination.name
    return {'opening_fen':fen,'leviathan_white':lev_white,'result':res,'score_leviathan':score_from_result(res,lev_white),'termination':term,'plies':len(moves),'moves':moves}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--engine',required=True);ap.add_argument('--opponent-engine',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--games',type=int,default=100);ap.add_argument('--opponent-threads',type=int,default=6);ap.add_argument('--hash',type=int,default=128);ap.add_argument('--opponent-movetime-ms',type=int,default=500);ap.add_argument('--max-plies',type=int,default=240)
    ap.add_argument('--opening-plies',type=int,default=10);ap.add_argument('--opening-nodes',type=int,default=1500);ap.add_argument('--seed',type=int,default=20260818);ap.add_argument('--sentinel-repeats',type=int,default=10)
    ap.add_argument('--floor-nodes',type=int,default=1000000);ap.add_argument('--candidate-nodes',type=int,default=150000);ap.add_argument('--verify-nodes',type=int,default=500000);ap.add_argument('--reply-nodes',type=int,default=250000);ap.add_argument('--continuation-nodes',type=int,default=500000);ap.add_argument('--panic-nodes',type=int,default=2000000)
    ap.add_argument('--candidate-mpv',type=int,default=8);ap.add_argument('--reply-mpv',type=int,default=8);ap.add_argument('--epsilon-cp',type=int,default=5)
    a=ap.parse_args()
    ident={'version':'p20-oracle-floor-v1','engine_sha256':sha256_file(Path(a.engine)),'stockfish_sha256':sha256_file(Path(a.opponent_engine)),'games':a.games,'floor_nodes':a.floor_nodes,'verify_nodes':a.verify_nodes,'reply_nodes':a.reply_nodes,'continuation_nodes':a.continuation_nodes,'panic_nodes':a.panic_nodes,'epsilon_cp':a.epsilon_cp,'sentinel_repeats':a.sentinel_repeats,'seed':a.seed}
    run_id=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:12];out=Path(a.out_dir)/run_id;out.mkdir(parents=True,exist_ok=True);(out/'manifest.json').write_text(json.dumps(ident,indent=2,sort_keys=True),encoding='utf-8');jprint({'event':'P20_CONFIG','run_id':run_id,**ident})
    funnel=OracleFloor(a.engine,a.opponent_engine,a.hash,a.floor_nodes,a.candidate_nodes,a.verify_nodes,a.reply_nodes,a.continuation_nodes,a.candidate_mpv,a.reply_mpv,a.epsilon_cp,a.panic_nodes)
    sf=chess.engine.SimpleEngine.popen_uci(a.opponent_engine,timeout=60.0);configure(sf,a.opponent_threads,a.hash);decisions=out/'decisions.jsonl'
    try:
        jprint({'event':'P20_SENTINELS_START','cases':len(LOSS_SENTINELS),'repeats':a.sentinel_repeats});idx=0
        for case in LOSS_SENTINELS:
            for rep in range(1,a.sentinel_repeats+1):
                idx+=1;r=play_game(funnel,sf,case['fen'],bool(case['leviathan_white']),f'sentinel-{idx}',a.opponent_movetime_ms,a.max_plies,decisions);r.update({'sentinel':case['name'],'repeat':rep});append_jsonl(out/'sentinels.jsonl',r);jprint({'event':'P20_SENTINEL_COMPLETE',**r})
                if r['score_leviathan']==0.0:jprint({'event':'P20_SENTINEL_LOSS',**r});return 21
        jprint({'event':'P20_SENTINELS_PASSED','games':idx,'losses':0})
        openings=generate_openings(a.opponent_engine,out/'openings.fen',a.games//2,a.opening_plies,a.seed,a.opening_nodes)
        rows=[]
        for g in range(1,a.games+1):
            fen=openings[(g-1)//2];lev_white=g%2==1;r=play_game(funnel,sf,fen,lev_white,f'fresh-{g}',a.opponent_movetime_ms,a.max_plies,decisions);r['game']=g;append_jsonl(out/'games.jsonl',r);rows.append(r);jprint({'event':'P20_GAME_COMPLETE',**r})
            if r['score_leviathan']==0.0:jprint({'event':'P20_ZERO_LOSS_FAILED','game':g,**r});return 31
        s={'games':len(rows),'wins':sum(x['score_leviathan']==1 for x in rows),'draws':sum(x['score_leviathan']==.5 for x in rows),'losses':sum(x['score_leviathan']==0 for x in rows),'score':sum(x['score_leviathan'] for x in rows)/len(rows)};(out/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8');jprint({'event':'P20_ZERO_LOSS_PASSED','summary':s});return 0
    finally:
        funnel.close();safe_quit(sf)

if __name__=='__main__':raise SystemExit(main())
