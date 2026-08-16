#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import chess
import chess.engine

from specialists.trajectory_lattice import recommend, cp_for

MATE_CP = 100000


def clear_hash(e: chess.engine.SimpleEngine):
    if "Clear Hash" in e.options:
        try:
            e.configure({"Clear Hash": None})
        except Exception:
            pass


def analyse_multi(e, b, nodes, multipv):
    r=e.analyse(b,chess.engine.Limit(nodes=nodes),multipv=min(multipv,b.legal_moves.count()))
    return r if isinstance(r,list) else [r]


def make_positions(engine, count, seed):
    rng=random.Random(seed); out=[]; seen=set()
    while len(out)<count:
        b=chess.Board(); clear_hash(engine)
        target=rng.randint(12,28)
        for _ in range(target):
            if b.is_game_over(): break
            infos=analyse_multi(engine,b,900,4)
            ranked=[]
            for z in infos:
                if z.get('pv'):
                    ranked.append((z['pv'][0],cp_for(z,b.turn)))
            if not ranked: break
            best=ranked[0][1]
            viable=[x for x in ranked if best-x[1]<=120] or ranked[:1]
            mv=rng.choices([x[0] for x in viable],weights=list(range(len(viable),0,-1)),k=1)[0]
            b.push(mv)
        if b.is_game_over(): continue
        key=' '.join(b.fen().split()[:4])
        if key in seen: continue
        seen.add(key); out.append(b.fen())
    return out


def oracle_scores(engine, board, nodes, multipv=8):
    clear_hash(engine)
    infos=analyse_multi(engine,board,nodes,multipv)
    root=board.turn; scores={}; pvs={}
    for z in infos:
        if not z.get('pv'): continue
        m=z['pv'][0].uci(); scores[m]=cp_for(z,root); pvs[m]=[x.uci() for x in z['pv']]
    return scores,pvs


def forced_score(engine, board, move_uci, nodes):
    mv=chess.Move.from_uci(move_uci)
    if mv not in board.legal_moves: return -MATE_CP
    clear_hash(engine)
    z=engine.analyse(board,chess.engine.Limit(nodes=nodes),root_moves=[mv])
    if isinstance(z,list): z=z[0]
    return cp_for(z,board.turn)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--engine',required=True)
    ap.add_argument('--oracle',required=True)
    ap.add_argument('--positions',type=int,default=12)
    ap.add_argument('--seed',type=int,default=8910)
    ap.add_argument('--budget-nodes',type=int,default=30000)
    ap.add_argument('--oracle-nodes',type=int,default=200000)
    ap.add_argument('--horizons',default='16,64,246')
    ap.add_argument('--out',type=Path,default=Path('trajectory-oracle.json'))
    args=ap.parse_args()
    horizons=[int(x) for x in args.horizons.split(',')]

    gen=chess.engine.SimpleEngine.popen_uci(args.oracle)
    base=chess.engine.SimpleEngine.popen_uci(args.engine)
    oracle=chess.engine.SimpleEngine.popen_uci(args.oracle)
    try:
        for e in (gen,base,oracle): e.configure({'Threads':1,'Hash':64})
        fens=make_positions(gen,args.positions,args.seed)
        rows=[]
        for idx,fen in enumerate(fens,1):
            b=chess.Board(fen)
            oscores,opvs=oracle_scores(oracle,b,args.oracle_nodes,8)
            if not oscores: continue
            best=max(oscores.values())
            best_moves=[m for m,s in oscores.items() if s==best]

            clear_hash(base)
            bi=base.analyse(b,chess.engine.Limit(nodes=args.budget_nodes))
            if isinstance(bi,list): bi=bi[0]
            baseline=bi['pv'][0].uci()
            if baseline not in oscores:
                oscores[baseline]=forced_score(oracle,b,baseline,args.oracle_nodes//2)
            methods={'baseline':baseline}
            details={}

            for h in horizons:
                clear_hash(base)
                rec=recommend(base,b,args.budget_nodes,3,h,2,2,12,'stable')
                methods[f'atl_{h}']=rec['move']
                details[f'atl_{h}']=rec
                if rec['move'] and rec['move'] not in oscores:
                    oscores[rec['move']]=forced_score(oracle,b,rec['move'],args.oracle_nodes//2)

            regret={k:(best-oscores.get(m,-MATE_CP) if m else 2*MATE_CP) for k,m in methods.items()}
            rows.append({'index':idx,'fen':fen,'oracle_best':best_moves,'oracle_best_cp':best,
                         'methods':methods,'regret_cp':regret,'oracle_scores':oscores,
                         'atl_details':details})
            print(idx, methods, regret)

        names=['baseline']+[f'atl_{h}' for h in horizons]
        summary={}
        for name in names:
            vals=[r['regret_cp'][name] for r in rows]
            summary[name]={
                'positions':len(vals),
                'mean_regret_cp':statistics.mean(vals) if vals else None,
                'median_regret_cp':statistics.median(vals) if vals else None,
                'max_regret_cp':max(vals) if vals else None,
                'oracle_best_agreement':sum(v==0 for v in vals)/len(vals) if vals else None,
                'regret_over_50':sum(v>50 for v in vals),
                'regret_over_100':sum(v>100 for v in vals),
            }
        result={'budget_nodes':args.budget_nodes,'oracle_nodes':args.oracle_nodes,
                'horizons':horizons,'summary':summary,'rows':rows}
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(summary,indent=2))
    finally:
        gen.quit();base.quit();oracle.quit()

if __name__=='__main__': main()
