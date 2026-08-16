#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random, statistics
from pathlib import Path
import chess, chess.engine

MATE=100000

def info_cp(info, pov):
    v=info['score'].pov(pov).score(mate_score=MATE)
    return int(v if v is not None else 0)

def clear(e):
    if 'Clear Hash' in e.options:
        try:e.configure({'Clear Hash':None})
        except Exception:pass

def analyse(e,b,nodes,multipv=1,root_moves=None):
    r=e.analyse(b,chess.engine.Limit(nodes=nodes),multipv=multipv,root_moves=root_moves)
    return r if isinstance(r,list) else [r]

def generate(sf,count,seed):
    r=random.Random(seed);out=[];seen=set()
    while len(out)<count:
        b=chess.Board();clear(sf)
        for _ in range(r.randint(10,34)):
            if b.is_game_over():break
            I=analyse(sf,b,1000,min(5,b.legal_moves.count()));R=[]
            for z in I:
                if z.get('pv'):R.append((z['pv'][0],info_cp(z,b.turn)))
            if not R:break
            best=R[0][1];V=[x for x in R if best-x[1]<=100] or R[:1]
            b.push(r.choices([m for m,_ in V],weights=list(range(len(V),0,-1)),k=1)[0])
        if b.is_game_over():continue
        k=' '.join(b.fen().split()[:4])
        if k in seen:continue
        seen.add(k);out.append(b.fen())
    return out

def material(board):
    vals={chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9}
    return sum(vals.get(p.piece_type,0) for p in board.piece_map().values())

def features(b):
    return {
      'legal_moves':b.legal_moves.count(),
      'in_check':b.is_check(),
      'pieces':len(b.piece_map()),
      'nonking_material':material(b),
      'pawns':len(b.pieces(chess.PAWN,chess.WHITE))+len(b.pieces(chess.PAWN,chess.BLACK)),
      'queens':len(b.pieces(chess.QUEEN,chess.WHITE))+len(b.pieces(chess.QUEEN,chess.BLACK)),
      'halfmove_clock':b.halfmove_clock,
      'phase':'endgame' if len(b.piece_map())<=12 else ('late' if len(b.piece_map())<=20 else 'middlegame'),
    }

def choice(e,b,nodes):
    clear(e);z=analyse(e,b,nodes,1)[0]
    return z['pv'][0].uci(),info_cp(z,b.turn)

def deep_map(e,b,nodes):
    clear(e);I=analyse(e,b,nodes,min(8,b.legal_moves.count()));m={}
    for z in I:
        if z.get('pv'):m[z['pv'][0].uci()]=info_cp(z,b.turn)
    return m

def forced(e,b,move,nodes):
    mv=chess.Move.from_uci(move);clear(e)
    z=analyse(e,b,nodes,1,[mv])[0]
    return info_cp(z,b.turn)

def summarize(rows,key):
    vals=[r[key] for r in rows]
    return {'mean_regret':statistics.mean(vals),'median_regret':statistics.median(vals),
            'best_agreement':sum(v==0 for v in vals)/len(vals),
            'over20':sum(v>20 for v in vals),'over50':sum(v>50 for v in vals),'over100':sum(v>100 for v in vals)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--leviathan',required=True);ap.add_argument('--stockfish',required=True)
    ap.add_argument('--positions',type=int,default=24);ap.add_argument('--seed',type=int,default=70031)
    ap.add_argument('--shallow',type=int,default=15000);ap.add_argument('--deep',type=int,default=250000);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    lev=chess.engine.SimpleEngine.popen_uci(a.leviathan);sf=chess.engine.SimpleEngine.popen_uci(a.stockfish);oracle=chess.engine.SimpleEngine.popen_uci(a.stockfish)
    for e in (lev,sf,oracle):e.configure({'Threads':1,'Hash':64})
    try:
        fens=generate(oracle,a.positions,a.seed);rows=[]
        for n,fen in enumerate(fens,1):
            b=chess.Board(fen);lm,_=choice(lev,b,a.shallow);sm,_=choice(sf,b,a.shallow);dm=deep_map(oracle,b,a.deep);best=max(dm.values())
            for m in {lm,sm}:
                if m not in dm:dm[m]=forced(oracle,b,m,max(50000,a.deep//2))
            row={'index':n,'fen':fen,'features':features(b),'lev_move':lm,'sf_move':sm,'oracle_best':[m for m,v in dm.items() if v==best],
                 'oracle_best_cp':best,'lev_cp':dm[lm],'sf_cp':dm[sm],'lev_regret':best-dm[lm],'sf_regret':best-dm[sm]}
            rows.append(row);print(n,lm,sm,row['lev_regret'],row['sf_regret'],row['features'])
        out={'seed':a.seed,'positions':len(rows),'leviathan':summarize(rows,'lev_regret'),'stockfish':summarize(rows,'sf_regret'),'rows':rows}
        a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
    finally:lev.quit();sf.quit();oracle.quit()
if __name__=='__main__':main()
