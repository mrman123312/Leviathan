#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random,statistics
from pathlib import Path
import chess,chess.engine
MATE=100000

def clear(e):
    if 'Clear Hash' in e.options:
        try:e.configure({'Clear Hash':None})
        except:pass

def cp(info,pov):
    v=info['score'].pov(pov).score(mate_score=MATE);return int(v if v is not None else 0)

def ana(e,b,nodes,multipv=1,root_moves=None):
    x=e.analyse(b,chess.engine.Limit(nodes=nodes),multipv=multipv,root_moves=root_moves)
    return x if isinstance(x,list) else [x]

def move(e,b,nodes):
    clear(e);z=ana(e,b,nodes)[0];return z['pv'][0].uci()

def forced(e,b,m,nodes):
    clear(e);mv=chess.Move.from_uci(m);z=ana(e,b,nodes,1,[mv])[0];return cp(z,b.turn)

def oracle(e,b,nodes,moves):
    vals={m:forced(e,b,m,nodes) for m in moves};best=max(vals.values());return vals,best

def gen_step(sf,b,r):
    I=ana(sf,b,1000,min(5,b.legal_moves.count()));R=[]
    for z in I:
        if z.get('pv'):R.append((z['pv'][0],cp(z,b.turn)))
    if not R:return None
    best=R[0][1];V=[x for x in R if best-x[1]<=110] or R[:1]
    return r.choices([m for m,_ in V],weights=list(range(len(V),0,-1)),k=1)[0]

def feats(b):
    return {'legal_moves':b.legal_moves.count(),'in_check':b.is_check(),'pieces':len(b.piece_map()),'pawns':len(b.pieces(chess.PAWN,0))+len(b.pieces(chess.PAWN,1)),'queens':len(b.pieces(chess.QUEEN,0))+len(b.pieces(chess.QUEEN,1)),'halfmove':b.halfmove_clock}

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--parent',required=True);p.add_argument('--oracle',required=True);p.add_argument('--disagreements',type=int,default=30);p.add_argument('--max-positions',type=int,default=1200);p.add_argument('--seed',type=int,default=550031);p.add_argument('--shallow',type=int,default=18000);p.add_argument('--deep',type=int,default=300000);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    C=chess.engine.SimpleEngine.popen_uci(a.candidate);P=chess.engine.SimpleEngine.popen_uci(a.parent);O=chess.engine.SimpleEngine.popen_uci(a.oracle);G=chess.engine.SimpleEngine.popen_uci(a.oracle)
    for e in (C,P,O,G):e.configure({'Threads':1,'Hash':64})
    r=random.Random(a.seed);rows=[];tested=0;seen=set()
    try:
      while len(rows)<a.disagreements and tested<a.max_positions:
        b=chess.Board()
        for _ in range(r.randint(8,34)):
          if b.is_game_over():break
          m=gen_step(G,b,r)
          if m is None:break
          b.push(m)
        if b.is_game_over():continue
        k=' '.join(b.fen().split()[:4])
        if k in seen:continue
        seen.add(k);tested+=1
        cm=move(C,b,a.shallow);pm=move(P,b,a.shallow)
        if cm==pm:continue
        vals,best=oracle(O,b,a.deep,{cm,pm});cr=best-vals[cm];pr=best-vals[pm]
        row={'fen':b.fen(),'candidate_move':cm,'parent_move':pm,'candidate_cp':vals[cm],'parent_cp':vals[pm],'candidate_regret':cr,'parent_regret':pr,'delta_regret':pr-cr,'winner':'candidate' if cr<pr else ('parent' if pr<cr else 'tie'),'features':feats(b)}
        rows.append(row);print(len(rows),row['winner'],row['delta_regret'],cm,pm,row['features'])
      wins=sum(x['winner']=='candidate' for x in rows);loss=sum(x['winner']=='parent' for x in rows);ties=len(rows)-wins-loss;d=[x['delta_regret'] for x in rows]
      out={'tested_positions':tested,'disagreements':len(rows),'candidate_wins':wins,'parent_wins':loss,'ties':ties,'candidate_nonloss_rate':(wins+ties)/len(rows) if rows else None,'candidate_win_rate_ex_ties':wins/(wins+loss) if wins+loss else None,'mean_regret_advantage_cp':statistics.mean(d) if d else None,'median_regret_advantage_cp':statistics.median(d) if d else None,'rows':rows}
      a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
    finally:
      for e in (C,P,O,G):e.quit()
if __name__=='__main__':main()
