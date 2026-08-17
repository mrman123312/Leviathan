#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import chess,chess.engine
MATE=100000
BASE={
 'Threads':1,'Hash':64,'Leviathan Fundamentals':True,'Leviathan Fundamentals Authority':1,
 'Leviathan Forcing Buyback':384,'Leviathan Recapture Buyback':256,'Leviathan Passer Buyback':320,
 'Leviathan Endgame Buyback':128,'Leviathan Quiet Overdrive':0,
}
PLAIN={'Threads':1,'Hash':64}

def start(path,opts):
 e=chess.engine.SimpleEngine.popen_uci(path);e.configure({k:v for k,v in opts.items() if k in e.options});return e

def analyse(e,b,nodes,game,root=None):
 z=e.analyse(b,chess.engine.Limit(nodes=nodes),game=game,root_moves=root)
 pv=z.get('pv',[])
 if not pv: raise RuntimeError('no pv')
 return pv[0],int(z['score'].pov(b.turn).score(mate_score=MATE) or 0),int(z.get('nodes',nodes))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--engine',required=True);ap.add_argument('--stockfish',required=True);ap.add_argument('--openings-json',type=Path,required=True);ap.add_argument('--profile-json',required=True);ap.add_argument('--choice-nodes',type=int,default=80000);ap.add_argument('--oracle-nodes',type=int,default=400000);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 profile=json.loads(a.profile_json);cand_opts=BASE|profile
 B=start(a.engine,BASE);C=start(a.engine,cand_opts);O=start(a.stockfish,PLAIN);OB=start(a.stockfish,PLAIN);OC=start(a.stockfish,PLAIN)
 rows=[]
 try:
  for i,x in enumerate(json.loads(a.openings_json.read_text()),1):
   b=chess.Board(x['final_fen']);tok=('param',i,x['final_fen'])
   bm,_,bn=analyse(B,b,a.choice_nodes,('b',tok));cm,_,cn=analyse(C,b,a.choice_nodes,('c',tok))
   om,os,_=analyse(O,b,a.oracle_nodes,('o',tok));_,bs,_=analyse(OB,b,a.oracle_nodes,('ob',tok),[bm]);_,cs,_=analyse(OC,b,a.oracle_nodes,('oc',tok),[cm])
   br=max(0,os-bs);cr=max(0,os-cs)
   rows.append({'i':i,'fen':b.fen(),'baseline_move':bm.uci(),'candidate_move':cm.uci(),'oracle_move':om.uci(),'baseline_regret':br,'candidate_regret':cr,'baseline_nodes':bn,'candidate_nodes':cn})
   print(f'{i:02d} b={bm} r={br} c={cm} r={cr} o={om}',flush=True)
 finally:
  for e in [B,C,O,OB,OC]:e.quit()
 br=[x['baseline_regret'] for x in rows];cr=[x['candidate_regret'] for x in rows]
 out={'profile':profile,'positions':len(rows),'choice_nodes':a.choice_nodes,'oracle_nodes':a.oracle_nodes,
  'baseline':{'mean_regret':statistics.fmean(br),'median_regret':statistics.median(br),'oracle_agreement':sum(x['baseline_move']==x['oracle_move'] for x in rows)/len(rows)},
  'candidate':{'mean_regret':statistics.fmean(cr),'median_regret':statistics.median(cr),'oracle_agreement':sum(x['candidate_move']==x['oracle_move'] for x in rows)/len(rows)},
  'delta_regret':statistics.fmean(cr)-statistics.fmean(br),'better':sum(c<b for b,c in zip(br,cr)),'worse':sum(c>b for b,c in zip(br,cr)),'same':sum(c==b for b,c in zip(br,cr)),'rows':rows}
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
