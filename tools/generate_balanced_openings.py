#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import chess,chess.engine

def cp_white(info,board):
    s=info["score"].pov(chess.WHITE).score(mate_score=100000)
    return 0 if s is None else int(s)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stockfish",required=True);ap.add_argument("--output",required=True);ap.add_argument("--count",type=int,default=50);ap.add_argument("--plies",type=int,default=8);ap.add_argument("--seed",type=int,default=8910);ap.add_argument("--max-abs-cp",type=int,default=80);a=ap.parse_args()
    rng=random.Random(a.seed);e=chess.engine.SimpleEngine.popen_uci(a.stockfish,timeout=30)
    if "Threads" in e.options:e.configure({"Threads":1})
    if "Hash" in e.options:e.configure({"Hash":64})
    out=[];seen=set();attempt=0
    try:
        while len(out)<a.count and attempt<a.count*50:
            attempt+=1;b=chess.Board();moves=[]
            for ply in range(a.plies):
                infos=e.analyse(b,chess.engine.Limit(depth=6),multipv=4,game=(a.seed,attempt))
                infos=infos if isinstance(infos,list) else [infos]
                candidates=[x for x in infos if x.get("pv")]
                if not candidates:break
                weights=[55,25,13,7][:len(candidates)];pick=rng.choices(range(len(candidates)),weights=weights,k=1)[0]
                m=candidates[pick]["pv"][0]
                if m not in b.legal_moves:break
                b.push(m);moves.append(m.uci())
            if len(moves)!=a.plies:continue
            key=" ".join(moves)
            if key in seen:continue
            final=e.analyse(b,chess.engine.Limit(depth=10),game=("final",a.seed,attempt))
            cp=cp_white(final,b)
            if abs(cp)>a.max_abs_cp:continue
            seen.add(key);out.append({"name":f"sf-balanced-{len(out)+1:02d}","moves":moves,"final_fen":b.fen(),"stockfish_depth10_cp_white":cp})
            print(f"accepted {len(out)}/{a.count} cp={cp} {' '.join(moves)}",flush=True)
    finally:e.quit()
    if len(out)!=a.count:raise SystemExit(f"only generated {len(out)} balanced openings after {attempt} attempts")
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    return 0
if __name__=="__main__":raise SystemExit(main())
