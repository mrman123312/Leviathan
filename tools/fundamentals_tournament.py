#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, statistics, time
from pathlib import Path
import chess, chess.engine, chess.pgn

DEFAULT_OPENINGS=[
("Open Game",["e2e4","e7e5","g1f3","b8c6"]),("Queen's Gambit",["d2d4","d7d5","c2c4","e7e6"]),
("English",["c2c4","e7e5","b1c3","g8f6"]),("Reti",["g1f3","d7d5","g2g3","c7c5"]),
("Sicilian",["e2e4","c7c5","g1f3","d7d6"]),("King's Indian",["d2d4","g8f6","c2c4","g7g6"]),
("French",["e2e4","e7e6","d2d4","d7d5"]),("Caro-Kann",["e2e4","c7c6","d2d4","d7d5"])]
INFO=chess.engine.INFO_BASIC|chess.engine.INFO_SCORE|chess.engine.INFO_PV

def load_openings(path):
    if not path:return DEFAULT_OPENINGS
    raw=json.loads(Path(path).read_text())
    out=[]
    for i,x in enumerate(raw):
        moves=x["moves"] if isinstance(x,dict) else x
        name=x.get("name",f"generated-{i+1}") if isinstance(x,dict) else f"generated-{i+1}"
        b=chess.Board()
        for u in moves:
            m=chess.Move.from_uci(u)
            if m not in b.legal_moves:raise RuntimeError(f"illegal seed {u} in {name}")
            b.push(m)
        out.append((name,moves))
    return out

def configure(e,fundamentals,threads=1,hash_mb=64):
    opts={}
    for n,v in [("Threads",threads),("Hash",hash_mb)]:
        if n in e.options:opts[n]=v
    if fundamentals:
        for n,v in [("Leviathan Fundamentals",True),("Leviathan Fundamentals Authority",1),("Leviathan Quiet Overdrive",0)]:
            if n in e.options:opts[n]=v
    if opts:e.configure(opts)
    return opts

def mean(xs):return statistics.fmean(xs) if xs else None
def rscore(r,c):return .5 if r=="1/2-1/2" else float((r=="1-0")==(c==chess.WHITE))
def elo(s):return None if s<=0 or s>=1 else 400*math.log10(s/(1-s))
def limit_for(move_ms,nodes):
    if nodes:return chess.engine.Limit(nodes=nodes)
    return chess.engine.Limit(time=move_ms/1000.0)

def play_one(cand,opp,name,moves,color,move_ms,nodes,max_plies,idx):
    b=chess.Board()
    for u in moves:b.push_uci(u)
    g=chess.pgn.Game.from_board(b); g.headers.update({"Event":"Fundamentals Ultra confirmation","Round":str(idx),"Opening":name,"CandidateColor":"White" if color else "Black","Limit":f"nodes={nodes}" if nodes else f"movetime={move_ms}ms"})
    node=g.end(); token=object(); ct=[];ot=[];cd=[];od=[];cn=[];on=[];err=None;term=None
    for _ in range(max_plies):
        outcome=b.outcome(claim_draw=True)
        if outcome:term=outcome.termination.name;break
        is_c=b.turn==color;e=cand if is_c else opp;t0=time.perf_counter()
        try:r=e.play(b,limit_for(move_ms,nodes),game=token,info=INFO)
        except Exception as exc:err=f"{type(exc).__name__}: {exc}";term="ENGINE_ERROR";break
        dt=(time.perf_counter()-t0)*1000; info=r.info or {}; ts,ds,ns=(ct,cd,cn) if is_c else (ot,od,on);ts.append(dt)
        if isinstance(info.get("depth"),int):ds.append(float(info["depth"]))
        if isinstance(info.get("nodes"),int):ns.append(float(info["nodes"]))
        if r.move is None or r.move not in b.legal_moves:err=f"invalid/no move: {r.move}";term="INVALID_MOVE";break
        b.push(r.move);node=node.add_variation(r.move)
    else:term="PLY_LIMIT_DRAW"
    outcome=b.outcome(claim_draw=True)
    if err:
        failed=b.turn==color;r="0-1" if (failed and color==chess.WHITE) or (not failed and color==chess.BLACK) else "1-0"
    else:r=outcome.result() if outcome else "1/2-1/2"
    g.headers["Result"]=r;g.headers["Termination"]=term or "unknown"
    return g,{"game":idx,"opening":name,"candidate_color":"white" if color else "black","result":r,"candidate_score":rscore(r,color),"termination":term,"error":err,"plies_after_seed":len(b.move_stack)-len(moves),"candidate_mean_ms":mean(ct),"opponent_mean_ms":mean(ot),"candidate_mean_depth":mean(cd),"opponent_mean_depth":mean(od),"candidate_mean_nodes":mean(cn),"opponent_mean_nodes":mean(on),"final_fen":b.fen()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--opponent",required=True);ap.add_argument("--candidate-label",required=True);ap.add_argument("--opponent-label",default="Stockfish");ap.add_argument("--candidate-fundamentals",action="store_true");ap.add_argument("--opponent-fundamentals",action="store_true");ap.add_argument("--candidate-threads",type=int,default=1);ap.add_argument("--opponent-threads",type=int,default=1);ap.add_argument("--candidate-hash",type=int,default=64);ap.add_argument("--opponent-hash",type=int,default=64);ap.add_argument("--move-time-ms",type=int,default=50);ap.add_argument("--nodes-per-move",type=int,default=0);ap.add_argument("--max-plies",type=int,default=180);ap.add_argument("--openings-json");ap.add_argument("--output-dir",required=True);a=ap.parse_args()
    if a.candidate_threads<1 or a.opponent_threads<1:raise SystemExit("thread counts must be >= 1")
    if a.candidate_hash<1 or a.opponent_hash<1:raise SystemExit("hash sizes must be >= 1 MB")
    openings=load_openings(a.openings_json);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);cand=chess.engine.SimpleEngine.popen_uci(a.candidate,timeout=30);opp=chess.engine.SimpleEngine.popen_uci(a.opponent,timeout=30);recs=[];games=[]
    try:
        co=configure(cand,a.candidate_fundamentals,a.candidate_threads,a.candidate_hash);oo=configure(opp,a.opponent_fundamentals,a.opponent_threads,a.opponent_hash);idx=0
        for name,moves in openings:
            for color in (chess.WHITE,chess.BLACK):
                idx+=1;g,r=play_one(cand,opp,name,moves,color,a.move_time_ms,a.nodes_per_move,a.max_plies,idx);g.headers["White"]=a.candidate_label if color else a.opponent_label;g.headers["Black"]=a.candidate_label if not color else a.opponent_label;games.append(g);recs.append(r);print(json.dumps(r,sort_keys=True),flush=True)
    finally:cand.quit();opp.quit()
    with (out/"games.pgn").open("w") as f:
        for g in games:print(g,file=f,end="\n\n")
    (out/"games.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in recs))
    w=sum(r["candidate_score"]==1 for r in recs);d=sum(r["candidate_score"]==.5 for r in recs);l=len(recs)-w-d;s=(w+.5*d)/len(recs);errs=[r for r in recs if r["error"]]
    sm={"candidate":a.candidate_label,"opponent":a.opponent_label,"games":len(recs),"opening_pairs":len(openings),"wins":w,"draws":d,"losses":l,"score_fraction":s,"naive_logistic_elo":elo(s),"candidate_options":co,"opponent_options":oo,"move_time_ms":a.move_time_ms if not a.nodes_per_move else None,"nodes_per_move":a.nodes_per_move or None,"errors":errs,"candidate_mean_ms":mean([r["candidate_mean_ms"] for r in recs if r["candidate_mean_ms"] is not None]),"opponent_mean_ms":mean([r["opponent_mean_ms"] for r in recs if r["opponent_mean_ms"] is not None]),"candidate_mean_depth":mean([r["candidate_mean_depth"] for r in recs if r["candidate_mean_depth"] is not None]),"opponent_mean_depth":mean([r["opponent_mean_depth"] for r in recs if r["opponent_mean_depth"] is not None]),"candidate_mean_nodes":mean([r["candidate_mean_nodes"] for r in recs if r["candidate_mean_nodes"] is not None]),"opponent_mean_nodes":mean([r["opponent_mean_nodes"] for r in recs if r["opponent_mean_nodes"] is not None])}
    (out/"summary.json").write_text(json.dumps(sm,indent=2,sort_keys=True)+"\n");print("SUMMARY "+json.dumps(sm,sort_keys=True));return 0 if not errs else 2
if __name__=="__main__":raise SystemExit(main())
