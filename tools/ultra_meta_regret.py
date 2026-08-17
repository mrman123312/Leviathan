#!/usr/bin/env python3
"""Deterministic equal-budget root-allocation lab for Fundamentals Ultra."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import chess,chess.engine

MATE=100000
FUNDAMENTALS={"Threads":1,"Hash":64,"Leviathan Fundamentals":True,"Leviathan Fundamentals Authority":1,"Leviathan Quiet Overdrive":0}
PLAIN={"Threads":1,"Hash":64}

def cp(info,color): return int(info["score"].pov(color).score(mate_score=MATE) or 0)

def best(engine,board,nodes,multipv=1,root_moves=None,game=None):
    r=engine.analyse(board,chess.engine.Limit(nodes=nodes),multipv=multipv,root_moves=root_moves,game=game)
    infos=r if isinstance(r,list) else [r]; rows=[]
    for z in infos:
        pv=z.get("pv",[])
        if pv: rows.append((pv[0],cp(z,board.turn),int(z.get("nodes",nodes))))
    if not rows: raise RuntimeError("no PV")
    return rows

def choose_monolith(k,b,sf,board,total,game):
    r=best(k,board,total,game=game)[0]
    return r[0],{"actual_nodes":r[2],"views":["kernel"]}

def choose_portfolio(k,b,sf,board,total,width,game):
    first=44000 if width==2 else 35000
    initial=best(k,board,first,multipv=width,game=game)
    candidates=[]
    for mv,_,_ in initial:
        if mv not in candidates:candidates.append(mv)
    used=max(x[2] for x in initial); remain=max(0,total-used); each=max(1000,remain//max(1,len(candidates)))
    verified=[]
    for mv in candidates:
        r=best(k,board,each,root_moves=[mv],game=game)[0]; used+=r[2];verified.append((mv,r[1]))
    verified.sort(key=lambda x:x[1],reverse=True)
    return verified[0][0],{"actual_nodes":used,"views":[f"kernel-multipv{width}","kernel-restricted-verify"],"candidates":[m.uci() for m in candidates],"verified":[(m.uci(),s) for m,s in verified]}

def choose_duel(k,other,board,total,other_name,game):
    ra=best(k,board,30000,game=game)[0];rb=best(other,board,30000,game=game)[0]
    used=ra[2]+rb[2];a,b=ra[0],rb[0]
    if a==b:return a,{"actual_nodes":used,"views":["kernel",other_name],"agreed":True}
    each=max(1000,(total-used)//2);verified=[]
    for mv in [a,b]:
        r=best(k,board,each,root_moves=[mv],game=game)[0];used+=r[2];verified.append((mv,r[1]))
    verified.sort(key=lambda x:x[1],reverse=True)
    return verified[0][0],{"actual_nodes":used,"views":["kernel",other_name,"kernel-restricted-verify"],"agreed":False,"candidates":[a.uci(),b.uci()],"verified":[(m.uci(),s) for m,s in verified]}

def choose_committee3(k,b,sf,board,total,game):
    props=[];used=0
    for name,e in [("kernel",k),("base",b),("stockfish",sf)]:
        r=best(e,board,20000,game=game)[0];used+=r[2];props.append((name,r[0]))
    candidates=[]
    for _,mv in props:
        if mv not in candidates:candidates.append(mv)
    remain=max(0,total-used);each=max(1000,remain//max(1,len(candidates)));verified=[]
    for mv in candidates:
        r=best(k,board,each,root_moves=[mv],game=game)[0];used+=r[2];verified.append((mv,r[1]))
    verified.sort(key=lambda x:x[1],reverse=True)
    return verified[0][0],{"actual_nodes":used,"views":["kernel","base","stockfish","kernel-restricted-verify"],"proposals":[(n,m.uci()) for n,m in props],"candidates":[m.uci() for m in candidates],"verified":[(m.uci(),s) for m,s in verified]}

def choose(strategy,k,b,sf,board,total,game):
    if strategy=="monolith":return choose_monolith(k,b,sf,board,total,game)
    if strategy=="portfolio2":return choose_portfolio(k,b,sf,board,total,2,game)
    if strategy=="portfolio3":return choose_portfolio(k,b,sf,board,total,3,game)
    if strategy=="duel-base":return choose_duel(k,b,board,total,"base",game)
    if strategy=="duel-stockfish":return choose_duel(k,sf,board,total,"stockfish",game)
    if strategy=="committee3":return choose_committee3(k,b,sf,board,total,game)
    raise ValueError(strategy)

def start(path,opts):
    e=chess.engine.SimpleEngine.popen_uci(path); e.configure(opts); return e

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--kernel",required=True);ap.add_argument("--base",required=True);ap.add_argument("--stockfish",required=True);ap.add_argument("--openings-json",type=Path,required=True);ap.add_argument("--strategy",required=True,choices=["monolith","portfolio2","portfolio3","duel-base","duel-stockfish","committee3"]);ap.add_argument("--total-nodes",type=int,default=80000);ap.add_argument("--oracle-nodes",type=int,default=400000);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    # Separate processes prevent baseline/candidate/oracle TT contamination.
    kb=start(a.kernel,FUNDAMENTALS);kc=start(a.kernel,FUNDAMENTALS);base=start(a.base,FUNDAMENTALS);proposer=start(a.stockfish,PLAIN)
    ob=start(a.stockfish,PLAIN);obb=start(a.stockfish,PLAIN);obc=start(a.stockfish,PLAIN)
    rows=[]
    try:
        openings=json.loads(a.openings_json.read_text());fens=[x.get("final_fen") or x["fen"] for x in openings]
        for i,fen in enumerate(fens,1):
            board=chess.Board(fen); token=("position",i,fen)
            bm,bmeta=choose_monolith(kb,base,proposer,board,a.total_nodes,("baseline",token))
            cm,cmeta=choose(a.strategy,kc,base,proposer,board,a.total_nodes,("candidate",token))
            # Three independent oracle processes each receive one search per position.
            o=best(ob,board,a.oracle_nodes,game=("oracle-best",token))[0]
            bs=best(obb,board,a.oracle_nodes,root_moves=[bm],game=("oracle-baseline",token))[0]
            cs=best(obc,board,a.oracle_nodes,root_moves=[cm],game=("oracle-candidate",token))[0]
            om,os=o[0],o[1];breg=max(0,os-bs[1]);creg=max(0,os-cs[1])
            rows.append({"index":i,"fen":fen,"oracle_move":om.uci(),"oracle_score":os,"baseline_move":bm.uci(),"baseline_score":bs[1],"baseline_regret_cp":breg,"baseline_meta":bmeta,"candidate_move":cm.uci(),"candidate_score":cs[1],"candidate_regret_cp":creg,"candidate_meta":cmeta})
            print(f"{i:02d}/{len(fens)} base={bm.uci()} r={breg} cand={cm.uci()} r={creg} oracle={om.uci()} nodes={cmeta['actual_nodes']}",flush=True)
    finally:
        for e in [kb,kc,base,proposer,ob,obb,obc]:e.quit()
    br=[r["baseline_regret_cp"] for r in rows];cr=[r["candidate_regret_cp"] for r in rows];actual=[r["candidate_meta"]["actual_nodes"] for r in rows]
    out={"strategy":a.strategy,"positions":len(rows),"node_budget":a.total_nodes,"oracle_nodes":a.oracle_nodes,"candidate_actual_nodes":{"mean":statistics.fmean(actual),"max":max(actual)},"budget_overrun_positions":sum(x>a.total_nodes*1.02 for x in actual),"baseline":{"mean_regret":statistics.fmean(br),"median_regret":statistics.median(br),"max_regret":max(br),"zero_count":sum(x==0 for x in br),"oracle_move_agreement":sum(r["baseline_move"]==r["oracle_move"] for r in rows)/len(rows)},"candidate":{"mean_regret":statistics.fmean(cr),"median_regret":statistics.median(cr),"max_regret":max(cr),"zero_count":sum(x==0 for x in cr),"oracle_move_agreement":sum(r["candidate_move"]==r["oracle_move"] for r in rows)/len(rows)},"mean_regret_delta_candidate_minus_baseline":statistics.fmean(cr)-statistics.fmean(br),"candidate_better_positions":sum(c<b for b,c in zip(br,cr)),"candidate_worse_positions":sum(c>b for b,c in zip(br,cr)),"same_positions":sum(c==b for b,c in zip(br,cr)),"rows":rows}
    a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))
if __name__=="__main__":main()
