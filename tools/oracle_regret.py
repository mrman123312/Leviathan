#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import chess,chess.engine

MATE_SCORE=100000

def cfg(e,fund):
    o={}
    for n,v in [("Threads",1),("Hash",64)]:
        if n in e.options:o[n]=v
    if fund:
        for n,v in [("Leviathan Fundamentals",True),("Leviathan Fundamentals Authority",1),("Leviathan Quiet Overdrive",0)]:
            if n in e.options:o[n]=v
    if o:e.configure(o)

def root_score(info,root_color):
    return int(info["score"].pov(root_color).score(mate_score=MATE_SCORE) or 0)

def analyse_root(e,b,nodes,token,root_moves=None):
    info=e.analyse(b,chess.engine.Limit(nodes=nodes),game=token,root_moves=root_moves)
    pv=info.get("pv",[])
    if not pv: raise RuntimeError("oracle returned no PV")
    return pv[0],root_score(info,b.turn),int(info.get("nodes",nodes))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--baseline",required=True);ap.add_argument("--oracle",required=True);ap.add_argument("--openings-json",required=True);ap.add_argument("--candidate-label",required=True);ap.add_argument("--candidate-fundamentals",action="store_true");ap.add_argument("--baseline-fundamentals",action="store_true");ap.add_argument("--choice-nodes",type=int,default=80000);ap.add_argument("--oracle-nodes",type=int,default=500000);ap.add_argument("--output",required=True);a=ap.parse_args()
    positions=[]
    for x in json.loads(Path(a.openings_json).read_text()):
        b=chess.Board()
        for u in x["moves"]:b.push_uci(u)
        positions.append((x["name"],b))

    cand=chess.engine.SimpleEngine.popen_uci(a.candidate,timeout=30)
    base=chess.engine.SimpleEngine.popen_uci(a.baseline,timeout=30)
    # Isolation is deliberate: selector and all three graders have independent TT/history state.
    oracle_select=chess.engine.SimpleEngine.popen_uci(a.oracle,timeout=30)
    oracle_best=chess.engine.SimpleEngine.popen_uci(a.oracle,timeout=30)
    oracle_cand=chess.engine.SimpleEngine.popen_uci(a.oracle,timeout=30)
    oracle_base=chess.engine.SimpleEngine.popen_uci(a.oracle,timeout=30)
    cfg(cand,a.candidate_fundamentals);cfg(base,a.baseline_fundamentals)
    for e in (oracle_select,oracle_best,oracle_cand,oracle_base):cfg(e,False)

    rows=[]
    try:
        for i,(name,b) in enumerate(positions,1):
            token=("pos",i,b.fen())
            cm=cand.play(b,chess.engine.Limit(nodes=a.choice_nodes),game=("cand",token)).move
            bm=base.play(b,chess.engine.Limit(nodes=a.choice_nodes),game=("base",token)).move
            om,_,select_nodes=analyse_root(oracle_select,b,a.oracle_nodes,("select",token))
            _,best,best_nodes=analyse_root(oracle_best,b,a.oracle_nodes,("grade-best",token),[om])
            _,cs,cand_nodes=analyse_root(oracle_cand,b,a.oracle_nodes,("grade-cand",token),[cm])
            _,bs,base_nodes=analyse_root(oracle_base,b,a.oracle_nodes,("grade-base",token),[bm])
            cr=max(0,best-cs);br=max(0,best-bs)
            r={"position":i,"name":name,"fen":b.fen(),"oracle_move":om.uci(),"oracle_score":best,"candidate_move":cm.uci(),"candidate_score":cs,"candidate_regret_cp":cr,"baseline_move":bm.uci(),"baseline_score":bs,"baseline_regret_cp":br,"candidate_matches_oracle":cm==om,"baseline_matches_oracle":bm==om,"oracle_select_nodes":select_nodes,"oracle_best_grade_nodes":best_nodes,"oracle_candidate_grade_nodes":cand_nodes,"oracle_baseline_grade_nodes":base_nodes};rows.append(r);print(json.dumps(r,sort_keys=True),flush=True)
    finally:
        for e in (cand,base,oracle_select,oracle_best,oracle_cand,oracle_base):e.quit()

    def stats(k):
        xs=[r[k] for r in rows];return {"mean":statistics.fmean(xs),"median":statistics.median(xs),"max":max(xs),"zero_count":sum(x==0 for x in xs)}
    out={"schema":"LV_ISOLATED_ROOT_REGRET_V2","candidate":a.candidate_label,"positions":len(rows),"choice_nodes":a.choice_nodes,"oracle_nodes_per_search":a.oracle_nodes,"candidate_regret":stats("candidate_regret_cp"),"baseline_regret":stats("baseline_regret_cp"),"candidate_oracle_agreement":sum(r["candidate_matches_oracle"] for r in rows)/len(rows),"baseline_oracle_agreement":sum(r["baseline_matches_oracle"] for r in rows)/len(rows),"mean_regret_delta_candidate_minus_baseline":statistics.fmean(r["candidate_regret_cp"]-r["baseline_regret_cp"] for r in rows),"rows":rows}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print("SUMMARY",json.dumps({k:v for k,v in out.items() if k!='rows'},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
