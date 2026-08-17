#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import chess,chess.engine

def cfg(e,fund):
    o={}
    for n,v in [("Threads",1),("Hash",64)]:
        if n in e.options:o[n]=v
    if fund:
        for n,v in [("Leviathan Fundamentals",True),("Leviathan Fundamentals Authority",1),("Leviathan Quiet Overdrive",0)]:
            if n in e.options:o[n]=v
    if o:e.configure(o)

def root_score(info,root_color):
    return info["score"].pov(root_color).score(mate_score=100000)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--baseline",required=True);ap.add_argument("--oracle",required=True);ap.add_argument("--openings-json",required=True);ap.add_argument("--candidate-label",required=True);ap.add_argument("--candidate-fundamentals",action="store_true");ap.add_argument("--baseline-fundamentals",action="store_true");ap.add_argument("--choice-nodes",type=int,default=80000);ap.add_argument("--oracle-nodes",type=int,default=500000);ap.add_argument("--output",required=True);a=ap.parse_args()
    positions=[]
    for x in json.loads(Path(a.openings_json).read_text()):
        b=chess.Board()
        for u in x["moves"]:b.push_uci(u)
        positions.append((x["name"],b))
    cand=chess.engine.SimpleEngine.popen_uci(a.candidate,timeout=30);base=chess.engine.SimpleEngine.popen_uci(a.baseline,timeout=30);oracle=chess.engine.SimpleEngine.popen_uci(a.oracle,timeout=30);cfg(cand,a.candidate_fundamentals);cfg(base,a.baseline_fundamentals);cfg(oracle,False)
    rows=[]
    try:
        for i,(name,b) in enumerate(positions,1):
            root=b.turn;token=("pos",i)
            cm=cand.play(b,chess.engine.Limit(nodes=a.choice_nodes),game=token).move
            bm=base.play(b,chess.engine.Limit(nodes=a.choice_nodes),game=token).move
            oi=oracle.analyse(b,chess.engine.Limit(nodes=a.oracle_nodes),game=("oracle-root",i));om=oi["pv"][0];best=root_score(oi,root)
            def grade(m,tag):
                if m==om:return best,0
                q=b.copy();q.push(m);inf=oracle.analyse(q,chess.engine.Limit(nodes=a.oracle_nodes),game=(tag,i));sc=root_score(inf,root);return sc,max(0,best-sc)
            cs,cr=grade(cm,"cand");bs,br=grade(bm,"base")
            r={"position":i,"name":name,"fen":b.fen(),"oracle_move":om.uci(),"oracle_score":best,"candidate_move":cm.uci(),"candidate_score":cs,"candidate_regret_cp":cr,"baseline_move":bm.uci(),"baseline_score":bs,"baseline_regret_cp":br,"candidate_matches_oracle":cm==om,"baseline_matches_oracle":bm==om};rows.append(r);print(json.dumps(r,sort_keys=True),flush=True)
    finally:cand.quit();base.quit();oracle.quit()
    def stats(k):
        xs=[r[k] for r in rows];return {"mean":statistics.fmean(xs),"median":statistics.median(xs),"max":max(xs),"zero_count":sum(x==0 for x in xs)}
    out={"candidate":a.candidate_label,"positions":len(rows),"choice_nodes":a.choice_nodes,"oracle_nodes_per_grade":a.oracle_nodes,"candidate_regret":stats("candidate_regret_cp"),"baseline_regret":stats("baseline_regret_cp"),"candidate_oracle_agreement":sum(r["candidate_matches_oracle"] for r in rows)/len(rows),"baseline_oracle_agreement":sum(r["baseline_matches_oracle"] for r in rows)/len(rows),"mean_regret_delta_candidate_minus_baseline":statistics.fmean(r["candidate_regret_cp"]-r["baseline_regret_cp"] for r in rows),"rows":rows}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print("SUMMARY",json.dumps({k:v for k,v in out.items() if k!='rows'},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
