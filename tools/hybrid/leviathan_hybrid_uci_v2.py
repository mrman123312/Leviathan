#!/usr/bin/env python3
"""P18.2 hybrid controller: expected-regret scheduling + ponder portfolio annealing."""
from __future__ import annotations
import threading, time
try:
    from leviathan_hybrid_uci import *
except ImportError:
    from .leviathan_hybrid_uci import *
try:
    from gpu_risk_model import ReplyFeatures
except ImportError:
    from .gpu_risk_model import ReplyFeatures

class HybridProxyV2(HybridProxy):
    def __init__(self,args):
        super().__init__(args)
        self.regret_weight=max(0.0,args.regret_weight)
        self.anneal_seconds=max(0.0,args.anneal_seconds)
        self.min_final_scouts=max(1,args.min_final_scouts)

    def candidate_value(self,c):
        danger=(1.0+self.risk_weight*c.risk)*(1.0+self.regret_weight*max(0.0,getattr(c,'expected_regret_cp',0.0))/50.0)
        return max(1e-9,c.reply_probability*danger)

    def _prepare_multi_ponder(self,s):
        try:
            k=max(1,min(self.max_scouts,self.full_threads))
            cands=self.generate_reply_candidates(s,max(k+4,k))
            if s.cancel.is_set() or self.session is not s:return
            if not cands:
                if s.predicted_reply:cands=[ReplyCandidate(s.predicted_reply,1,0,0,0,0,0,0,0,0,True)]
                else:return
            if s.predicted_reply and all(c.move!=s.predicted_reply for c in cands):
                best=cands[0].score_cp if cands else 0;cands.append(ReplyCandidate(s.predicted_reply,len(cands)+1,best-10,0,0,0,0,0,0,0,True))
            for c in cands:c.predicted=c.predicted or c.move==s.predicted_reply
            best=max(c.score_cp for c in cands)
            feats=[ReplyFeatures(c.rank,c.score_cp,max(0,best-c.score_cp),c.depth,c.seldepth,c.nodes,c.nps,c.hashfull,c.pv_len,1. if c.predicted else 0.,float(c.mate_flag)) for c in cands]
            scores=self.scorer.score(feats)
            for c,x in zip(cands,scores):
                c.reply_probability=float(x['reply_probability']);c.risk=float(x['risk']);c.expected_regret_cp=float(x.get('expected_regret_cp',25*c.risk));c.utility=self.candidate_value(c)
            cands.sort(key=lambda c:(c.utility,c.predicted,-c.rank),reverse=True)
            selected=cands[:k]
            if s.predicted_reply and all(c.move!=s.predicted_reply for c in selected):selected[-1]=next(c for c in cands if c.move==s.predicted_reply)
            selected.sort(key=lambda c:c.utility,reverse=True)
            alloc=allocate_integer_budget(self.full_threads,[c.utility for c in selected],1)
            engines=[self.primary]+self.ensure_spares(max(0,len(selected)-1));assign={}
            for cand,threads,e in zip(selected,alloc,engines):
                if s.cancel.is_set() or self.session is not s:return
                h=self.full_hash if e is self.primary else min(self.full_hash,self.scout_hash)
                self.configure_search_engine(e,max(1,threads),h,1);pos=append_move_to_position(s.after_our_move_cmd,cand.move);e.drain();e.send(pos);e.send(s.go_ponder_cmd);assign[norm(pos)]=ScoutAssignment(cand,e,norm(pos),max(1,threads),h,time.monotonic())
            with self.state_lock:
                if s.cancel.is_set() or self.session is not s:return
                s.candidates=selected;s.assignments=assign;s.ready.set()
            self.log('ponder_pool_ready_v2',generation=s.generation,setup_ms=int((time.monotonic()-s.started_at)*1000),candidates=[{'move':c.move,'rank':c.rank,'reply_probability':c.reply_probability,'risk':c.risk,'expected_regret_cp':c.expected_regret_cp,'utility':c.utility,'threads':alloc[i]} for i,c in enumerate(selected)],gpu=self.scorer.describe())
            if self.anneal_seconds>0 and len(selected)>self.min_final_scouts:
                threading.Thread(target=self._anneal_pool,args=(s,),daemon=True,name=f'ponder-anneal-{s.generation}').start()
        except Exception as exc:self.log('ponder_prepare_error',generation=s.generation,error=repr(exc));s.ready.set()

    def _anneal_pool(self,s):
        target=max(self.min_final_scouts,1)
        while True:
            if s.cancel.wait(self.anneal_seconds):return
            with self.state_lock:
                if self.session is not s or not s.ready.is_set():return
                live=[a for a in s.assignments.values() if not a.stopped]
            if len(live)<=target:return
            live.sort(key=lambda a:a.candidate.utility,reverse=True)
            keep_n=max(target,(len(live)+1)//2);keep=live[:keep_n];drop=live[keep_n:]
            for a in drop:self.stop_assignment(a,False)
            alloc=allocate_integer_budget(self.full_threads,[a.candidate.utility for a in keep],1)
            for a,t in zip(keep,alloc):
                if s.cancel.is_set() or self.session is not s:return
                if a.threads==t:continue
                self.stop_assignment(a,False);a.stopped=False
                self.configure_search_engine(a.engine,t,a.hash_mb,1);a.engine.drain();a.engine.send(a.position_cmd);a.engine.send(s.go_ponder_cmd);a.threads=t;a.started_at=time.monotonic()
            self.log('ponder_pool_anneal',generation=s.generation,elapsed_ms=int((time.monotonic()-s.started_at)*1000),survivors=[{'move':a.candidate.move,'utility':a.candidate.utility,'threads':a.threads} for a in keep],dropped=[a.candidate.move for a in drop])

def build_v2_parser():
    p=build_arg_parser();p.description='Leviathan P18.2 CPU+GPU expected-regret multi-ponder proxy'
    p.add_argument('--regret-weight',type=float,default=1.0)
    p.add_argument('--anneal-seconds',type=float,default=2.0)
    p.add_argument('--min-final-scouts',type=int,default=2)
    return p

def main():return HybridProxyV2(build_v2_parser().parse_args()).run()
if __name__=='__main__':raise SystemExit(main())
