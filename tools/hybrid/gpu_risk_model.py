#!/usr/bin/env python3
"""GPU-assisted reply/risk/regret scoring for Leviathan Hybrid.

The model is intentionally advisory. CPU alpha-beta remains authoritative.
If no trained checkpoint is available, the scorer uses a deterministic
hand-coded prior so the hybrid controller remains functional and safe.
"""
from __future__ import annotations
import argparse,json,math
from dataclasses import dataclass
from pathlib import Path
from typing import List,Sequence

FEATURE_NAMES=("rank","score_cp","score_gap_cp","depth","seldepth","depth_gap","nodes_log10","nps_log10","hashfull_frac","pv_len","predicted_reply","mate_flag")
@dataclass(frozen=True)
class ReplyFeatures:
    rank:float;score_cp:float;score_gap_cp:float;depth:float;seldepth:float;nodes:float;nps:float;hashfull:float;pv_len:float;predicted_reply:float;mate_flag:float=0.
    def vector(self)->List[float]:return [float(self.rank),float(self.score_cp),float(self.score_gap_cp),float(self.depth),float(self.seldepth),float(self.seldepth-self.depth),math.log10(max(1.,float(self.nodes))),math.log10(max(1.,float(self.nps))),max(0.,min(1.,float(self.hashfull)/1000.)),float(self.pv_len),float(self.predicted_reply),float(self.mate_flag)]
class GpuRiskScorer:
    def __init__(self,device="auto",checkpoint=None):
        self.requested_device=device;self.checkpoint=checkpoint;self.torch=None;self.device="cpu";self.model=None;self.mode="heuristic";self.heads=3;self.normalizer_mean=None;self.normalizer_std=None;self.metrics={};self._init_torch();
        if checkpoint:self._load_checkpoint(checkpoint)
    def _init_torch(self):
        try:import torch
        except Exception:return
        self.torch=torch
        if self.requested_device=="off":return
        if self.requested_device=="cpu":self.device="cpu"
        elif self.requested_device in ("cuda","auto") and torch.cuda.is_available():self.device="cuda"
        else:self.device="cpu"
    def _load_checkpoint(self,checkpoint):
        if self.torch is None:return
        p=Path(checkpoint)
        if not p.exists():return
        torch=self.torch;payload=torch.load(p,map_location=self.device)
        hidden=int(payload.get("hidden",48));heads=int(payload.get("heads",3));m=torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES),hidden),torch.nn.SiLU(),torch.nn.Linear(hidden,hidden),torch.nn.SiLU(),torch.nn.Linear(hidden,heads))
        try:m.load_state_dict(payload.get("state_dict",payload))
        except Exception:return
        m.to(self.device);m.eval();self.model=m;self.mode="checkpoint";self.heads=heads;self.normalizer_mean=payload.get("normalizer_mean");self.normalizer_std=payload.get("normalizer_std");self.metrics=payload.get("metrics",{})
    def _norm(self,vectors):
        if self.normalizer_mean is None or self.normalizer_std is None:return vectors
        return [[(x-m)/max(1e-6,s) for x,m,s in zip(v,self.normalizer_mean,self.normalizer_std)] for v in vectors]
    def score(self,rows:Sequence[ReplyFeatures])->List[dict]:
        if not rows:return []
        return self._score_model(rows) if self.model is not None and self.torch is not None else self._score_heuristic(rows)
    def _score_model(self,rows):
        torch=self.torch;x=torch.tensor(self._norm([r.vector() for r in rows]),dtype=torch.float32,device=self.device)
        with torch.inference_mode():z=self.model(x);reply_prob=torch.softmax(z[:,0],dim=0);risk=torch.sigmoid(z[:,1]);regret=torch.expm1(torch.nn.functional.softplus(z[:,2])) if self.heads>=3 else 25*risk
        return [{"reply_probability":float(p),"risk":float(r),"expected_regret_cp":float(max(0.,g)),"device":self.device,"mode":self.mode} for p,r,g in zip(reply_prob.cpu().tolist(),risk.cpu().tolist(),regret.cpu().tolist())]
    def _score_heuristic(self,rows):
        logits=[];risks=[];regrets=[]
        for r in rows:
            gap=max(0.,r.score_gap_cp);logits.append(-gap/70.+.35*r.predicted_reply-.08*max(0.,r.rank-1.));dg=max(0.,r.seldepth-r.depth);close=math.exp(-gap/90.);raw=-.8+1.25*close+.035*dg+.25*r.mate_flag;rk=1/(1+math.exp(-raw));risks.append(rk);regrets.append(8.+30.*rk+8.*min(2.,dg/10.))
        mx=max(logits);ex=[math.exp(v-mx) for v in logits];s=sum(ex) or 1.;probs=[v/s for v in ex]
        return [{"reply_probability":float(p),"risk":float(r),"expected_regret_cp":float(g),"device":self.device,"mode":self.mode} for p,r,g in zip(probs,risks,regrets)]
    def describe(self):return {"requested_device":self.requested_device,"device":self.device,"mode":self.mode,"checkpoint":self.checkpoint,"heads":self.heads,"normalized":self.normalizer_mean is not None,"metrics":self.metrics,"torch_available":self.torch is not None,"cuda_available":bool(self.torch is not None and self.torch.cuda.is_available())}
def self_test(device):
    scorer=GpuRiskScorer(device=device)
    if device=="cuda" and not scorer.describe()["cuda_available"]:
        print(json.dumps({"runtime":scorer.describe(),"error":"CUDA explicitly requested but unavailable"},indent=2));return 4
    rows=[ReplyFeatures(1,35,0,14,22,50000,750000,120,8,1),ReplyFeatures(2,22,13,14,25,50000,750000,120,8,0),ReplyFeatures(3,-40,75,14,19,50000,750000,120,7,0)]
    out=scorer.score(rows);print(json.dumps({"runtime":scorer.describe(),"scores":out},indent=2))
    if len(out)!=3 or not math.isclose(sum(x["reply_probability"] for x in out),1.,rel_tol=1e-5):return 2
    if out[0]["reply_probability"]<=out[2]["reply_probability"]:return 3
    return 0
def main():
    a=argparse.ArgumentParser();a.add_argument("--device",default="auto",choices=("auto","cpu","cuda","off"));a.add_argument("--checkpoint",default=None);a.add_argument("--self-test",action="store_true");x=a.parse_args()
    if x.self_test:return self_test(x.device)
    s=GpuRiskScorer(device=x.device,checkpoint=x.checkpoint);print(json.dumps(s.describe(),indent=2));return 4 if x.device=="cuda" and not s.describe()["cuda_available"] else 0
if __name__=="__main__":raise SystemExit(main())