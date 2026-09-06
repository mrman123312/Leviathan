"""No-fit compute decisions. Stability/confidence proxies are NOT truth certificates.

The token controller inspects the owning model's ORIGINAL final norm/output head.
MCQ decisions use scores only, never answer keys; their extra work must be charged.
Thresholds are fixed before this version's ARC evaluation, not optimized on its labels.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Sequence
import torch
from torch import Tensor


@dataclass(frozen=True)
class StopPolicy:
    min_passes: int = 2
    patience: int = 1
    initial_pmax: float = .95
    initial_margin: float = 4.0
    initial_entropy: float = .20
    js_limit: float = 1e-4
    logprob_limit: float = .04
    entropy_limit: float = .003
    delta_limit: float = .03
    topk: int = 32
    chunk_positions: int = 16

    def __post_init__(self):
        if min(self.min_passes,self.patience,self.topk,self.chunk_positions)<1:
            raise ValueError("Positive stop-policy budgets required")
        for name,value in asdict(self).items():
            if not math.isfinite(value) or value<0:
                raise ValueError(f"Invalid stop threshold: {name}")
        if not 0<self.initial_pmax<=1 or not 0<=self.initial_entropy<=1:
            raise ValueError("Invalid confidence-proxy threshold")


@dataclass
class PredictionSummary:
    winner: Tensor
    margin: Tensor
    pmax: Tensor
    entropy: Tensor
    top_ids: Tensor
    top_probs: Tensor


def summarize(logits: Tensor, topk: int=32) -> PredictionSummary:
    if logits.ndim!=2 or logits.shape[-1]<2 or not torch.isfinite(logits).all():
        raise FloatingPointError("Prediction probe needs finite [positions,vocabulary] logits")
    values=logits.float()
    logp=values.log_softmax(-1)
    p=logp.exp()
    best,ids=values.topk(min(topk,values.shape[-1]),-1)
    # topk can be 1, but the top-two margin always uses two values.
    top_two=values.topk(2,-1).values
    return PredictionSummary(ids[:,0],top_two[:,0]-top_two[:,1],p.amax(-1),
        -(p*logp).sum(-1)/math.log(values.shape[-1]),ids,p.gather(-1,ids))


def compare(previous: PredictionSummary, current_logits: Tensor, *, topk: int=32):
    """JS on the PREVIOUS top-k categories plus one 'other' bin.

    This is a coarsened distribution-change proxy, not full-vocabulary JS and not a
    bound on answer correctness. Log-probability changes ignore common logit shifts.
    """
    now=summarize(current_logits,topk)
    logp=current_logits.float().log_softmax(-1)
    q=logp.exp().gather(-1,previous.top_ids)
    p=previous.top_probs
    p=torch.cat((p,(1-p.sum(-1,keepdim=True)).clamp_min(0)),dim=-1)
    q=torch.cat((q,(1-q.sum(-1,keepdim=True)).clamp_min(0)),dim=-1)
    p=p/p.sum(-1,keepdim=True);q=q/q.sum(-1,keepdim=True)
    m=(p+q)*.5
    safe=lambda v:v.clamp_min(1e-30).log()
    js=.5*((p*(safe(p)-safe(m))).sum(-1)+(q*(safe(q)-safe(m))).sum(-1))
    change=(logp.gather(-1,previous.top_ids)-safe(previous.top_probs)).abs().amax(-1)
    return now,{"same_winner":now.winner==previous.winner,"coarse_js":js,
                "max_top_logprob_change":change,"entropy_change":(now.entropy-previous.entropy).abs(),
                "margin_change":(now.margin-previous.margin).abs()}


def initial_stop(summary: PredictionSummary, policy: StopPolicy) -> Tensor:
    return ((summary.pmax>=policy.initial_pmax)&(summary.margin>=policy.initial_margin)
            &(summary.entropy<=policy.initial_entropy))


def stable_stop(metrics: dict, relative_delta: Tensor, policy: StopPolicy) -> Tensor:
    return (metrics["same_winner"]&(metrics["coarse_js"]<=policy.js_limit)
            &(metrics["max_top_logprob_change"]<=policy.logprob_limit)
            &(metrics["entropy_change"]<=policy.entropy_limit)&(relative_delta<=policy.delta_limit))


@dataclass(frozen=True)
class ChoicePolicy:
    direct_pmax: float=.92
    direct_margin: float=2.0
    direct_entropy: float=.40
    js_limit: float=.001
    score_movement: float=.10
    entropy_movement: float=.02

    def __post_init__(self):
        if any(not math.isfinite(v) or v<0 for v in asdict(self).values()):
            raise ValueError("Invalid choice controller threshold")
        if not 0<self.direct_pmax<=1 or not 0<=self.direct_entropy<=1:
            raise ValueError("Invalid choice confidence proxy")


def choice_summary(scores: Sequence[float]) -> dict:
    if len(scores)<2 or any(not math.isfinite(x) for x in scores):
        raise ValueError("At least two finite candidate scores required")
    shifted=[float(x)-max(scores) for x in scores]
    z=sum(math.exp(x) for x in shifted)
    p=[math.exp(x)/z for x in shifted]
    order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
    return {"winner":order[0],"pmax":max(p),"margin":scores[order[0]]-scores[order[1]],
        "entropy":-sum(x*math.log(max(x,1e-300)) for x in p)/math.log(len(p)),
        "probabilities":p,"scores":[float(v) for v in scores],"calibrated":False}


def choose_next(scores: Sequence[float], *, previous: Sequence[float]|None=None,
                policy: ChoicePolicy=ChoicePolicy()) -> dict:
    """Task-level DIRECT -> REFINE -> EXPLORE decision, WITHOUT gold labels.

    Candidate alternatives are part of the task, not held-out ground-truth access.
    A stable wrong answer may halt. The benchmark measures that failure possibility.
    """
    current=choice_summary(scores)
    if previous is None:
        direct=(current["pmax"]>=policy.direct_pmax and current["margin"]>=policy.direct_margin
                and current["entropy"]<=policy.direct_entropy)
        return {"action":"DIRECT" if direct else "REFINE","reason":"confident_proxy" if direct else "uncertain_proxy",
                "signals":current,"truth_certificate":False}
    old=choice_summary(previous)
    if len(old["scores"])!=len(current["scores"]):raise ValueError("Choice identities/length changed")
    p,q=old["probabilities"],current["probabilities"]
    m=[(a+b)/2 for a,b in zip(p,q)]
    js=.5*sum(a*math.log(max(a,1e-300)/max(b,1e-300)) for a,b in zip(p,m))
    js+=.5*sum(a*math.log(max(a,1e-300)/max(b,1e-300)) for a,b in zip(q,m))
    # Relative choice scores, invariant to any common additive likelihood shift.
    a=[v-max(old["scores"]) for v in old["scores"]]
    b=[v-max(current["scores"]) for v in current["scores"]]
    movement=max(abs(x-y) for x,y in zip(a,b))
    stable=(old["winner"]==current["winner"] and js<=policy.js_limit
            and movement<=policy.score_movement
            and abs(old["entropy"]-current["entropy"])<=policy.entropy_movement)
    return {"action":"STOP_REFINE" if stable else "EXPLORE",
            "reason":"stable_answer_proxy" if stable else "unresolved_change",
            "signals":{**current,"js":js,"relative_score_change":movement},"truth_certificate":False}
