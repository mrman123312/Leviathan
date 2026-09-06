"""Training-free ancestral cells, conservative discussion and bounded recruitment.

All proposals are actual frozen SwiGLU slices. No randomly initialized router,
confidence head, message head, or optimizer is used. Bounds are real-arithmetic
inequalities, not floating-point interval certificates or quality guarantees.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ..consumer.cells import SwiGLUCells
from ..consumer.quantization import slice_weight


@dataclass(frozen=True)
class CellPolicy:
    width: int = 128
    seed: int = 4
    max_cells: int = 16
    rounds: int = 2
    neighbors: int = 4
    tail_tolerance: float = 0.0
    mode: str = "off"             # off, observe, bounded
    message_mix: float = 0.25
    state_mix: float = 0.0

    def __post_init__(self):
        if min(self.width, self.seed, self.max_cells, self.rounds, self.neighbors) < 1:
            raise ValueError("Positive cell geometry/budgets required")
        if self.seed > self.max_cells or self.rounds > 8:
            raise ValueError("Invalid cell budgets")
        if self.mode not in {"off", "observe", "bounded"}:
            raise ValueError("Unknown cell execution mode")
        if not math.isfinite(self.tail_tolerance) or self.tail_tolerance < 0:
            raise ValueError("Invalid tail tolerance")
        if not 0 <= self.state_mix <= 1:
            raise ValueError("Invalid local-state influence")
        if not 0 <= self.message_mix <= .5:
            raise ValueError("Stable discussion mix must be in [0, .5]")


@dataclass
class CellState:
    """Four observable moments per token/cell, kept only across depth in one call."""
    scope: str
    moments: Tensor | None = None
    old_query: Tensor | None = None
    old_scores: Tensor | None = None
    old_route: Tensor | None = None

    def reset(self):
        self.moments = self.old_query = self.old_scores = self.old_route = None


def conservative_discussion(messages: Tensor, *, rounds: int, neighbors: int,
                            mix: float) -> tuple[Tensor, list[float]]:
    """Symmetric graph diffusion. Pair exchange conserves sum_i message_i.

    Update is M' = M + mix (A M - degree(A) M)/max_degree, with A symmetric.
    Therefore 1^T(M'-M)=0 in real arithmetic; sum remains the donor contribution.
    It changes local proposals, not the aggregate, and is not itself extra knowledge.
    """
    if messages.ndim != 3 or messages.shape[1] < 1 or min(rounds, neighbors) < 1:
        raise ValueError("Expected [tokens,cells,hidden] and positive budgets")
    if not 0 <= mix <= .5 or not torch.isfinite(messages).all():
        raise ValueError("Invalid diffusion input")
    current = messages.float()
    trace = []
    count = current.shape[1]
    for _ in range(rounds):
        normalized = F.normalize(current, dim=-1)
        similarity = normalized @ normalized.transpose(-1, -2)
        if count == 1:
            trace.append(0.0)
            continue
        eye = torch.eye(count, device=current.device, dtype=torch.bool)[None]
        score = similarity.masked_fill(eye, -torch.inf)
        ids = score.topk(min(neighbors, count-1), -1).indices
        directed = torch.zeros_like(similarity).scatter(-1, ids, 1.)
        # Mutual-neighbor graph is symmetric AND keeps each degree <= neighbors.
        adjacency = torch.minimum(directed, directed.transpose(-1, -2))
        degree = adjacency.sum(-1, keepdim=True)
        step = (adjacency @ current - degree * current) / degree.amax(-2, keepdim=True).clamp_min(1)
        current = current + mix * step
        trace.append(float(current.var(1, unbiased=False).mean().detach()))
    return current.to(messages.dtype), trace


class FrozenCellBank:
    """Read-only parameter views and parameter-derived keys within one FFN.

    Pre-execution bounds may be loose. When the cell cap cannot satisfy the bound,
    bounded mode falls back to the original dense FFN; it never calls the cap a pass.
    All routing/metadata work is overhead until measured otherwise.
    """
    def __init__(self, donor: nn.Module, width: int = 128):
        self.bank = SwiGLUCells(donor, width)
        self.donor = donor
        self._metadata = None

    def _prepare(self):
        if self._metadata is not None:
            return self._metadata
        data, keys = [], []
        with torch.no_grad():
            for cid in range(self.bank.count):
                r = slice(cid*self.bank.width, (cid+1)*self.bank.width)
                gate = slice_weight(self.donor.gate_proj, rows=r).float()
                up = slice_weight(self.donor.up_proj, rows=r).float()
                down = slice_weight(self.donor.down_proj, cols=r).float()
                bg = self.donor.gate_proj.bias
                bu = self.donor.up_proj.bias
                data.append(torch.stack((gate.norm(), up.norm(), down.norm(),
                    gate.new_zeros(()) if bg is None else bg[r].float().norm(),
                    up.new_zeros(()) if bu is None else bu[r].float().norm())))
                keys.append(F.normalize(gate.mean(0)+up.mean(0), dim=0))
        self._metadata = torch.stack(data), torch.stack(keys)
        return self._metadata

    def bounds(self, x: Tensor) -> Tensor:
        norms, _ = self._prepare()
        radius = x.float().norm(dim=-1, keepdim=True)
        return (radius*norms[:,0] + norms[:,3]) * (radius*norms[:,1] + norms[:,4]) * norms[:,2]

    def run(self, x: Tensor, policy: CellPolicy, *, state: CellState | None = None,
            scope: str = "ephemeral") -> tuple[Tensor, dict]:
        if policy.mode == "off":
            return self.donor(x), {"mode": "off", "extra_cell_work": 0}
        if policy.width != self.bank.width:
            raise ValueError("Policy and cell-bank geometry differ")
        shape = x.shape
        flat = x.reshape(-1, self.bank.hidden)
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Nonfinite cell input")
        if not len(flat):
            return self.donor(x), {"mode": policy.mode, "empty": True}
        if state is not None and state.scope != scope:
            raise ValueError("Cell state belongs to a different request/layer scope")
        _, keys = self._prepare()
        bounds = self.bounds(flat)
        cap = min(policy.max_cells, self.bank.count)
        seed = min(policy.seed, cap)
        # Cheap frozen-weight proxy, not an estimate of epistemic confidence.
        query = F.normalize(flat.float(), dim=-1)
        reuse = False
        if state is not None and state.old_query is not None and policy.state_mix == 0:
            if state.old_query.shape == query.shape and state.old_route.shape[-1] == seed:
                # |q.k| is 1-Lipschitz in q when ||k||<=1. Strict separation
                # preserves top-k membership, with a conservative FP guard.
                selected = torch.zeros_like(state.old_scores, dtype=torch.bool).scatter(1, state.old_route, True)
                if seed == self.bank.count:
                    reuse = True
                else:
                    gap = state.old_scores.masked_fill(~selected, torch.inf).amin(-1) - state.old_scores.masked_fill(selected, -torch.inf).amax(-1)
                    displacement = (query.double()-state.old_query.double()).norm(dim=-1)*keys.double().norm(dim=-1).max()
                    guard = 32*torch.finfo(query.dtype).eps*(state.old_scores.abs().amax(-1)+1)
                    reuse = bool((gap > 2*displacement+guard).all())
        if reuse:
            active = state.old_route.clone()
        else:
            score = (query @ keys.T).abs()
            if state is not None and policy.state_mix and state.moments is not None:
                if state.moments.shape[:2] != score.shape:
                    raise ValueError("Cell moments belong to different token positions")
                history = state.moments[..., 1]
                score = score + policy.state_mix*history/history.amax(-1,keepdim=True).clamp_min(1e-12)
            active = score.topk(seed, -1).indices
            if state is not None:
                state.old_query=query.clone();state.old_scores=score.clone();state.old_route=active.clone()
        bodies = self.bank.body(flat[:,None].expand(-1,seed,-1).reshape(-1,self.bank.hidden),
                                active.reshape(-1)).reshape(len(flat),seed,-1)
        disagreements, recruited = [], 0
        for round_index in range(policy.rounds):
            discussed, ds = conservative_discussion(bodies, rounds=1,
                neighbors=policy.neighbors, mix=policy.message_mix)
            disagreements.extend(ds)
            # Recruitment follows actual peer proposals, still within this FFN.
            selected = torch.zeros_like(bounds, dtype=torch.bool).scatter(1, active, True)
            tail = bounds.masked_fill(selected, 0).sum(-1)
            need = tail > policy.tail_tolerance
            available = cap-active.shape[1]
            # The last round is discussion/commit only, so every recruit joins
            # a subsequent discussion before any final proposal is committed.
            if available <= 0 or not bool(need.any()) or round_index + 1 >= policy.rounds:
                break
            count = min(seed, available)
            # Conservative discussion preserves the mean. Recruiting from that
            # mean would disconnect communication from recruitment entirely.
            # Instead peers nominate independently; the budget resolves nominees.
            queries = F.normalize(discussed.float(), dim=-1)
            recruitment = (queries @ keys.T).abs().amax(1).masked_fill(selected, -torch.inf)
            extra = recruitment.topk(count, -1).indices
            more = self.bank.body(flat[:,None].expand(-1,count,-1).reshape(-1,self.bank.hidden),
                                  extra.reshape(-1)).reshape(len(flat),count,-1)
            active = torch.cat((active, extra), 1)
            bodies = torch.cat((bodies, more), 1)
            recruited += len(flat)*count
        selected = torch.zeros_like(bounds, dtype=torch.bool).scatter(1, active, True)
        tail = bounds.masked_fill(selected, 0).sum(-1)
        # Conservative discussion is observed, never substituted for the sum:
        # this avoids accumulation/summation-order noise from otherwise neutral peers.
        result = bodies.sum(1)
        if self.donor.down_proj.bias is not None:
            result = result+self.donor.down_proj.bias.to(result.dtype)
        fallback = tail > policy.tail_tolerance
        if policy.mode == "observe":
            result = self.donor(flat)
        elif bool(fallback.any()):
            rows = fallback.nonzero(as_tuple=True)[0]
            result = result.index_copy(0, rows, self.donor(flat[rows]))
        if state is not None:
            if state.moments is None:
                state.moments=flat.new_zeros(len(flat),self.bank.count,4,dtype=torch.float32)
            old=state.moments.gather(1,active[...,None].expand(-1,-1,4))
            norms=bodies.float().norm(dim=-1)
            direction=F.normalize(bodies.float(),dim=-1)
            agreement=(direction*F.normalize(bodies.float().mean(1),dim=-1)[:,None]).sum(-1)
            updated=torch.stack((norms,.75*old[...,1]+.25*norms,agreement,old[...,3]+1),dim=-1)
            state.moments=state.moments.scatter(1,active[...,None].expand(-1,-1,4),updated)
        return result.reshape(shape), {"mode": policy.mode, "seed_route_reused": reuse,
            "local_state_updated": state is not None, "active_cells_per_token": active.shape[1],
            "cell_pairs_executed": int(bodies.shape[0]*bodies.shape[1]), "recruited_pairs": recruited,
            "dense_fallback_tokens": int(fallback.sum()) if policy.mode == "bounded" else len(flat),
            "max_analytic_tail_bound": float(tail.max()), "bound_kind": "real_arithmetic_FFN_L2_not_logit_quality",
            "disagreement_proxy": disagreements, "routes": active.detach().cpu().tolist(),
            "confidence_calibrated": False, "speedup_claim": False}

    def partitions(self, width: int) -> tuple[tuple[int,int], ...]:
        """Lossless split/merge of contiguous views, not new pretrained knowledge."""
        if width < 1 or self.bank.intermediate % width:
            raise ValueError("New width must divide the FFN")
        return tuple((i,i+width) for i in range(0,self.bank.intermediate,width))

    def zero_cells(self) -> tuple[int, ...]:
        """A zero down slice is sufficient for exact pruning for all finite inputs."""
        result=[]
        for cid in range(self.bank.count):
            r=slice(cid*self.bank.width,(cid+1)*self.bank.width)
            if bool((slice_weight(self.donor.down_proj,cols=r)==0).all()):
                result.append(cid)
        return tuple(result)
