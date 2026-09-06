"""Training-free, activation-conditioned FFN relevance and tail diagnostics.

Computes gate/up ONCE. This spends the dense gate/up cost; it cannot claim to avoid
an entire FFN. Dense output remains authoritative by default. No raw weight-norm
product is used as a useful-confidence estimate.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from .cells import FrozenCellBank,CellPolicy
from ..consumer.quantization import slice_weight


@dataclass(frozen=True)
class ActivationPolicy:
    width: int=128
    seed: int=4
    max_cells: int=16
    absolute_tail_tolerance: float=0.
    mode: str="observe"

    def __post_init__(self):
        if min(self.width,self.seed,self.max_cells)<1 or self.seed>self.max_cells:
            raise ValueError("Invalid activation-cell budgets")
        if self.mode not in {"observe","bounded"}:raise ValueError("Unknown mode")
        if not math.isfinite(self.absolute_tail_tolerance) or self.absolute_tail_tolerance<0:
            raise ValueError("Finite nonnegative tail tolerance required")


class ActivationCellBank:
    """Read-only view, not a second model or a registered parameter copy."""
    def __init__(self,donor:nn.Module,width:int=128):
        self.reference=FrozenCellBank(donor,width)
        self.donor=donor;self.width=width
        self.count=self.reference.bank.count
        self.hidden=self.reference.bank.hidden
        self._down_frob=None;self._col_norms=None

    def _prepare(self):
        if self._down_frob is not None:return
        frob=[];cols=[]
        for i in range(self.count):
            w=slice_weight(self.donor.down_proj,cols=slice(i*self.width,(i+1)*self.width)).double()
            frob.append(w.norm());cols.append(w.norm(dim=0))
        self._down_frob=torch.stack(frob)
        self._col_norms=torch.stack(cols)

    def analyze(self,x:Tensor,policy:ActivationPolicy=ActivationPolicy()):
        if policy.width!=self.width:raise ValueError("Cell width mismatch")
        if torch.is_grad_enabled():raise RuntimeError("Activation diagnostics require inference/no_grad mode")
        if not torch.isfinite(x).all():raise FloatingPointError("Nonfinite FFN input")
        self._prepare()
        shape=x.shape;flat=x.reshape(-1,self.hidden)
        if not len(flat):return self.donor(x),{"empty":True}
        # Preserve original activation and matrix dtypes in the donor path.
        gate=self.donor.gate_proj(flat);up=self.donor.up_proj(flat)
        z=getattr(self.donor,'act_fn',F.silu)(gate)*up
        if not torch.isfinite(z).all():raise FloatingPointError("Nonfinite actual SwiGLU activations")
        groups=z.reshape(len(flat),self.count,self.width).double()
        frob=groups.norm(dim=-1)*self._down_frob
        triangle=(groups.abs()*self._col_norms).sum(-1)
        bounds=torch.minimum(frob,triangle)
        cap=min(policy.max_cells,self.count)
        order=bounds.argsort(dim=-1,descending=True)
        sorted_bounds=bounds.gather(1,order)
        # Reverse summation of excluded terms avoids catastrophic cancellation
        # from total-minus-selected. All numbers are nonnegative.
        tails=torch.flip(torch.cumsum(torch.flip(sorted_bounds,[-1]),-1),[-1])
        tail_after=torch.cat((tails[:,1:],torch.zeros_like(tails[:,:1])),dim=-1)
        eligible=(tail_after<=policy.absolute_tail_tolerance)
        sizes=torch.arange(1,self.count+1,device=x.device)[None].expand(len(flat),-1)
        valid=eligible&(sizes>=min(policy.seed,self.count))&(sizes<=cap)
        requested=torch.where(valid,sizes,self.count+1).amin(-1)
        selected_count=torch.where(requested<=cap,requested,cap)
        ranks=torch.arange(cap,device=x.device)[None]
        mask=ranks<selected_count[:,None]
        ids=order[:,:cap]
        selected=torch.zeros_like(flat)
        for cell_t in torch.unique(ids[mask]):
            cell=int(cell_t)
            rows=((ids==cell)&mask).any(-1).nonzero(as_tuple=True)[0]
            r=slice(cell*self.width,(cell+1)*self.width)
            contribution=F.linear(z[rows,r],slice_weight(self.donor.down_proj,cols=r).to(z.dtype))
            selected=selected.index_add(0,rows,contribution)
        if self.donor.down_proj.bias is not None:selected=selected+self.donor.down_proj.bias.to(selected.dtype)
        dense=self.donor.down_proj(z)
        remainder_mask=torch.ones_like(bounds,dtype=torch.bool)
        remainder_mask.scatter_(1,ids,~mask)
        analytic_tail=bounds.masked_fill(~remainder_mask,0).sum(-1)
        old_tail=self.reference.bounds(flat).double().masked_fill(~remainder_mask,0).sum(-1)
        measured=(dense.float()-selected.float()).norm(dim=-1)
        fallback=analytic_tail>policy.absolute_tail_tolerance
        # Both modes deliberately compute a dense audit here. Bounded mode is
        # an experimental numerical-quality path, NOT a speed implementation.
        out=dense if policy.mode=="observe" else torch.where(fallback[:,None],dense,selected)
        ratio=old_tail/analytic_tail.clamp_min(1e-300)
        trace={"mode":"activation_"+policy.mode,"tokens":len(flat),
               "selected_cells_per_token":selected_count.cpu().tolist(),
               "max_activation_tail_bound":float(analytic_tail.max()),
               "mean_activation_tail_bound":float(analytic_tail.mean()),
               "max_legacy_weight_only_bound":float(old_tail.max()),
               "median_bound_tightening_factor":float(ratio.median()),
               "max_measured_excluded_output_l2":float(measured.max()),
               "max_full_output_l2":float(dense.float().norm(dim=-1).max()),
               "dense_fallback_tokens":int(fallback.sum()) if policy.mode=="bounded" else len(flat),
               "routes":ids.cpu().tolist(),"dense_gate_up_computed":True,
               "dense_down_audit_computed":True,"speedup_claim":False,
               "bound_scope":"real_arithmetic_on_actual_activations_not_interval_or_logit_certificate"}
        return out.reshape(shape),trace
