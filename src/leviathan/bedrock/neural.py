"""Frozen pretrained band recurrence, request-local plasticity and one-model search.

No new nn.Parameter, optimizer, fit, adapter training, or remote model is created.
The repeated function is the donor's existing decoder band. Reuse may be harmful;
this module guarantees boundaries, NOT improved intelligence or lower latency.
"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import inspect
import math
from threading import RLock
from typing import Callable
import torch
from torch import Tensor, nn
from .cells import CellPolicy, CellState, FrozenCellBank
from .contracts import Meter, Outcome, stable_hash
from ..consumer.efficiency import ExactDeltaCache, CacheScope
from ..consumer.speculation import verify_greedy, verify_sampled


@dataclass(frozen=True)
class FrozenPolicy:
    start: int = -4
    end: int = -1
    passes: int = 1                    # includes the inherited first pass
    gain: float = 0.0
    relative_radius: float = .25
    halt_delta: float = 0.0           # heuristic, not confidence in truth
    halt_patience: int = 2
    fast_gain: float = 0.0
    cells: CellPolicy = field(default_factory=CellPolicy)
    exact_ffn_cache: bool = False
    feedback: str = "repeat"           # repeat, anchored_difference

    def __post_init__(self):
        if self.feedback not in {"repeat", "anchored_difference"}:
            raise ValueError("Unknown feedback rule")
        if self.exact_ffn_cache and self.cells.mode != "off":
            raise ValueError("Exact FFN cache is only enabled for pure stateless donor calls")
        if not 1 <= self.passes <= 16 or self.halt_patience < 1:
            raise ValueError("Invalid bounded recurrence")
        if not 0 <= self.gain <= 1 or not 0 <= self.fast_gain <= 1:
            raise ValueError("Gains must be finite and in [0,1]")
        if not math.isfinite(self.relative_radius) or not 0 <= self.relative_radius <= 2:
            raise ValueError("Invalid trust-region radius")
        if not math.isfinite(self.halt_delta) or self.halt_delta < 0:
            raise ValueError("Invalid convergence threshold")

    @property
    def neutral(self):
        return ((self.passes == 1 or self.gain == 0) and self.fast_gain == 0
                and self.cells.mode == "off" and not self.exact_ffn_cache)


class FastAssociations:
    """Finite key/value low-rank activation map with explicit provenance.

    delta(h) = sum_j <h,k_j/||k_j||> v_j, projected into a relative L2 ball.
    Task-local; no donor tensor is changed. This is associative state, not a claim
    of biologically faithful plasticity or a learned correction model.
    """
    def __init__(self, scope: tuple[str, str, int], capacity: int = 8):
        if capacity < 1:
            raise ValueError("Positive association capacity required")
        self.scope, self.capacity, self.version = scope, capacity, 0
        self._entries: list[tuple[Tensor, Tensor, str]] = []

    def bind(self, key: Tensor, correction: Tensor, *, evidence: str):
        if key.ndim != 1 or key.shape != correction.shape or not evidence:
            raise ValueError("Aligned vectors and evidence reference required")
        if not torch.isfinite(key).all() or not torch.isfinite(correction).all() or key.norm() == 0:
            raise ValueError("Finite, nonzero association key required")
        self._entries.append((key.detach().float().clone(), correction.detach().float().clone(), evidence))
        self._entries = self._entries[-self.capacity:]
        self.version += 1

    def apply(self, hidden: Tensor, *, scope: tuple[str, str, int], radius: float) -> Tensor:
        if scope != self.scope:
            raise ValueError("Fast state revision/layer/request scope mismatch")
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("Invalid association radius")
        if not self._entries or radius == 0:
            return hidden
        keys=torch.stack([k for k,_,_ in self._entries]).to(hidden.device)
        values=torch.stack([v for _,v,_ in self._entries]).to(hidden.device)
        keys=torch.nn.functional.normalize(keys, dim=-1)
        delta=(hidden.float() @ keys.T) @ values
        return project_delta(hidden, delta, radius)

    def fork(self):
        other=FastAssociations(self.scope,self.capacity)
        other._entries=[(k.clone(),v.clone(),e) for k,v,e in self._entries]
        other.version=self.version
        return other

    def reset(self):
        self._entries.clear()
        self.version += 1


def project_delta(anchor: Tensor, delta: Tensor, radius: float) -> Tensor:
    """Per-position L2 radius in real arithmetic, with ordinary FP rounding."""
    d=delta.float()
    limit=radius*anchor.float().norm(dim=-1,keepdim=True)
    scaled=d*(limit/d.norm(dim=-1,keepdim=True).clamp_min(1e-30)).clamp(max=1)
    out=anchor+scaled.to(anchor.dtype)
    if not torch.isfinite(out).all():
        raise FloatingPointError("Nonfinite bounded update")
    return out


class _CellCall(nn.Module):
    """Temporary read-only donor wrapper; removed after each request."""
    def __init__(self, donor, bank, policy, cache, scope, trace):
        super().__init__()
        self.donor=donor
        self.bank,self.policy,self.cache,self.scope,self.trace=bank,policy,cache,scope,trace
        self.cell_state=CellState(scope.request_id+scope.operator)
    def forward(self,x):
        def compute(rows):
            y,t=self.bank.run(rows,self.policy,state=self.cell_state,scope=self.cell_state.scope)
            self.trace.append(t)
            return y
        if self.cache is None:
            return compute(x)
        flat=x.reshape(-1,x.shape[-1])
        y=self.cache.run(flat,compute,self.scope).reshape(x.shape)
        self.trace.append({"exact_ffn_rows_reused":self.cache.last_reused_rows})
        return y


def _hidden(output):
    if isinstance(output,Tensor):
        return output
    if isinstance(output,tuple) and output and isinstance(output[0],Tensor):
        return output[0]
    raise TypeError("Unsupported decoder return type; no silent shape conversion")


def _replace_hidden(output, hidden):
    return hidden if isinstance(output,Tensor) else (hidden,*output[1:])


class FrozenExecutor:
    """Own one pretrained model. All altered routes use full-prefix, cache-free calls.

    Stateful KV is not reused across depths. Temporary hook installation is serialized
    and always undone. Repeating arbitrary frozen layers is experimental because the
    end-of-band representation was not pretrained as the band's input distribution.
    """
    def __init__(self,model:nn.Module,*,model_id:str,revision:str):
        if not model_id or not revision:
            raise ValueError("Model identity/revision required")
        self.model=model.eval()
        self.model_id,self.revision=model_id,revision
        for p in model.parameters():
            p.requires_grad_(False)
        self._lock=RLock()
        self._versions=[(p,p._version) for p in model.parameters()]
        self._banks={}
        self._row_caches={}
        self.last_trace={}
        # Decoder backbone only. Do not guess that vision or DeltaNet obeys this API.
        decoder=getattr(model,"model",None)
        if decoder is None or not hasattr(decoder,"layers"):
            raise TypeError("Expected a Qwen-style language decoder with model.layers")
        self.decoder=decoder
        family=getattr(getattr(model,"config",None),"model_type","")
        if family not in {"qwen3","qwen2","bedrock_test"}:
            raise TypeError("Frozen-band adapter currently supports Qwen2/Qwen3; hybrid donor needs its own cache proof")

    def reset_request(self):
        self._row_caches.clear()
        self.last_trace={}

    def unchanged(self):
        """Version-counter tripwire. Exact byte comparison is a separate test."""
        return all(p._version==v and not p.requires_grad for p,v in self._versions)

    def run(self,input_ids:Tensor,*,policy:FrozenPolicy=FrozenPolicy(),meter:Meter|None=None,
            request_id:str="ephemeral",fast:FastAssociations|None=None,**kwargs):
        meter=meter or Meter()
        if input_ids.ndim!=2 or input_ids.shape[-1]<1:
            raise ValueError("Nonempty batched token IDs required")
        if any(kwargs.get(k) is not None for k in ("past_key_values","past_key_value")) or kwargs.get("use_cache",False):
            raise ValueError("This correctness executor forbids external caches; use direct donor for cached production")
        with self._lock,torch.inference_mode():
            if not self.unchanged():
                raise RuntimeError("Frozen parameter tripwire failed")
            n=len(self.decoder.layers)
            start=policy.start if policy.start>=0 else n+policy.start
            end=policy.end if policy.end>=0 else n+policy.end
            if not 0<=start<=end<n:
                raise ValueError("Decoder band is outside model")
            meter.charge("model_calls")
            meter.charge("layer_calls",n)
            trace={"policy":asdict(policy),"donor_layer_calls":n,"extra_layer_calls":0,
                   "passes_executed":1,"halts_are_correctness_certificates":False,"cells":[],
                   "no_new_parameters":True,"request_id":request_id}
            if policy.neutral:
                result=self.model(input_ids=input_ids,use_cache=False,**kwargs)
                trace["neutral_direct_path"]=True
                self.last_trace=trace
                return result
            captures={}
            band_entry=None
            in_replay=False
            with ExitStack() as stack:
                if policy.cells.mode!="off" or policy.exact_ffn_cache:
                    layer=self.decoder.layers[end]
                    donor=layer.mlp
                    key=(end,policy.cells.width)
                    if key not in self._banks:
                        self._banks[key]=FrozenCellBank(donor,policy.cells.width)
                    cache_key=(request_id,end,stable_hash(asdict(policy.cells)))
                    cache=None
                    if policy.exact_ffn_cache:
                        cache=self._row_caches.setdefault(cache_key,ExactDeltaCache())
                    scope=CacheScope(self.revision,0,request_id,str(next(self.model.parameters()).dtype),
                                     f"FFN:{end}:{cache_key[-1]}")
                    layer.mlp=_CellCall(donor,self._banks[key],policy.cells,cache,scope,trace["cells"])
                    stack.callback(setattr,layer,"mlp",donor)
                def capture(index):
                    def hook(module,args,kw):
                        nonlocal band_entry
                        if in_replay:
                            return
                        if len(args)>1:
                            raise TypeError("Decoder positional context not supported; use named masks/positions")
                        if index == start:
                            band_entry=(args[0] if args else kw["hidden_states"]).detach()
                        clean=dict(kw)
                        clean.pop("hidden_states",None)
                        for field in ("past_key_values","past_key_value"):
                            if clean.get(field) is not None:
                                raise RuntimeError("Unexpected cache in frozen recurrence")
                        captures[index]=clean
                    return hook
                for index in range(start,end+1):
                    handle=self.decoder.layers[index].register_forward_pre_hook(capture(index),with_kwargs=True)
                    stack.callback(handle.remove)
                def loop(module,args,kw,output):
                    nonlocal in_replay
                    if in_replay:
                        return output
                    anchor=_hidden(output)
                    current=anchor
                    if fast is not None and policy.fast_gain:
                        current=fast.apply(current,scope=(request_id,self.revision,end),
                                           radius=policy.relative_radius*policy.fast_gain)
                    active=torch.ones(current.shape[:-1],dtype=torch.bool,device=current.device)
                    stable=torch.zeros_like(active,dtype=torch.long)
                    steps=torch.ones_like(stable)
                    delta_trace=[]
                    in_replay=True
                    try:
                        for _ in range(1,policy.passes):
                            if policy.gain==0 or not bool(active.any()):
                                break
                            meter.charge("layer_calls",end-start+1)
                            previous=current
                            transformed=(band_entry+policy.gain*(current-band_entry)
                                         if policy.feedback=="anchored_difference" else current)
                            for index in range(start,end+1):
                                transformed=_hidden(self.decoder.layers[index](transformed,**captures[index]))
                            proposed=(anchor+policy.gain*(transformed-anchor)
                                      if policy.feedback=="anchored_difference" else
                                      current+policy.gain*(transformed-current))
                            proposed=project_delta(anchor,proposed.float()-anchor.float(),policy.relative_radius)
                            delta=(proposed.float()-previous.float()).norm(dim=-1)/previous.float().norm(dim=-1).clamp_min(1e-12)
                            # Halting decisions are PER POSITION. A sequence-wide decision
                            # would let future tokens leak into earlier likelihood scores.
                            current=torch.where(active[...,None],proposed,current)
                            steps=steps+active.long()
                            stable=torch.where(delta<=policy.halt_delta,stable+1,torch.zeros_like(stable))
                            if policy.halt_delta>0:
                                active=active & (stable<policy.halt_patience)
                            delta_trace.append(float(delta.max()))
                            trace["extra_layer_calls"]+=end-start+1
                            trace["passes_executed"]+=1
                    finally:
                        in_replay=False
                    trace["position_depths"]=steps.detach().cpu().tolist()
                    trace["max_relative_step_deltas"]=delta_trace
                    return _replace_hidden(output,current)
                handle=self.decoder.layers[end].register_forward_hook(loop,with_kwargs=True)
                stack.callback(handle.remove)
                result=self.model(input_ids=input_ids,use_cache=False,**kwargs)
            if not self.unchanged():
                raise RuntimeError("Donor changed during frozen execution")
            self.last_trace=trace
            return result

    def generate(self,input_ids:Tensor,*,policy:FrozenPolicy=FrozenPolicy(),meter:Meter|None=None,
                 max_new_tokens:int=32,eos_token_id:int|None=None,request_id:str="ephemeral",
                 fast:FastAssociations|None=None,forced_first:int|None=None):
        if input_ids.shape[0]!=1 or max_new_tokens<1:
            raise ValueError("Reference generation supports one prompt and a positive token limit")
        meter=meter or Meter()
        ids=input_ids.clone()
        traces=[]
        for step in range(max_new_tokens):
            output=self.run(ids,policy=policy,meter=meter,request_id=request_id,fast=fast)
            token=int(output.logits[0,-1].argmax()) if step or forced_first is None else forced_first
            if not 0<=token<output.logits.shape[-1]:
                raise ValueError("Invalid forced branch token")
            meter.charge("generated_tokens")
            ids=torch.cat((ids,ids.new_tensor([[token]])),dim=-1)
            traces.append(self.last_trace)
            if token==eos_token_id:
                break
        return ids,traces

    def brainstorm(self,input_ids:Tensor,*,policies:tuple[FrozenPolicy,...],meter:Meter,
                   verifier:Callable[[list[int]],Outcome]|None=None,max_new_tokens:int=8,
                   request_id:str="ephemeral"):
        """Sequential latent-route variants and token branches in ONE parameter owner.

        The same donor coda/head decodes each route. Only host verification may promote
        a candidate over the donor answer. Entropy/consensus alone is not authority.
        """
        if not policies or not policies[0].neutral:
            raise ValueError("First branch must be the untouched baseline for fallback")
        branches=[]
        unique={}
        for index,policy in enumerate(policies):
            meter.charge("branches")
            branch_id=f"{request_id}:branch:{index}"
            ids,traces=self.generate(input_ids,policy=policy,meter=meter,
                                     max_new_tokens=max_new_tokens,request_id=branch_id)
            tokens=ids[0,input_ids.shape[-1]:].tolist()
            key=stable_hash(tokens)
            if key in unique:
                branches[unique[key]]["equivalent_routes"].append(index)
                continue
            receipt=verifier(tokens) if verifier else None
            if receipt and not receipt.binds(tokens):
                raise ValueError("Verifier receipt is bound to different output")
            branches.append({"tokens":tokens,"policy":asdict(policy),"traces":traces,
                "equivalent_routes":[index],"verification":asdict(receipt) if receipt else None})
            unique[key]=len(branches)-1
        winner=next((i for i,b in enumerate(branches) if b["verification"] and
                     b["verification"]["passed"] is True and b["verification"]["independent"]),0)
        return {"branches":branches,"selected":winner,
                "selection":"host_verified" if branches[winner]["verification"] and
                  branches[winner]["verification"]["passed"] is True and
                  branches[winner]["verification"]["independent"] else "donor_fallback",
                "learned_semantic_slots":False,"one_parameter_owner":True}

    def speculative(self,input_ids:Tensor,*,draft:FrozenPolicy,target:FrozenPolicy,meter:Meter,
                    max_new_tokens:int=16,block:int=3,sampled:bool=False,
                    generator:torch.Generator|None=None,request_id:str="speculation",eos_token_id:int|None=None):
        """Same frozen weights draft/verify; full-prefix reference, no speed claim.

        Corrected rejection sampling preserves the target's raw T=1 distribution.
        Greedy mode preserves target greedy decoding. No cache can survive a rejected
        branch because this reference never keeps a KV cache at all.
        """
        if block<1 or input_ids.shape[0]!=1 or max_new_tokens<1:
            raise ValueError("Invalid speculative budgets")
        prefix=input_ids.clone()
        accepted=proposed=0
        while prefix.shape[1]-input_ids.shape[1]<max_new_tokens:
            k=min(block,max_new_tokens-(prefix.shape[1]-input_ids.shape[1]))
            trial=prefix
            qs=[]; tokens=[]
            for _ in range(k):
                logits=self.run(trial,policy=draft,meter=meter,request_id=request_id).logits[0,-1].float()
                q=logits.softmax(-1)
                token=int(torch.multinomial(q,1,generator=generator)) if sampled else int(logits.argmax())
                tokens.append(token); qs.append(q)
                trial=torch.cat((trial,trial.new_tensor([[token]])),1)
                if token==eos_token_id:
                    break
            logits=self.run(trial,policy=target,meter=meter,request_id=request_id).logits[0,prefix.shape[1]-1:].float()
            decision=(verify_sampled(prefix.new_tensor(tokens),torch.stack(qs),logits.softmax(-1),
                generator=generator,eos_token_id=eos_token_id) if sampled else
                verify_greedy(prefix.new_tensor(tokens),logits,eos_token_id=eos_token_id))
            room=max_new_tokens-(prefix.shape[1]-input_ids.shape[1])
            emitted=list(decision.tokens)[:room]
            if eos_token_id in emitted:
                emitted=emitted[:emitted.index(eos_token_id)+1]
            meter.charge("generated_tokens",len(emitted))
            prefix=torch.cat((prefix,prefix.new_tensor([emitted])),1)
            accepted+=decision.accepted; proposed+=len(tokens)
            if emitted[-1]==eos_token_id:
                break
        return prefix,{"accepted":accepted,"proposed":proposed,"speedup_claim":False,
                       "algorithm":"same_frozen_weights_full_prefix"}
