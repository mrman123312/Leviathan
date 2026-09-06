"""Frozen neural research operators with demonstration-only activation selection.

Task-state contrasts are proposed interventions, not a replication of Function
Vectors. Cell ablations estimate conditional intervention effects, not semantic
labels for neurons. Every operation uses the single existing model owner.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from torch.nn import functional as F
from ..bedrock.neural import _hidden, _replace_hidden
from .contracts import digest


def bounded(anchor:Tensor,delta:Tensor,radius:float)->Tensor:
    if not 0<=radius<=1 or not math.isfinite(radius):raise ValueError('Invalid activation radius')
    if not torch.isfinite(anchor).all() or not torch.isfinite(delta).all():raise FloatingPointError('Nonfinite activation')
    a=anchor.float();d=delta.float();limit=radius*a.norm(dim=-1,keepdim=True)
    d=d*(limit/d.norm(dim=-1,keepdim=True).clamp_min(1e-20)).clamp(max=1)
    result=(a+d).to(anchor.dtype)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite bounded result')
    return result

@dataclass(frozen=True)
class NeuralRoute:
    kind: str='donor'
    layer: int=12
    passes: int=2
    anchor_mix: float=.5
    cell_start: int=0
    cell_width: int=128
    radius: float=.02
    def __post_init__(self):
        if self.kind not in ('donor','task_state','damped_band','cell_ablation'):raise ValueError('Unknown route')
        if self.layer<0 or not 1<=self.passes<=8 or not 0<=self.anchor_mix<=1 or not 0<=self.radius<=1:
            raise ValueError('Invalid neural route')
        if self.cell_start<0 or self.cell_width<1:raise ValueError('Invalid cell slice')

class TaskWorkspace:
    """Task-local multi-slot activation state, bound to layer, revision and support."""
    def __init__(self,revision:str,layer:int,support_hash:str,vectors:Tensor):
        if vectors.ndim!=2 or not 1<=len(vectors)<=8 or not torch.isfinite(vectors).all():raise ValueError('Invalid slots')
        self.revision,self.layer,self.support_hash=revision,layer,support_hash
        self.vectors=vectors.detach().float().clone()
    def metadata(self):
        return {'revision':self.revision,'layer':self.layer,'support_hash':self.support_hash,
                'slots':len(self.vectors),'norms':self.vectors.norm(dim=-1).cpu().tolist(),
                'learned_by_sgd':False,'semantics_verified':False}
    def serialize(self):return {**self.metadata(),'vectors':self.vectors.cpu().tolist()}
    def direction(self,index):
        if not 0<=index<len(self.vectors):raise ValueError('Slot outside workspace')
        return self.vectors[index]

class NeuralFabric:
    def __init__(self,executor):
        self.executor=executor;self.model=executor.model;self.layers=executor.decoder.layers
        self.last_trace={};self.forward_calls=0
    @contextmanager
    def use(self,route:NeuralRoute,*,workspace:TaskWorkspace|None=None,slot=0):
        trace={'kind':route.kind,'extra_layer_calls':0,'nonfinite_fallbacks':0,'cell_intervention':False}
        self.last_trace=trace
        if route.kind=='donor':yield;return
        if route.layer>=len(self.layers):raise ValueError('Intervention outside layer range')
        with self.executor._lock,ExitStack() as stack:
            if route.kind=='task_state':
                if workspace is None or (workspace.revision,workspace.layer)!=(self.executor.revision,route.layer):
                    raise ValueError('Workspace revision/layer mismatch')
                vector=workspace.direction(slot)
                def inject(module,args,output):
                    h=_hidden(output)
                    if h.shape[-1]!=vector.numel():raise ValueError('Task state hidden size mismatch')
                    # Apply to every query-prefix position under one fixed policy.
                    # Values depend only on visible support, never query suffix labels.
                    v=F.normalize(vector.to(h.device),dim=-1)*h.float().norm(dim=-1,keepdim=True)*route.radius
                    return _replace_hidden(output,bounded(h,v,route.radius))
                handle=self.layers[route.layer].register_forward_hook(inject);stack.callback(handle.remove)
            elif route.kind=='cell_ablation':
                down=self.layers[route.layer].mlp.down_proj
                width=getattr(down,'in_features',None)
                if width is None or route.cell_start+route.cell_width>width:raise ValueError('Cell slice outside FFN')
                def mask(module,args):
                    z=args[0].clone();z[...,route.cell_start:route.cell_start+route.cell_width]=0
                    return (z,*args[1:])
                handle=down.register_forward_pre_hook(mask);stack.callback(handle.remove)
                trace['cell_intervention']=True
            elif route.kind=='damped_band':
                end=min(route.layer+3,len(self.layers)-1);captures={};entry=None;replaying=False
                if route.passes==1 or route.anchor_mix==1:
                    yield;return
                def capture(index):
                    def cb(module,args,kw):
                        nonlocal entry
                        if replaying:return
                        if len(args)>1:raise ValueError('Use named layer context')
                        if index==route.layer:entry=args[0] if args else kw['hidden_states']
                        ctx=dict(kw);ctx.pop('hidden_states',None)
                        if any(ctx.get(k) is not None for k in ('past_key_values','past_key_value')):
                            raise ValueError('Damped band uses cache-free forwards only')
                        captures[index]=ctx
                    return cb
                for index in range(route.layer,end+1):
                    handle=self.layers[index].register_forward_pre_hook(capture(index),with_kwargs=True);stack.callback(handle.remove)
                def loop(module,args,kw,output):
                    nonlocal replaying
                    if replaying:return output
                    donor=_hidden(output);k=route.passes;replaying=True
                    try:
                        # Damped residual integration, not clipped e->a transport.
                        z=(entry.float()+(donor.float()-entry.float())/k).to(entry.dtype)
                        for _ in range(1,k):
                            f=z
                            for index in range(route.layer,end+1):
                                f=_hidden(self.layers[index](f,**captures[index]));trace['extra_layer_calls']+=1
                                if not torch.isfinite(f).all():raise FloatingPointError('Nonfinite damped replay')
                            z=(z.float()+(f.float()-z.float())/k).to(z.dtype)
                        proposed=route.anchor_mix*donor.float()+(1-route.anchor_mix)*z.float()
                        h=bounded(donor,proposed-donor.float(),min(1.,max(.1,route.radius)))
                        return _replace_hidden(output,h)
                    except FloatingPointError:
                        trace['nonfinite_fallbacks']+=1;return output
                    finally:replaying=False
                handle=self.layers[end].register_forward_hook(loop,with_kwargs=True);stack.callback(handle.remove)
            yield
        if not self.executor.unchanged():raise RuntimeError('Frozen donor mutation tripwire')
    def capture(self,ids,layer):
        out={}
        def hook(module,args,output):out['v']=_hidden(output)[0,-1].detach().float().clone()
        with self.executor._lock,torch.inference_mode():
            handle=self.layers[layer].register_forward_hook(hook)
            try:self.model(input_ids=ids,use_cache=False);self.forward_calls+=1
            finally:handle.remove()
        return out['v']
    def contrast_workspace(self,positive_ids,control_ids,*,layer,support_hash):
        if len(positive_ids)!=len(control_ids) or not positive_ids:raise ValueError('Aligned support/control probes required')
        vectors=[]
        for positive,control in zip(positive_ids[:4],control_ids[:4]):
            vectors.append(self.capture(positive,layer)-self.capture(control,layer))
        return TaskWorkspace(self.executor.revision,layer,support_hash,torch.stack(vectors))
    def nll(self,ids,answer_start,route,workspace=None,slot=0):
        if not 1<=answer_start<ids.shape[-1]:raise ValueError('Aligned nonempty answer tokens required')
        with torch.inference_mode(),self.use(route,workspace=workspace,slot=slot):
            out=self.model(input_ids=ids,use_cache=False).logits[0,answer_start-1:-1].float();self.forward_calls+=1
            return float(F.cross_entropy(out,ids[0,answer_start:]))
    def select_on_demonstrations(self,validation,*,workspace=None,prior_routes=()):
        """Two or more support folds; each proposed route must improve EVERY fold.

        These folds are not a held-out benchmark. No guarantee of better task accuracy.
        A supplied workspace must have been extracted without those fold answers.
        Caller records support/control construction. No inference-time gradients.
        """
        if len(validation)<2:return NeuralRoute(),{'status':'insufficient_support','accepted':False}
        layer=max(0,len(self.layers)//2-2)
        routes=[(NeuralRoute('damped_band',layer=layer,radius=.25),0)]
        if workspace is not None:
            routes.extend((NeuralRoute('task_state',layer=workspace.layer,radius=.02),i)
                          for i in range(len(workspace.vectors)))
        for start in (0,128,256):
            if start+128<=self.layers[layer].mlp.down_proj.in_features:
                routes.append((NeuralRoute('cell_ablation',layer=layer,cell_start=start),0))
        # Prior coalition effects are proposals only. Retest under CURRENT support.
        for raw in prior_routes[:8]:
            route=NeuralRoute(**raw)
            if route.kind=='cell_ablation' and (route,0) not in routes:routes.append((route,0))
        baseline=[self.nll(ids,pos,NeuralRoute()) for ids,pos in validation]
        trials=[];winner=NeuralRoute();best=sum(baseline);selected_slot=0
        for route,slot in routes:
            try:
                losses=[self.nll(ids,pos,route,workspace,slot) for ids,pos in validation]
                ok=all(math.isfinite(b) and b<a-1e-3 for a,b in zip(baseline,losses))
                trials.append({'route':route.__dict__,'slot':slot,'losses':losses,'accepted':ok,
                               'claim':'conditional_support_NLL_effect_not_semantic_function'})
                if ok and sum(losses)<best:winner=route;best=sum(losses);selected_slot=slot
            except (ValueError,FloatingPointError) as e:
                trials.append({'route':route.__dict__,'slot':slot,'accepted':False,'error':str(e)})
        return winner,{'status':'support_gate','baseline_losses':baseline,'trials':trials,
                       'selected':winner.__dict__,'selected_slot':selected_slot,
                       'selected_validation_loss':best,'accepted':winner.kind!='donor',
                       'benchmark_labels_used':False,'accuracy_gain_proven':False}
