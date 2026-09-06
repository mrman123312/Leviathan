"""Dual-form, evidence-scoped skill memory on the existing MemoryEcology.

A demonstration-consistent program is an episodic proposal, NOT a verified theorem.
Compiled macros expand to the same AST exactly; generality is separately evaluated.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
from copy import deepcopy
from ..memory_ecology import MemoryEcology, MemoryKind, MemoryRecord, MemoryStatus
from ..types import Provenance, ProvenanceKind
from ..bedrock.contracts import Outcome
from .contracts import digest
from .programs import Program
from .grid import shape

class SkillLibrary:
    def __init__(self,journal=None,*,model_revision='symbolic',memory=None):
        self.memory=memory if memory is not None else MemoryEcology(journal)
        self.revision=model_revision
    def remember_candidate(self,task,program:Program,*,neural_state=None,dependencies=(),intervention_evidence=None):
        if not all(program.run(e.input)==e.output for e in task.examples):
            raise ValueError('Skill candidate contradicts its demonstration evidence')
        payload={'kind':'strength_skill','program':program.as_dict(),'dsl_version':1,
            'model_revision':self.revision,'support_hash':task.support_hash,'scope':task.id,
            'split':task.split,'preconditions':{'max_dimension':30,'colors':list(range(10))},
            'shape_relations':[(shape(e.input),shape(e.output)) for e in task.examples],
            'neural_state':deepcopy(neural_state),'dependencies':list(dependencies),
            'intervention_evidence':deepcopy(intervention_evidence),
            'status':'support_consistent_only','heldout_verification':None}
        sid='strength-candidate-'+digest([task.support_hash,program.as_dict(),self.revision])[:32]
        self.memory.write(MemoryRecord(sid,MemoryKind.EPISODIC,payload,0.,
            Provenance(ProvenanceKind.SELF_INFERENCE,'strength',0.,self.revision),
            evidence_refs=(task.support_hash,),tags=('strength',task.id),verified=False))
        return sid
    def retrieve(self,task,limit=32):
        # Default inference never writes cross-task evaluation answers into this store.
        candidates=[]
        for record in self.memory.records:
            if record.status!=MemoryStatus.ACTIVE:continue
            p=record.payload
            if not isinstance(p,dict) or p.get('kind')!='strength_skill' or p.get('dsl_version')!=1:continue
            if p.get('model_revision')!=self.revision:continue
            try:
                program=Program.from_dict(p['program'])
                # Fit on CURRENT demonstrations before memory can influence output.
                if all(program.run(e.input)==e.output for e in task.examples):candidates.append(program)
            except ValueError:continue
        return tuple(sorted(set(candidates),key=lambda x:(x.size,str(x)))[:limit])
    def compatible_neural_states(self,task):
        states=[]
        for r in self.memory.records:
            p=r.payload
            if r.status!=MemoryStatus.ACTIVE or not isinstance(p,dict) or p.get('kind')!='strength_skill':continue
            if p.get('model_revision')!=self.revision or not p.get('neural_state'):continue
            # Different support evidence is needed for a transfer trial. A memory
            # of these very demonstrations cannot pose as held-out validation.
            if p.get('support_hash')==task.support_hash:continue
            try:
                program=Program.from_dict(p['program'])
                if all(program.run(e.input)==e.output for e in task.examples):states.append(deepcopy(p['neural_state']))
            except ValueError:continue
        return states[:4]

    def compatible_interventions(self,task):
        proposals=[]
        for r in self.memory.records:
            p=r.payload
            if r.status!=MemoryStatus.ACTIVE or not isinstance(p,dict) or p.get('kind')!='strength_skill':continue
            if p.get('model_revision')!=self.revision or p.get('support_hash')==task.support_hash:continue
            try:
                program=Program.from_dict(p['program'])
                if not all(program.run(e.input)==e.output for e in task.examples):continue
            except ValueError:continue
            evidence=p.get('intervention_evidence') or {}
            for trial in evidence.get('trials',[]):
                route=trial.get('route',{})
                if trial.get('accepted') and route.get('kind')=='cell_ablation' and route not in proposals:
                    proposals.append(deepcopy(route))
        return proposals[:8]

    def promote(self,identifier,receipt:Outcome):
        record=next(r for r in self.memory.records if r.id==identifier)
        if receipt.passed is not True or not receipt.independent or not receipt.binds(record.payload):
            raise ValueError('Promotion needs independent evidence bound to this exact skill record')
        new_id='strength-verified-'+digest([identifier,receipt.evidence_id])[:32]
        self.memory.promote_episode(identifier,new_id=new_id,destination=MemoryKind.PROCEDURAL,
            verification_ref=receipt.evidence_id,verification_confidence=.8,independence_score=1.)
        return new_id
    def invalidate(self,identifier,*,evidence_id):
        if not evidence_id:raise ValueError('Counterevidence reference required')
        records=self.memory.records
        if not any(r.id==identifier for r in records):raise KeyError(identifier)
        affected={identifier};change=True
        while change:
            change=False
            for r in records:
                deps=r.payload.get('dependencies',[]) if isinstance(r.payload,dict) else []
                if r.id not in affected and (affected.intersection(deps) or affected.intersection(r.source_refs)):
                    affected.add(r.id);change=True
        for rid in affected:self.memory.deprecate(rid,reason_ref=evidence_id)
        return tuple(sorted(affected))
    def macros(self,min_independent_tasks=2):
        hits=defaultdict(set);programs={}
        for record in self.memory.records:
            p=record.payload
            if record.status!=MemoryStatus.ACTIVE or not isinstance(p,dict) or p.get('kind')!='strength_skill':continue
            if p.get('model_revision')!=self.revision:continue
            pending=[Program.from_dict(p['program'])];seen=set()
            while pending:
                cur=pending.pop()
                if cur.id in seen:continue
                seen.add(cur.id)
                if cur.size>=2:hits[cur.id].add(p['support_hash']);programs[cur.id]=cur
                pending.extend(q for q in (cur.child,cur.other) if q is not None)
        return tuple(programs[k] for k in sorted(hits) if len(hits[k])>=min_independent_tasks)
    @staticmethod
    def expand(macro:Program,argument:Program):
        # Syntactic substitution, not an approximation or learned compilation head.
        return macro.replace_input(argument)
