"""Counterexample-compiled, representation-diverse program search.

No optimizer or neural head. No query output is accepted by this API. Wrong complete
programs are deduplicated, NOT forbidden as partial computations of longer programs.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import math
import ast
from typing import Callable, Iterable
import numpy as np
from .contracts import ArcTask, Example, Grid, SearchConfig, SearchMeter, Witness, digest
from .programs import Program, apply, parse, vocabulary, infer_color_map, apply_binary, constant_hole_repairs
from .grid import shape, summarize


def mismatch(a: Grid, b: Grid) -> float:
    if shape(a)!=shape(b):
        h,w=shape(a);r,c=shape(b)
        return 1.+(abs(h-r)+abs(w-c))/(h+w+r+c)
    return sum(x!=y for r,s in zip(a,b) for x,y in zip(r,s))/(len(a)*len(a[0]))


def description_cost(p):
    # A full 10-color table costs more than one rotation; no learned semantic claim.
    if p.op=='x':return 0.
    literals=sum(len(x) if isinstance(x,tuple) else 1 for x in p.params)
    if p.op=='python_grid':
        # Compare actual program complexity, not one opaque source-string token.
        literals=sum(1 for _ in ast.walk(ast.parse(p.params[0])))
    return 1.+.12*literals+description_cost(p.child)+(description_cost(p.other) if p.other else 0.)

@dataclass
class Candidate:
    program: Program
    outputs: tuple[Grid,...]
    loss: float
    view: str
    source: str
    witness: Witness | None
    @property
    def rank(self):return (self.loss,description_cost(self.program),str(self.program))

class Counterexamples:
    def __init__(self):self.rejected={};self.seen=set()
    def register(self,p,outputs,examples,error=None):
        self.seen.add(p.id)
        witness=None
        if error:
            witness=Witness(p.id,0,shape(examples[0].output),None,error=str(error)[:240])
        else:
            for i,(actual,ex) in enumerate(zip(outputs,examples)):
                if actual==ex.output:continue
                wrong=None
                if shape(actual)==shape(ex.output):
                    for r,(a,b) in enumerate(zip(actual,ex.output)):
                        for c,(x,y) in enumerate(zip(a,b)):
                            if x!=y:wrong=(r,c,x,y);break
                        if wrong is not None:break
                witness=Witness(p.id,i,shape(ex.output),shape(actual),wrong)
                break
        if witness is not None:self.rejected[p.id]=witness
        return witness

class ProgressController:
    """No-progress contract; method selection changes only from observable progress.

    Evidence is a new support-consistent candidate, lower support error, or a new
    predictive behavior. Posterior counts are explicitly not calibrated competence.
    """
    def __init__(self):self.events=[];self.used_states=set();self.best=math.inf
    def allow(self,method,state):
        key=(method,state)
        if key in self.used_states:return False
        self.used_states.add(key);return True
    def record(self,method,*,before,after,new_behaviors,consistent):
        event={'method':method,'loss_before':None if math.isinf(before) else before,
               'loss_after':None if math.isinf(after) else after,
               'new_behaviors':new_behaviors,'consistent_candidates':consistent,
               'progress':after<before or new_behaviors>0}
        self.events.append(event);self.best=min(self.best,after)
    def next_view(self,index):return ('rows','objects','differences')[index%3]

class CompiledGuidance:
    """Proposal-prefix guidance with an unguided beam reserve.

    This borrows the general program-guided-synthesis principle. It is not an
    implementation or replication of Narcissus. Positive priority never certifies truth.
    """
    def __init__(self):self.transitions=Counter();self.fragments={}
    def add(self,p):
        pending=[p];seen=set()
        while pending:
            cur=pending.pop()
            if cur.id in seen:continue
            seen.add(cur.id)
            if cur.op=='x':continue
            self.transitions[(cur.child.op,cur.op)]+=1
            if cur.other:self.transitions[(cur.other.op,cur.op)]+=1
            if cur.size>=2:self.fragments[cur.id]=cur
            pending.extend(x for x in (cur.child,cur.other) if x is not None)
    def priority(self,p):return self.transitions[(p.child.op,p.op)] if p.child else 0

class StrengthSearch:
    def __init__(self,config:SearchConfig=SearchConfig(),proposer=None,library=None,
                 progress:Callable[[str],None]|None=None):
        self.config=config;self.proposer=proposer;self.library=library
        self.progress=progress or (lambda s:None)
    def solve(self,task:ArcTask):
        if not isinstance(task,ArcTask):raise TypeError('Use the label-free ArcTask boundary')
        meter=SearchMeter(self.config);ledger=Counterexamples();control=ProgressController();guide=CompiledGuidance()
        inputs=tuple(e.input for e in task.examples)+task.queries
        targets=tuple(e.output for e in task.examples);n=len(targets)
        outputs_cache={};behaviors={};solutions={};all_candidates={};neural_records=[]
        primitive=vocabulary(task.examples)
        # Goal-conditioned junctions: invert a known bijection on all visible
        # targets, then meet a forward prefix there. This never queries hidden
        # outcomes and does not need a pixel-distance heuristic to be informative.
        inverse={'rot90':'rot270','rot270':'rot90','rot180':'rot180',
                 'flip_lr':'flip_lr','flip_ud':'flip_ud','transpose':'transpose'}
        goal_junctions={}
        for suffix,inv in inverse.items():
            key=digest(tuple(apply(inv,g) for g in targets))
            goal_junctions.setdefault(key,[]).append(suffix)
        baseline_program=Program()
        def evaluate(p,view='geometry',source='search',known=None):
            if p.id in ledger.seen:
                meter.duplicate_programs+=1;return all_candidates.get(p.id)
            if not meter.available():return None
            meter.candidates+=1
            try:
                outputs=known or tuple(p.run(x) for x in inputs)
                witness=ledger.register(p,outputs,task.examples)
                loss=sum(mismatch(x,y) for x,y in zip(outputs,targets))/n
                item=Candidate(p,outputs,loss,source=source,view=view,witness=witness)
            except (ValueError,IndexError,OverflowError,ZeroDivisionError) as e:
                ledger.register(p,(),task.examples,e);return None
            all_candidates[p.id]=item;outputs_cache[p.id]=outputs
            # Include QUERY INPUT behavior so distinct held-out predictions survive.
            # Query outputs are absent. This dedup is not global functional equivalence.
            sig=digest(outputs)
            if sig in behaviors:
                meter.duplicate_behaviors+=1
                if description_cost(p)>=description_cost(behaviors[sig].program):
                    if loss==0 and len(solutions)<self.config.max_solutions:solutions[p.id]=item
                    return item
            behaviors[sig]=item
            if loss==0:
                solutions[p.id]=item
                if len(solutions)>self.config.max_solutions:
                    worst=max(solutions,key=lambda k:description_cost(solutions[k].program))
                    solutions.pop(worst)
            if p.size < self.config.max_depth:
                for suffix in goal_junctions.get(digest(outputs[:n]),()):
                    joined=p.then(suffix)
                    if joined.id not in ledger.seen:
                        evaluate(joined,'goal_junction','inverse_goal_join')
            return item
        root=evaluate(baseline_program)
        if self.library:
            for p in self.library.retrieve(task):
                evaluate(p,'memory','memory');guide.add(p)
        def color_repair(item):
            if item is None or not meter.available() or item.program.size>=16:return
            mapping=infer_color_map(item.outputs[:n],targets)
            if mapping:
                p=item.program.then('color_map',(mapping,))
                evaluate(p,'color','support_inferred_color_map')
        color_repair(root)
        def propose(round_index):
            if self.proposer is None or round_index>=self.config.neural_rounds or not meter.available():return []
            best=sorted(all_candidates.values(),key=lambda x:x.rank)[:4]
            rejected=[{'program':str(c.program),'witness':c.witness.as_dict()} for c in best if c.witness]
            for rec in neural_records[-6:]:
                if rec.get('witness') and rec['text'] not in {x['program'] for x in rejected}:
                    rejected.append({'program':rec['text'],'witness':rec['witness']})
            state=digest([task.support_hash,[r['program'] for r in rejected],control.next_view(round_index)])
            if not control.allow('neural',state):return []
            before=min((c.loss for c in best),default=math.inf);old=len(behaviors)
            self.progress(f'  Neural representation: {control.next_view(round_index)}; revise from concrete counterexamples')
            proposals=self.proposer.propose(task,rejected=rejected,view=control.next_view(round_index),
                                            count=self.config.proposals_per_round,round_index=round_index)
            meter.neural_calls+=getattr(self.proposer,'last_calls',len(proposals))
            added=[]
            for raw in proposals:
                text=raw['text'] if isinstance(raw,dict) else str(raw)
                rec={'round':round_index,'text':text,'source':'frozen_qwen' if getattr(self.proposer,'is_neural',False) else 'injected_control'}
                try:
                    p=parse(text);guide.add(p)
                    duplicate=p.id in ledger.seen
                    item=evaluate(p,'neural',rec['source']);color_repair(item)
                    rec.update(valid_syntax=True,duplicate=duplicate,support_consistent=item is not None and item.loss==0,
                               loss=None if item is None else item.loss,
                               witness=None if item is None or item.witness is None else item.witness.as_dict())
                    if item:added.append(item)
                    if item is None or item.loss>0:
                        for repaired in constant_hole_repairs(p,task.examples,limit=min(64,self.config.repair_budget)):
                            if not meter.available():break
                            fixed=evaluate(repaired,'neural','counterexample_literal_repair');color_repair(fixed)
                            if fixed:added.append(fixed)
                    # Repair a complete rejected proposal without freezing its mistake.
                    for op,params,view in primitive[:self.config.repair_budget]:
                        if not meter.available() or p.size>=15:break
                        repaired=evaluate(p.then(op,params),view,'proposal_repair');color_repair(repaired)
                    # Every proper subexpression is valid evidence about proposal structure,
                    # but it is not presumed correct. Make it available to normal search.
                    pending=[q for q in (p.child,p.other) if q is not None];seen_fragments=set()
                    while pending:
                        cur=pending.pop()
                        if cur.id in seen_fragments:continue
                        seen_fragments.add(cur.id)
                        fragment=evaluate(cur,'neural','proposal_fragment');color_repair(fragment)
                        if fragment:added.append(fragment)
                        pending.extend(q for q in (cur.child,cur.other) if q is not None)
                except (SyntaxError,ValueError,RecursionError) as exc:
                    rec.update(valid_syntax=False,error=str(exc)[:240])
                neural_records.append(rec)
            after=min((c.loss for c in all_candidates.values()),default=math.inf)
            control.record('neural',before=before,after=after,new_behaviors=len(behaviors)-old,consistent=len(solutions))
            return added
        # Gather both neural and generic seed programs. Wrong proposals never replace the control grammar.
        added=propose(0)
        frontier=[root]+added
        macros=self.library.macros() if self.library else ()
        for depth in range(1,self.config.max_depth+1):
            if not meter.available():break
            before=min(c.loss for c in all_candidates.values());old=len(behaviors)
            self.progress(f'  Search depth {depth}; {meter.candidates} programs; {len(solutions)} exact support fits')
            next_items=[]
            for parent in frontier:
                if parent is None:continue
                for macro in macros:
                    if not meter.available() or parent.program.size+macro.size>16:break
                    p=macro.replace_input(parent.program)
                    item=evaluate(p,'memory','compiled_macro');color_repair(item)
                    if item:next_items.append(item)
                ordered=sorted(primitive,key=lambda op:-guide.transitions[(parent.program.op,op[0])])
                for op,params,view in ordered:
                    if not meter.available():break
                    p=parent.program.then(op,params)
                    # Apply one operation to already computed grids. No reexecution of prefix.
                    try:known=tuple(apply(op,g,params) for g in parent.outputs)
                    except (ValueError,IndexError,OverflowError,ZeroDivisionError):continue
                    item=evaluate(p,view,'composition',known);color_repair(item)
                    if item:next_items.append(item)
            # Compose distinct representation branches as two operands. General
            # mask-pattern/object-canvas joins, not a memorized task catalogue.
            join_pool=sorted(behaviors.values(),key=lambda x:x.rank)[:12]
            color_set=sorted({v for g in targets for row in g for v in row})
            for left in join_pool:
                if not meter.available():break
                for right in join_pool:
                    if not meter.available():break
                    if left.program.size+right.program.size+1>self.config.max_depth+1:continue
                    operators=[('merge',(0,)),('kronecker',(0,))]
                    for color in color_set:
                        operators += [('intersection',(color,0)),('xor_grids',(color,0)),('paint_mask',(color,0))]
                    for op,params in operators:
                        if not meter.available():break
                        program=Program(op,left.program,params,right.program)
                        try:known=tuple(apply_binary(op,a,b,params) for a,b in zip(left.outputs,right.outputs))
                        except ValueError:continue
                        item=evaluate(program,'relational','binary_composition',known);color_repair(item)
                        if item:next_items.append(item)
            after=min(c.loss for c in all_candidates.values())
            control.record('enumerate',before=before,after=after,new_behaviors=len(behaviors)-old,consistent=len(solutions))
            # Stratified beams stop a temporarily bad crop from erasing the object view.
            # Half of each view is ordered without neural guidance. Beam remains incomplete.
            pool=defaultdict(list)
            for c in next_items:
                if behaviors.get(digest(c.outputs)) is c:pool[c.view].append(c)
            chosen={}
            for view,items in pool.items():
                reserve=max(1,self.config.beam_per_view//2)
                plain=sorted(items,key=lambda x:x.rank)[:reserve]
                guided=sorted(items,key=lambda x:(x.loss,-guide.priority(x.program),description_cost(x.program),str(x.program)))[:reserve]
                for c in plain+guided:chosen[c.program.id]=c
            frontier=sorted(chosen.values(),key=lambda c:c.rank)
            if depth<self.config.neural_rounds:
                frontier+=propose(depth)
            if not frontier:break
        ranked=sorted(solutions.values(),key=lambda x:(description_cost(x.program),str(x.program)))
        # Distinct full-query predictions, not merely distinct syntax.
        query_groups={}
        for c in ranked:
            sig=digest(c.outputs[n:])
            query_groups.setdefault(sig,[]).append(c)
        selected=[v[0] for v in query_groups.values()][:2]
        guesses=[]
        for qi in range(len(task.queries)):
            unique=[]
            for c in selected:
                g=c.outputs[n+qi]
                if g not in unique:unique.append(g)
            guesses.append(unique)
        best=sorted(all_candidates.values(),key=lambda c:c.rank)[:8]
        report={'task_id':task.id,'split':task.split,'status':'support_consistent_candidates' if selected else 'abstained',
            'attempts':guesses,'support_solutions':len(solutions),'distinct_query_behaviors':len(query_groups),
            'selected_programs':[str(c.program) for c in selected],
            'selected_sources':[c.source for c in selected],
            'programs':[c.program.as_dict() for c in selected],
            'best_partial':[{'program':str(c.program),'loss':c.loss,'witness':None if c.witness is None else c.witness.as_dict()} for c in best],
            'neural_proposals':neural_records,'controller_events':control.events,
            'budget':meter.snapshot(),'config':asdict(self.config),'policy_sha256':self.config.fingerprint,
            'support_hash':task.support_hash,'query_labels_received':False,'training_steps':0,
            'neural_model_used':bool(getattr(self.proposer,'is_neural',False)),
            'support_consistency_is_test_success':False,'search_complete':False,
            'selection_rule':'exact_support_fit_then_description_length_then_distinct_query_predictions',
            'representation_families':sorted({c.view for c in all_candidates.values()})}
        return report
