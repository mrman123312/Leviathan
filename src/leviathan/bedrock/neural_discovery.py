"""Same-model hypothesis proposal -> host experiments -> beliefs -> executable memory.

No enumerated true-rule catalogue is passed to the proposer. The grammar is bounded
integer expressions, not arbitrary science. Malformed/contradicted proposals fail
honestly; no fallback secretly solves the task with the supplied answer family.
"""
from __future__ import annotations
import ast
from dataclasses import asdict
import json
from typing import Callable
import torch
from .contracts import Budget,BudgetExceeded,stable_hash
from .world import Expr,Rule,VersionSpace,X,const,op
from .runtime import Pulse
from .stable_neural import StableFrozenPolicy
from ..cognitive_kernel import CognitiveOperator
from ..memory_ecology import MemoryKind
from ..types import Belief,Provenance,ProvenanceKind,UncertaintyKind


def parse_expression(text:str,domain:tuple[int,...])->Rule:
    """Translate a bounded expression into the existing typed DSL, NEVER eval()."""
    if not isinstance(text,str) or not text.strip() or len(text)>512:
        raise ValueError("Expression must contain 1..512 characters")
    tree=ast.parse(text.strip(),mode='eval')
    if sum(1 for _ in ast.walk(tree))>128:raise ValueError("AST budget exceeded")
    def convert(node,depth=0):
        if depth>12:raise ValueError("Expression depth exceeded")
        rec=lambda n:convert(n,depth+1)
        if isinstance(node,ast.Name) and node.id=='x':return X
        if isinstance(node,ast.Constant) and type(node.value) is int:return const(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return op('mul',const(-1),rec(node.operand))
        if isinstance(node,ast.BinOp):
            if isinstance(node.op,ast.Sub):return op('add',rec(node.left),op('mul',const(-1),rec(node.right)))
            names={ast.Add:'add',ast.Mult:'mul',ast.Mod:'mod',ast.BitXor:'xor',ast.BitAnd:'and',ast.BitOr:'or'}
            kind=names.get(type(node.op))
            if kind:return op(kind,rec(node.left),rec(node.right))
        if isinstance(node,ast.Compare) and len(node.ops)==1 and isinstance(node.ops[0],ast.Lt):
            return op('lt',rec(node.left),rec(node.comparators[0]))
        if isinstance(node,ast.IfExp):return op('if',rec(node.test),rec(node.body),rec(node.orelse))
        raise ValueError('Only x, bounded integers, arithmetic/bitwise operators and conditional expressions allowed')
    rule=Rule(convert(tree.body),domain)
    for x in domain:rule.predict(x)  # validate bounds/division on entire declared domain
    return rule


class FrozenRuleProposer:
    """Uses the runtime's one frozen Qwen; proposer never receives observe() or truth."""
    def __init__(self,runtime,tokenizer,max_new_tokens=40,progress:Callable[[str],None]|None=None):
        if runtime.neural is None:raise ValueError('A single frozen neural executor is required')
        self.runtime,self.tokenizer=runtime,tokenizer
        self.max_new_tokens=max_new_tokens;self.progress=progress or (lambda s:None)
        if not 1<=max_new_tokens<=96:raise ValueError('Proposal token budget is 1..96')

    def __call__(self,domain,observations,rejections,task):
        # Toy demonstrations teach syntax only; no target-family label or hidden rule.
        demonstrations='Input/output examples: 0->2, 1->3, 2->4\nExpression: x + 2\n\n'
        demos2='Input/output examples: 0->0, 1->1, 2->0, 3->1\nExpression: x % 2\n\n'
        pairs=', '.join(f'{a}->{b}' for a,b,_ in observations)
        rejected='\nPreviously rejected: '+', '.join(rejections[-3:]) if rejections else ''
        prompt=('Complete ONE Python-style integer expression using x, integers, +, -, *, %, ^, &, |. '
                'Return only the expression, no code execution or explanation.\n\n'+demonstrations+demos2+
                f'Declared input domain: {list(domain)}\nInput/output examples: {pairs}'+rejected+'\nExpression:')
        ids=self.tokenizer(prompt,return_tensors='pt')['input_ids']
        if ids.shape[-1]>1024:raise ValueError('Hypothesis prompt token budget exceeded')
        engine=self.runtime.neural
        ids=ids.to(engine.model.get_input_embeddings().weight.device)
        variants=(StableFrozenPolicy(),StableFrozenPolicy(passes=2,gain=.06,reentry_radius=.05,
                   branch_direction='orthogonal_context',branch_mix=.5,branch_sign=-1))
        proposals=[]
        for index,policy in enumerate(variants):
            task.meter.charge('branches')
            prefix=ids
            self.progress(f'  Neural rule proposal {index+1}/{len(variants)} (same frozen model)')
            with torch.inference_mode():
                for step in range(self.max_new_tokens):
                    out=engine.run(prefix,policy=policy,meter=task.meter,request_id=task.session.id)
                    token=int(out.logits[0,-1].argmax())
                    task.meter.charge('generated_tokens')
                    prefix=torch.cat((prefix,prefix.new_tensor([[token]])),dim=-1)
                    generated=self.tokenizer.decode(prefix[0,ids.shape[1]:],skip_special_tokens=True)
                    if '\n' in generated or token==getattr(self.tokenizer,'eos_token_id',None):break
                    if (step+1)%8==0:self.progress(f'    Proposal {index+1}: {step+1} tokens generated')
            expression=generated.splitlines()[0].strip() if generated.strip() else ''
            proposals.append({'text':expression,'source':'frozen_qwen','policy':asdict(policy)})
        return proposals


def discover_neural(runtime,*,scope:str,problem:str,domain:tuple[int,...],observe:Callable[[int],int],
                    tokenizer=None,proposer=None,max_rounds:int=3,validation_count:int=2,
                    budget:Budget|None=None,progress:Callable[[str],None]|None=None):
    """Executable neural/world bridge. Injected proposers exist solely for tests.

    A singleton neural proposal is NOT an exhaustive hypothesis class. Passing fresh
    tests makes a scoped empirical skill candidate, not a theorem about unseen data.
    """
    if not scope or len(domain)<4 or len(domain)>64 or len(set(domain))!=len(domain):
        raise ValueError('Scope and a unique finite 4..64-input domain required')
    if not 1<=max_rounds<=6 or not 1<=validation_count<=len(domain)-2:
        raise ValueError('Invalid discovery/validation budget')
    if proposer is None and runtime.neural is None:raise ValueError('No frozen model bound')
    progress=progress or (lambda s:None)
    budget=budget or Budget(model_calls=384,layer_calls=14000,generated_tokens=384,
                            branches=12,environment_steps=16,wall_seconds=240.)
    task=runtime.open(problem,task_type='hidden_rule',scope=scope,budget=budget)
    source_kind='frozen_qwen' if proposer is None else 'injected_test_proposer'
    proposer=proposer or FrozenRuleProposer(runtime,tokenizer,progress=progress)
    observations=[];attempts=[];rejections=[];validation=[];candidate=None
    bid='neural-belief-'+task.session.id
    prior=Provenance(ProvenanceKind.SELF_INFERENCE,runtime.model_id,0.)
    def record_candidates(rules,evidence):
        value={'candidate_programs':[r.as_dict() for r in rules],
               'conditional_on_proposed_set':True,'calibrated':False,
               'observations':[{'x':a,'y':b,'evidence':e} for a,b,e in observations]}
        belief=Belief(bid,value,0.,prior,UncertaintyKind.EPISTEMIC,
                      proposition='Model-proposed rules, not an exhaustive world model',
                      evidence_refs=[e for _,_,e in observations])
        runtime.add_belief(task.session.id,belief,reason_ref=evidence)
    def measure(action,expected):
        actual,receipt=runtime._observe(task,scope,action,expected,observe)
        observations.append((action,actual,receipt.evidence_id))
        return actual,receipt
    try:
        for action in (domain[0],domain[-1]):measure(action,None)
        for round_id in range(max_rounds):
            progress(f'Neural discovery round {round_id+1}/{max_rounds}: {len(observations)} observed examples')
            raw=runtime.step(task,CognitiveOperator.HYPOTHESIZE,
                lambda:proposer(tuple(domain),tuple(observations),tuple(rejections),task),
                summary='Ask the same frozen neural model for candidate expressions')
            rules=[]
            for item in raw[:8]:
                text=item.get('text','') if isinstance(item,dict) else str(item)
                record={'round':round_id+1,'text':text,'source':source_kind}
                try:
                    rule=parse_expression(text,domain)
                    if not all(rule.predict(a)==b for a,b,_ in observations):
                        raise ValueError('Contradicts an already observed outcome')
                    if rule.id not in {r.id for r in rules}:rules.append(rule)
                    record['accepted_as_hypothesis']=True
                except (ValueError,SyntaxError,RecursionError) as exc:
                    record.update(accepted_as_hypothesis=False,error=str(exc));rejections.append(text[:200])
                attempts.append(record)
            runtime.step(task,CognitiveOperator.UPDATE_BELIEF,
                lambda:record_candidates(rules,observations[-1][2]),summary='Parse, validate and filter neural proposals')
            if not rules:continue
            world=VersionSpace(rules);task.world=world
            for a,b,e in observations:world.observe(a,b,evidence_id=e)
            while len(world.rules)>1:
                used={a for a,_,_ in observations}
                available=[a for a in domain if a not in used]
                if len(available)<=validation_count:break
                action=runtime.step(task,CognitiveOperator.PLAN,lambda:world.choose(available),
                    summary='Choose an input that separates the model-proposed hypotheses')
                actual,receipt=measure(action,world.predict(action))
                survivors=[r for r in world.rules if r.predict(action)==actual]
                if not survivors:break
                world.observe(action,actual,evidence_id=receipt.evidence_id)
                runtime.step(task,CognitiveOperator.UPDATE_BELIEF,
                    lambda:record_candidates(world.rules,receipt.evidence_id),summary='Revise after host observation')
            survivors=[r for r in world.rules if all(r.predict(a)==b for a,b,_ in observations)]
            if len(survivors)!=1:continue
            proposed=survivors[0]
            fresh=[a for a in domain if a not in {p[0] for p in observations}]
            if len(fresh)<validation_count:break
            validation=[]
            for action in fresh[:validation_count]:
                expected=proposed.predict(action)
                actual,receipt=measure(action,expected)
                validation.append({'action':action,'expected':expected,'observed':actual,**asdict(receipt)})
                if not receipt.passed:
                    runtime.add_pulse(task,Pulse('contradiction',f'Candidate failed at {action}',(receipt.evidence_id,),True))
                    rejections.append(str(proposed.as_dict())[:200]);break
            if len(validation)==validation_count and all(r['passed'] for r in validation):
                candidate=proposed;break
    except BudgetExceeded as exc:
        attempts.append({'budget_exhausted':str(exc)})
    finally:
        if runtime.neural is not None:runtime.neural.reset_request()
    report={'status':'empirically_validated_candidate' if candidate else 'not_solved',
            'proposal_source':source_kind,'attempts':attempts,'validation':validation,
            'observations':[{'x':a,'y':b,'evidence':e} for a,b,e in observations],
            'training_steps':0,'budget':task.meter.snapshot(),
            'neural_model_calls':task.meter.used.get('model_calls',0),
            'catalogue_fallback_used':False,'exhaustive_hypothesis_class':False,
            'session_id':task.session.id}
    eid=validation[-1]['evidence_id'] if validation else 'unresolved-'+stable_hash(report)
    runtime.record_verification(task.session.id,eid)
    payload={'type':'bedrock_rule' if candidate else 'neural_rule_failure','scope':scope,
             'claim_scope':'finite_domain_empirical_support_only','calibrated_confidence':False,
             'proposal_source':source_kind,'validation':validation,'attempts':attempts}
    if candidate:payload['rule']=candidate.as_dict()
    completion=runtime.complete_task(task.session.id,outcome_ref=eid,verified_success=candidate is not None,
        provenance=Provenance(ProvenanceKind.DETERMINISTIC_EXECUTION,scope,1.),
        truth_quality=.8 if candidate else 0.,novelty=1.,transfer_value=.7,
        independent_verification=candidate is not None,rollback_available=True,episode_payload=payload)
    if candidate:
        sid='neural-skill-'+stable_hash([scope,candidate.as_dict(),task.session.id])[:24]
        runtime.memory.promote_episode(completion.episode_id,new_id=sid,destination=MemoryKind.PROCEDURAL,
            verification_ref=eid,verification_confidence=.8,independence_score=1.)
        report.update(skill_id=sid,rule=candidate.as_dict())
    report['graph_complete']=task.session.graph.complete
    return report
