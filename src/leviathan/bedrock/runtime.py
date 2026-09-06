"""One shared no-training cognitive runtime, built on Leviathan's existing kernel.

Neural execution, safe rule programs, belief revisions, host verification, memory
and procedural reuse share one task identity and dependency graph. Deterministic
world learning is not relabelled a neural capability or an AGI demonstration.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
import json
from typing import Any, Callable, Iterable
from uuid import uuid4
from ..runtime import LeviathanRuntime, TaskSession
from ..cognitive_kernel import (CognitiveGraphNode,CognitiveInstruction,CognitiveOperator,
    CognitiveProgram,DynamicCognitiveGraph,GoalState,RepresentationKind,RepresentationPlan)
from ..memory_ecology import MemoryKind
from ..types import Belief,MetaState,Provenance,ProvenanceKind,UncertaintyKind
from .contracts import Budget,BudgetExceeded,CausalTrace,Competence,Meter,Outcome,stable_hash
from .world import Contradiction,Rule,VersionSpace


@dataclass(frozen=True)
class Pulse:
    """Explicit compact artifact, not a claim to read hidden/private model reasoning."""
    kind: str
    content: str
    evidence: tuple[str,...]=()
    verified: bool=False

    def __post_init__(self):
        if self.kind not in {"hypothesis","prediction","observation","contradiction","plan","equation"}:
            raise ValueError("Unknown pulse type")
        if not self.content.strip() or len(self.content)>1024:
            raise ValueError("Pulse must contain 1..1024 characters")
        if self.verified and not self.evidence:
            raise ValueError("Verified pulse needs evidence")


@dataclass
class BedrockTask:
    session: TaskSession
    meter: Meter
    trace: CausalTrace=field(default_factory=CausalTrace)
    last_node: str|None=None
    last_trace: str|None=None
    pulses: list[Pulse]=field(default_factory=list)
    world: VersionSpace|None=None
    skill_id: str|None=None


class BedrockRuntime(LeviathanRuntime):
    """Extends the existing runtime; does not construct a second semantic model.

    Optional neural executor owns the sole model. Host callback observations and
    typed programs can run without it, but reports explicitly mark neural_calls=0.
    """
    def __init__(self,*,model_id:str,neural=None,memory_journal:str|None=None):
        super().__init__(model_id=model_id,memory_journal=memory_journal)
        if neural is not None and neural.model_id!=model_id:
            raise ValueError("Cognitive and neural identities must be the same")
        self.neural=neural
        self.competence=Competence()
        self.tasks:dict[str,BedrockTask]={}
        self.training_enabled=False

    def _next_session_id(self,*,problem,goal):
        # Persistent memory must not collide after process restart.
        return "task-"+uuid4().hex

    def open(self,problem:str,*,task_type:str="reasoning",budget:Budget=Budget(),
             scope:str="local") -> BedrockTask:
        goal=GoalState(problem,constraints=("one_model","no_training","verified_promotion"),
                       success_tests=("external_outcome",),risk_limit=0.)
        meta=MetaState(task_type=task_type,goal=problem,success_probability=.5,
            epistemic_uncertainty=1.,aleatoric_uncertainty=0.,stakes=0.,risk_budget=0.,
            compute_budget=.5,latency_budget=.5)
        session=super().begin_task(problem=problem,task_type=task_type,goal=goal,meta_state=meta)
        # Replace the old abstract template with a live graph populated by executed
        # operators. We do not mark placeholder HYPOTHESIZE/VERIFY nodes as executed.
        instruction=CognitiveInstruction("bedrock-0",CognitiveOperator.ENCODE)
        representation=RepresentationPlan(RepresentationKind.SYMBOLIC if task_type=="hidden_rule"
            else RepresentationKind.TOKEN,(),True,1,("typed_runtime_program",))
        session.program=CognitiveProgram(goal,representation,(instruction,),256,1)
        session.graph=DynamicCognitiveGraph.from_program(session.program)
        task=BedrockTask(session,Meter(budget))
        self.tasks[session.id]=task
        session.graph.start(instruction.id)
        trace=task.trace.add("encode",{"problem":problem,"scope":scope})
        session.graph.finish(instruction.id,result_ref=trace)
        task.last_node=instruction.id;task.last_trace=trace
        return task

    def step(self,task:BedrockTask,operator:CognitiveOperator,fn:Callable[[],Any],*,summary:str):
        if task.session.closed:raise RuntimeError("Task is already closed")
        task.meter.check_time()
        graph=task.session.graph
        if len(graph.nodes)>=task.session.program.max_steps:
            raise BudgetExceeded("Cognitive operator budget exhausted")
        name=f"bedrock-{len(graph.nodes)}"
        parents=(task.last_node,) if task.last_node else ()
        instruction=CognitiveInstruction(name,operator,dependencies=parents)
        graph.nodes[name]=CognitiveGraphNode(instruction)
        task.session.program=replace(task.session.program,instructions=(*task.session.program.instructions,instruction))
        graph.start(name)
        try:
            result=fn()
            eid=task.trace.add(operator.value,{"summary":summary},
                                (task.last_trace,) if task.last_trace else ())
            graph.finish(name,result_ref=eid)
        except Exception as exc:
            graph.fail(name,error=f"{type(exc).__name__}: {exc}")
            self.kernel.event_log.append(event_type="operator_failed",module="bedrock",
                metadata={"task_id":task.session.id,"operator":operator.value,"error":str(exc)})
            raise
        task.last_node,task.last_trace=name,eid
        self.kernel.event_log.append(event_type="operator_executed",module="bedrock",
            input_refs=parents,output_refs=(name,),metadata={"task_id":task.session.id,"operator":operator.value})
        return result

    def add_pulse(self,task:BedrockTask,pulse:Pulse):
        if task.session.closed:raise RuntimeError("Task is closed")
        task.pulses.append(pulse)
        task.pulses=task.pulses[-16:]

    def pulse_text(self,task:BedrockTask):
        # Scope labels survive re-encoding. A hypothesis is not silently made a fact.
        return "\n".join(json.dumps({"kind":p.kind,"content":p.content,
            "verified":p.verified,"evidence":p.evidence},ensure_ascii=False) for p in task.pulses)

    def _observe(self,task,scope,action,expected,observe):
        prediction={"scope":scope,"action":action,"expected":expected}
        pid="prediction-"+stable_hash([task.session.id,len(task.session.prediction_ids),prediction])
        self.step(task,CognitiveOperator.PREDICT,
                  lambda:self.record_prediction(task.session.id,pid),summary=json.dumps(prediction))
        self.add_pulse(task,Pulse("prediction",json.dumps(prediction)))
        def act():
            task.meter.charge("environment_steps")
            self.record_action(task.session.id,"action-"+pid)
            value=observe(action)
            if type(value) is not int:raise TypeError("Environment must return a bounded integer observation")
            return value
        actual=self.step(task,CognitiveOperator.EXECUTE,act,summary=f"experiment {action}")
        eid="observation-"+stable_hash([pid,actual])
        data={"scope":scope,"action":action,"expected":expected,"observed":actual}
        outcome=Outcome(stable_hash(data),"host_environment",None if expected is None else expected==actual,
                        eid,scope,True,"Trust applies only to this supplied environment callback")
        self.step(task,CognitiveOperator.VERIFY,
                  lambda:self.record_verification(task.session.id,eid),summary=json.dumps(data))
        self.add_pulse(task,Pulse("observation",json.dumps({"action":action,"observed":actual}),
                                 (eid,),True))
        return actual,outcome

    def discover(self,*,problem:str,scope:str,rules:Iterable[Rule],observe:Callable[[int],int],
                 budget:Budget=Budget(),validation_count:int=2):
        """Explore a declared grammar, validate on fresh actions, then compile a rule.

        Validation is within the supplied environment, not a protected language test.
        Protected transfer queries remain outside this API. No optimizer is involved.
        """
        if not scope or validation_count<1:raise ValueError("Scope and fresh validation required")
        task=self.open(problem,task_type="hidden_rule",scope=scope,budget=budget)
        pool=tuple(rules)
        world=self.step(task,CognitiveOperator.HYPOTHESIZE,lambda:VersionSpace(pool),
                        summary="Construct declared finite hypothesis class")
        task.world=world
        self.step(task,CognitiveOperator.ABSTRACT,lambda:world.domain,
                  summary="Use bounded executable rule AST, not unstructured text")
        self.step(task,CognitiveOperator.RECALL,
                  lambda:self.memory.retrieve(kinds=(MemoryKind.PROCEDURAL,),tags=(scope,),limit=8),
                  summary="Retrieve scoped procedural experience")
        grammar_hash=stable_hash(sorted(r.id for r in world.rules))
        provenance=Provenance(ProvenanceKind.DETERMINISTIC_EXECUTION,scope,1.,grammar_hash)
        bid=f"belief-{task.session.id}"
        self.add_belief(task.session.id,Belief(bid,{"survivors":len(world.rules)},.0,provenance,
            UncertaintyKind.EPISTEMIC,proposition="Current finite-rule version space"),reason_ref="declared_grammar")
        transcript=[]
        try:
            while len(world.rules)>1:
                action=self.step(task,CognitiveOperator.PLAN,lambda:world.choose(world.domain),
                                 summary="Maximize expected information gain per experiment cost")
                predictions=world.predictions(action)
                expected=world.predict(action)
                actual,receipt=self._observe(task,scope,action,expected,observe)
                self.step(task,CognitiveOperator.UPDATE_BELIEF,
                    lambda:world.observe(action,actual,evidence_id=receipt.evidence_id),
                    summary="Eliminate rules contradicted by the observed outcome")
                # Conditional mass is NOT published as calibrated real-world confidence.
                self.beliefs.put(Belief(bid,{"survivors":len(world.rules),"grammar":grammar_hash},
                    1/len(world.rules),provenance,UncertaintyKind.EPISTEMIC,
                    proposition="Candidate mass conditional on the declared deterministic grammar",
                    evidence_refs=[e for _,_,e in world.observations]),reason_ref=receipt.evidence_id)
                task.session.meta_state=replace(task.session.meta_state,
                    epistemic_uncertainty=min(1.,world.entropy/max(1.,math_log2(len(pool)))))
                transcript.append({"action":action,"distribution":predictions,"observed":actual,
                                   "survivors":len(world.rules),"evidence":receipt.evidence_id})
            rule=world.rules[0]
            used={a for a,_,_ in world.observations}
            fresh=[a for a in world.domain if a not in used][:validation_count]
            if len(fresh)<validation_count:
                raise Contradiction("Not enough fresh actions for compilation validation")
            validation=[]
            for action in fresh:
                actual,receipt=self._observe(task,scope,action,rule.predict(action),observe)
                validation.append({**asdict(receipt), "action":action, "prediction":rule.predict(action), "observed":actual})
                if not receipt.passed:
                    self.add_pulse(task,Pulse("contradiction",f"Rule failed fresh action {action}",
                                              (receipt.evidence_id,),True))
                    raise Contradiction("Candidate failed fresh validation")
            payload={"type":"bedrock_rule","scope":scope,"rule":rule.as_dict(),
                     "grammar_hash":grammar_hash,"validation":validation,
                     "claim_scope":"finite_domain_only","training_steps":0}
            completion=self.complete_task(task.session.id,outcome_ref=validation[-1]["evidence_id"],
                verified_success=True,provenance=provenance,truth_quality=1.,novelty=1.,transfer_value=.8,
                independent_verification=True,rollback_available=True,episode_payload=payload)
            sid="rule-skill-"+stable_hash([scope,rule.as_dict()])[:24]
            prior=[m for m in self.memory.records if m.id==sid]
            if not prior:
                self.memory.promote_episode(completion.episode_id,new_id=sid,
                    destination=MemoryKind.PROCEDURAL,verification_ref=validation[-1]["evidence_id"],
                    verification_confidence=1.,independence_score=1.)
            task.skill_id=sid
            self.competence.record(scope,"finite_rule_induction",Outcome(stable_hash(payload),
                "fresh_environment_validation",True,validation[-1]["evidence_id"],scope,True))
            self.kernel.event_log.append(event_type="procedural_rule_compiled",module="bedrock",
                output_refs=(sid,),metadata={"scope":scope,"no_parameter_updates":True})
            return {"status":"validated_in_declared_domain","skill_id":sid,"rule":rule.as_dict(),
                    "transcript":transcript,"validation":validation,"world":world.report(),
                    "budget":task.meter.snapshot(),"neural_calls":0,"training_steps":0,
                    "session_id":task.session.id,"graph_complete":task.session.graph.complete,
                    "general_intelligence_demonstrated":False}
        except (Contradiction,BudgetExceeded) as exc:
            eid="failure-"+stable_hash([task.session.id,str(exc)])
            self.record_verification(task.session.id,eid)
            self.complete_task(task.session.id,outcome_ref=eid,verified_success=False,provenance=provenance,
                truth_quality=0.,novelty=1.,transfer_value=0.,independent_verification=True,
                rollback_available=True,episode_payload={"scope":scope,"error":str(exc),"world":world.report()})
            return {"status":"not_solved","reason":str(exc),"world":world.report(),
                    "transcript":transcript,"budget":task.meter.snapshot(),"training_steps":0,
                    "session_id":task.session.id,"neural_calls":0}

    def transfer(self,*,scope:str,action:int,observe:Callable[[int],int]|None=None):
        """Execute a scoped saved rule after original prompt/context has disappeared.

        A failed fresh observation deprecates the skill. Without a verifier this is
        explicitly a prediction, not a newly verified fact.
        """
        records=self.memory.retrieve(kinds=(MemoryKind.PROCEDURAL,),tags=("hidden_rule",scope),limit=128)
        candidates=[r for r in records if isinstance(r.payload,dict) and r.payload.get("type")=="bedrock_rule" and r.payload.get("scope")==scope]
        if not candidates:raise KeyError("No verified active skill for this scope")
        predictions={Rule.parse(r.payload["rule"]).predict(action) for r in candidates}
        if len(predictions)!=1:raise Contradiction("Stored skills disagree; new evidence is required")
        record=candidates[0];prediction=next(iter(predictions))
        result={"action":action,"prediction":prediction,"skill_id":record.id,"verified_now":False,
                "source_evidence":list(record.evidence_refs),"neural_calls":0,"training_steps":0}
        if observe is not None:
            task=self.open(f"Verify reused rule for action {action}",task_type="hidden_rule",scope=scope)
            actual,receipt=self._observe(task,scope,action,prediction,observe)
            result.update({"observed":actual,"verified_now":receipt.passed})
            if receipt.passed is False:
                for r in candidates:self.memory.deprecate(r.id,reason_ref=receipt.evidence_id)
                result["skill_deprecated"]=True
            self.competence.record(scope,"procedural_reuse",receipt)
            self.complete_task(task.session.id,outcome_ref=receipt.evidence_id,verified_success=bool(receipt.passed),
                provenance=Provenance(ProvenanceKind.DETERMINISTIC_EXECUTION,scope,1.),
                truth_quality=float(bool(receipt.passed)),novelty=0.,transfer_value=.8,
                independent_verification=True,rollback_available=True,episode_payload=result)
        return result

    def respond(self,text:str,tokenizer,*,policies=None,verifier=None,max_new_tokens:int=16,
                budget:Budget=Budget(),pulses:tuple[Pulse,...]=()):
        """Cognitive graph -> actual same-model latent routes -> verification -> memory.

        The existing tokenizer/embedding/coda/head encode explicit pulses. No new
        pulse projection is initialized. Without an external verifier, branch
        consensus cannot override the untouched donor answer.
        """
        if self.neural is None:raise RuntimeError("Bind the single frozen model before text inference")
        from .neural import FrozenPolicy
        task=self.open(text,task_type="text",budget=budget)
        for pulse in pulses:self.add_pulse(task,pulse)
        memory=self.step(task,CognitiveOperator.RECALL,
            lambda:self.memory.retrieve(tags=("text",),limit=4),summary="Read prior scoped experience")
        # Memory informs mode selection; unverified old answers are never replayed as truth.
        method="verified_search" if verifier is not None else "direct"
        defaults=(FrozenPolicy(),FrozenPolicy(passes=2,gain=.15),FrozenPolicy(passes=4,gain=.15))
        routes=tuple(policies) if policies is not None else (defaults if method=="verified_search" else defaults[:1])
        self.step(task,CognitiveOperator.PLAN,lambda:method,
                  summary=f"Select {method}; {len(memory)} past episodes, independent verifier={verifier is not None}")
        prompt=text
        if task.pulses:
            prompt += "\nStructured working artifacts (hypotheses are not facts):\n"+self.pulse_text(task)+"\nResponse:"
        tokens=tokenizer(prompt,return_tensors="pt")["input_ids"]
        if tokens.shape[-1]>2048:raise ValueError("Prompt exceeds the fixed 2048-token experiment budget")
        device=self.neural.model.get_input_embeddings().weight.device
        tokens=tokens.to(device)
        try:
            result=self.step(task,CognitiveOperator.SEARCH,
                lambda:self.neural.brainstorm(tokens,policies=routes,meter=task.meter,
                    verifier=verifier,max_new_tokens=max_new_tokens,request_id=task.session.id),
                summary="Evaluate frozen routes with one parameter owner")
            chosen=result["branches"][result["selected"]]
            answer=tokenizer.decode(chosen["tokens"],skip_special_tokens=True)
            receipt=chosen["verification"]
            known=bool(receipt and receipt["passed"] is True and receipt["independent"])
            verification_ref=receipt["evidence_id"] if receipt else "unverified-"+stable_hash(chosen["tokens"])
            self.step(task,CognitiveOperator.VERIFY,
                lambda:self.record_verification(task.session.id,verification_ref),
                summary="Host verification" if known else "No independent verifier; donor fallback, outcome unknown")
            self.complete_task(task.session.id,outcome_ref=verification_ref,verified_success=known,
                provenance=Provenance(ProvenanceKind.DETERMINISTIC_EXECUTION if known else
                    ProvenanceKind.SELF_INFERENCE, self.model_id,1. if known else .0),
                truth_quality=1. if known else .0,novelty=.5,transfer_value=.3,
                independent_verification=known,rollback_available=True,
                episode_payload={"text":text,"answer":answer,"verified":known,"routes":len(routes)})
            return {"answer":answer,"search":result,"budget":task.meter.snapshot(),
                    "session_id":task.session.id,"training_steps":0,"graph_complete":task.session.graph.complete,
                    "verification_status":"passed" if known else "unknown"}
        finally:
            self.neural.reset_request()


def math_log2(value):
    import math
    return math.log2(max(1,value))
