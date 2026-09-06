"""Training-free finite hypothesis induction and safe executable rule programs.

Generalization is relative to the declared grammar/domain, not arbitrary worlds.
Only the environment callback returns observations. No candidate reads a hidden
rule or a protected query label. This is a cognitive algorithm, not a neural head.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterable
from .contracts import stable_hash


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()

    def __post_init__(self):
        arity={"x":0,"const":1,"add":2,"mul":2,"mod":2,"xor":2,
               "and":2,"or":2,"bit":2,"lt":2,"if":3}
        if self.op not in arity or len(self.args)!=arity[self.op]:
            raise ValueError("Unknown opcode or wrong arity")
        if self.op=="const":
            if type(self.args[0]) is not int or abs(self.args[0])>65536:
                raise ValueError("Constant outside bounded integer DSL")
        elif any(not isinstance(a,Expr) for a in self.args):
            raise ValueError("Operands must be typed expressions")
        if self.nodes>64:
            raise ValueError("Program node budget exceeded")

    @property
    def nodes(self):
        return 1+sum(a.nodes for a in self.args if isinstance(a,Expr))

    def run(self,x:int)->int:
        if type(x) is not int or abs(x)>65536:
            raise ValueError("Input outside bounded integer DSL")
        if self.op=="x":return x
        if self.op=="const":return self.args[0]
        if self.op=="if":
            return self.args[1 if self.args[0].run(x) else 2].run(x)
        a,b=(v.run(x) for v in self.args)
        if self.op=="add":out=a+b
        elif self.op=="mul":out=a*b
        elif self.op=="mod":
            if b<=0:raise ValueError("Positive modulus required")
            out=a%b
        elif self.op=="bit":
            if not 0<=b<=15:raise ValueError("Bit index outside budget")
            out=(a>>b)&1
        elif self.op=="xor":out=a^b
        elif self.op=="and":out=a&b
        elif self.op=="or":out=a|b
        elif self.op=="lt":out=int(a<b)
        else:raise AssertionError("Opcode dispatch incomplete")
        if abs(out)>2**31:raise ValueError("Intermediate integer budget exceeded")
        return out

    def as_dict(self):
        return {"op":self.op,"args":[a.as_dict() if isinstance(a,Expr) else a for a in self.args]}

    @classmethod
    def parse(cls,raw:dict,depth:int=0):
        if depth>16 or not isinstance(raw,dict) or set(raw)!={"op","args"}:
            raise ValueError("Malformed or too-deep rule AST")
        if not isinstance(raw["args"],list) or len(raw["args"])>3:
            raise ValueError("Invalid AST operands")
        args=tuple(a if raw["op"]=="const" else cls.parse(a,depth+1) for a in raw["args"])
        return cls(raw["op"],args)


def const(n):return Expr("const",(n,))
def op(name,*args):return Expr(name,args)
X=Expr("x")


@dataclass(frozen=True)
class Rule:
    program: Expr
    domain: tuple[int,...]
    label: str = "candidate"

    def __post_init__(self):
        if not self.domain or len(self.domain)>256 or len(set(self.domain))!=len(self.domain):
            raise ValueError("Finite unique domain of at most 256 inputs required")
        if any(type(x) is not int or abs(x)>65536 for x in self.domain):
            raise ValueError("Rule domain outside integer budget")

    def predict(self,action:int):
        if action not in self.domain:raise ValueError("Skill precondition/domain violated")
        return self.program.run(action)

    @property
    def id(self):return stable_hash(self.as_dict())

    def as_dict(self):
        return {"program":self.program.as_dict(),"domain":list(self.domain)}

    @classmethod
    def parse(cls,raw):
        if not isinstance(raw,dict) or set(raw)!={"program","domain"}:
            raise ValueError("Malformed rule")
        return cls(Expr.parse(raw["program"]),tuple(raw["domain"]))


class Contradiction(RuntimeError):
    pass


class VersionSpace:
    """Exact consistency elimination, conditional on deterministic in-grammar world.

    Entropy and information gain are over a finite hypothesis prior; they are not
    calibrated confidence about real-world truth. Equivalent programs are deduplicated
    by behavior on the declared finite domain before exploration.
    """
    def __init__(self,rules:Iterable[Rule]):
        pool=list(rules)
        if not pool or len(pool)>4096:raise ValueError("Hypothesis budget is 1..4096")
        self.domain=pool[0].domain
        if any(r.domain!=self.domain for r in pool):raise ValueError("Incompatible hypothesis domains")
        behavioral={}
        for rule in sorted(pool,key=lambda r:r.program.nodes):
            behavior=tuple(rule.predict(a) for a in self.domain)
            behavioral.setdefault(behavior,rule)
        self.rules=tuple(behavioral.values())
        self._predictions={r.id:tuple(r.predict(a) for a in self.domain) for r in self.rules}
        self._index={a:i for i,a in enumerate(self.domain)}
        self._seen:dict[str,tuple[int,int]]={}
        self.observations:list[tuple[int,int,str]]=[]
        self.contradictions:list[dict]=[]
        self.misspecified=False

    @property
    def entropy(self):return math.log2(len(self.rules))

    def predictions(self,action):
        if action not in self._index:raise ValueError("Action outside declared domain")
        column=self._index[action]
        counts={}
        for rule in self.rules:
            result=self._predictions[rule.id][column]
            counts[result]=counts.get(result,0)+1
        return {k:v/len(self.rules) for k,v in counts.items()}

    def information_gain(self,action):
        # Deterministic predictions: I(H;Y|a) = H(Y|a).
        return -sum(p*math.log2(p) for p in self.predictions(action).values())

    def choose(self,actions:Iterable[int],costs:dict[int,float]|None=None):
        if self.misspecified:raise Contradiction("Cannot plan as though the inconsistent hypothesis class is true")
        used={a for a,_,_ in self.observations}
        choices=[a for a in actions if a not in used]
        if not choices:raise ValueError("No unobserved actions available")
        costs=costs or {}
        if any(not math.isfinite(costs.get(a,1.0)) or costs.get(a,1.0)<=0 for a in choices):
            raise ValueError("Positive finite experiment costs required")
        return max(choices,key=lambda a:(self.information_gain(a)/costs.get(a,1.),-self.domain.index(a)))

    def observe(self,action:int,value:int,*,evidence_id:str):
        if action not in self.domain or type(value) is not int or not evidence_id:
            raise ValueError("Malformed observation")
        old=self._seen.get(evidence_id)
        if old is not None:
            if old!=(action,value):raise ValueError("An evidence ID cannot change its content")
            return False
        if self.misspecified:raise Contradiction("Expand/replace the declared grammar before continuing")
        surviving=tuple(r for r in self.rules if self._predictions[r.id][self._index[action]]==value)
        self._seen[evidence_id]=(action,value)
        self.observations.append((action,value,evidence_id))
        if not surviving:
            self.contradictions.append({"action":action,"observed":value,"evidence":evidence_id})
            self.misspecified=True
            raise Contradiction("Observation contradicts every declared hypothesis; no invented confidence")
        self.rules=surviving
        return True

    def predict(self,action):
        if self.misspecified:raise Contradiction("World model is inconsistent")
        values=self.predictions(action)
        return next(iter(values)) if len(values)==1 else None

    def report(self):
        return {"survivors":len(self.rules),"entropy_bits":self.entropy,
                "observations":len(self.observations),"misspecified":self.misspecified,
                "uncertainty_scope":"finite_deterministic_hypothesis_class_only"}


def catalogue(family:str)->tuple[Rule,...]:
    """Declared research grammars, not a universal environment learner."""
    if family=="affine_mod11":
        return tuple(Rule(op("mod",op("add",op("mul",const(a),X),const(b)),const(11)),
                          tuple(range(11)),f"affine-{a}-{b}") for a in range(1,11) for b in range(11))
    if family=="bit_permutation":
        domain=tuple(range(8));rules=[]
        for permutation in itertools.permutations(range(3)):
            permuted=op("add",op("add",op("bit",X,const(permutation[0])),
                op("mul",const(2),op("bit",X,const(permutation[1])))),
                op("mul",const(4),op("bit",X,const(permutation[2]))))
            for mask in range(8):rules.append(Rule(op("xor",permuted,const(mask)),domain))
        return tuple(rules)
    if family=="boolean_circuit":
        rules=[]
        for i,j,k in itertools.permutations(range(3)):
            for first,second in itertools.product(("and","or","xor"),repeat=2):
                rules.append(Rule(op(second,op(first,op("bit",X,const(i)),op("bit",X,const(j))),
                                             op("bit",X,const(k))),tuple(range(8))))
        return tuple(rules)
    raise ValueError("Unknown declared family")


def structural_counterfactual(equations:dict[str,tuple[tuple[str,...],Any]],
                              inputs:dict[str,int],intervention:dict[str,int]):
    """Exact recomputation GIVEN a supplied deterministic structural causal model.

    Callable equations are installed by the trusted host, never executed from model
    text. This does not infer causal structure from correlations or prove the SCM true.
    """
    if not set(intervention)<=set(inputs)|set(equations):raise ValueError("Unknown intervention target")
    if set(inputs)&set(equations):raise ValueError("Inputs and endogenous variables overlap")
    def evaluate(do):
        values=dict(inputs);values.update(do)
        pending=dict(equations)
        while pending:
            progress=False
            for name,(parents,fn) in list(pending.items()):
                if name in do:del pending[name];progress=True
                elif all(p in values for p in parents):
                    values[name]=fn(*(values[p] for p in parents));del pending[name];progress=True
            if not progress:raise ValueError("Causal graph is cyclic or has missing parents")
        return values
    return {"actual":evaluate({}),"counterfactual":evaluate(intervention),
            "claim_scope":"conditional_on_supplied_SCM"}
