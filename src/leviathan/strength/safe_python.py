"""A bounded interpreter for a deliberately small Python-shaped grid language.

Never exec/eval/compile model text. No imports, arbitrary attributes, file/network
access, globals, recursion or while loops. The only objects are bounded integers,
bools, None, and bounded lists/tuples. This is a language interpreter, not a general
Python sandbox. Unsupported code is a rejected hypothesis, not an execution request.
"""
from __future__ import annotations
import ast
import copy
import operator
from functools import cmp_to_key
from .contracts import grid

class ReturnValue(Exception):
    def __init__(self,value):self.value=value

BINARY={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.FloorDiv:operator.floordiv,
        ast.Mod:operator.mod,ast.BitXor:operator.xor,ast.BitAnd:operator.and_,ast.BitOr:operator.or_}
COMPARE={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,
         ast.Gt:operator.gt,ast.GtE:operator.ge,ast.In:lambda a,b:a in b,ast.NotIn:lambda a,b:a not in b}
BUILTINS={'len','range','min','max','sum','abs','sorted','enumerate','zip','list','reversed','all','any'}
LIST_METHODS={'append','extend','copy','count','index','reverse'}
FORBIDDEN=(ast.Import,ast.ImportFrom,ast.While,ast.With,ast.AsyncFunctionDef,ast.Lambda,
           ast.ClassDef,ast.Try,ast.Raise,ast.Global,ast.Nonlocal,ast.Delete,ast.Yield,ast.YieldFrom,
           ast.Await,ast.NamedExpr,ast.Dict,ast.Set,ast.DictComp,ast.SetComp)

def syntax(source):
    if not isinstance(source,str) or not 1<=len(source)<=8192:raise ValueError('Program source length outside budget')
    tree=ast.parse(source,mode='exec')
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('Exactly one transform(x) function required')
    fn=tree.body[0]
    if fn.name!='transform' or len(fn.args.args)!=1 or fn.args.args[0].arg!='x' or fn.decorator_list:
        raise ValueError('Expected undecorated transform(x)')
    if fn.args.defaults or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs:
        raise ValueError('No optional or variadic function parameters')
    nodes=list(ast.walk(tree))
    parents={id(child):parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    if len(nodes)>600:raise ValueError('AST node budget exceeded')
    for node in nodes:
        if isinstance(node,FORBIDDEN):raise ValueError('Unsupported or unsafe syntax: '+type(node).__name__)
        if isinstance(node,ast.FunctionDef) and node is not fn:raise ValueError('No nested functions')
        if isinstance(node,ast.Name) and (node.id.startswith('_') or len(node.id)>64):raise ValueError('Invalid variable name')
        if isinstance(node,ast.Attribute):
            parent=parents.get(id(node))
            if not isinstance(parent,ast.Call) or parent.func is not node or node.attr not in LIST_METHODS:
                raise ValueError('Only explicitly interpreted list methods are allowed')
        if isinstance(node,ast.Call):
            allowed=(isinstance(node.func,ast.Name) and node.func.id in BUILTINS) or (isinstance(node.func,ast.Attribute) and node.func.attr in LIST_METHODS)
            if not allowed or node.keywords:raise ValueError('Only approved calls without keyword arguments')
        if isinstance(node,ast.Constant) and not (node.value is None or type(node.value) in (int,bool)):
            raise ValueError('Only integer/bool/None constants')
    return fn

class Interpreter:
    def __init__(self,max_steps=100000):self.max_steps=max_steps;self.steps=0;self.sequence_slots=0
    def tick(self):
        self.steps+=1
        if self.steps>self.max_steps:raise ValueError('Interpreter operation budget exhausted')
    def bound(self,value):
        if type(value) is int and abs(value)>100000:raise ValueError('Integer range exceeded')
        if isinstance(value,(list,tuple)):
            if len(value)>900:raise ValueError('Sequence length exceeded')
            self.sequence_slots+=len(value)
            if self.sequence_slots>1000000:raise ValueError('Total sequence-work budget exceeded')
        return value
    def ordering(self,a,b,depth=0):
        self.tick()
        if depth>16:raise ValueError('Nested comparison depth exceeded')
        if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
            for x,y in zip(a,b):
                relation=self.ordering(x,y,depth+1)
                if relation:return relation
            return (len(a)>len(b))-(len(a)<len(b))
        if a is None or b is None:
            if a is b:return 0
            raise ValueError('Cannot order None against another value')
        if type(a) not in (int,bool) or type(b) not in (int,bool):raise ValueError('Incompatible comparison types')
        return (a>b)-(a<b)
    def equal(self,a,b,depth=0):
        self.tick()
        if depth>16:raise ValueError('Nested equality depth exceeded')
        if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
            if type(a)!=type(b) or len(a)!=len(b):return False
            return all(self.equal(x,y,depth+1) for x,y in zip(a,b))
        return a==b
    def compare(self,op,a,b):
        if isinstance(op,(ast.In,ast.NotIn)):
            if not isinstance(b,(list,tuple)):raise ValueError('Membership requires a bounded sequence')
            found=any(self.equal(a,v) for v in b)
            return not found if isinstance(op,ast.NotIn) else found
        if isinstance(op,ast.Eq):return self.equal(a,b)
        if isinstance(op,ast.NotEq):return not self.equal(a,b)
        relation=self.ordering(a,b)
        if isinstance(op,ast.Lt):return relation<0
        if isinstance(op,ast.LtE):return relation<=0
        if isinstance(op,ast.Gt):return relation>0
        if isinstance(op,ast.GtE):return relation>=0
        raise ValueError('Unsupported comparison')
    def assign(self,node,value,env):
        self.tick()
        if isinstance(node,ast.Name):
            if node.id in BUILTINS:raise ValueError('Cannot replace builtin')
            env[node.id]=value;return
        if isinstance(node,(ast.Tuple,ast.List)):
            if not isinstance(value,(tuple,list)) or len(node.elts)!=len(value):raise ValueError('Tuple assignment arity')
            for sub,v in zip(node.elts,value):self.assign(sub,v,env)
            return
        if isinstance(node,ast.Subscript):
            seq=self.expr(node.value,env);index=self.expr(node.slice,env)
            if not isinstance(seq,list) or type(index) is not int:raise ValueError('Only integer-index list writes')
            seq[index]=value;return
        raise ValueError('Unsupported assignment target')
    def expr(self,node,env):
        self.tick()
        if isinstance(node,ast.Constant):return self.bound(node.value)
        if isinstance(node,ast.Name):
            if node.id not in env:raise ValueError('Unknown variable '+node.id)
            return env[node.id]
        if isinstance(node,(ast.Tuple,ast.List)):
            result=[self.expr(x,env) for x in node.elts]
            return self.bound(tuple(result) if isinstance(node,ast.Tuple) else result)
        if isinstance(node,ast.UnaryOp):
            v=self.expr(node.operand,env)
            if isinstance(node.op,ast.Not):return not v
            if isinstance(node.op,ast.USub) and type(v) in (int,bool):return self.bound(-v)
            raise ValueError('Unsupported unary operator')
        if isinstance(node,ast.BinOp):
            a,b=self.expr(node.left,env),self.expr(node.right,env);fn=BINARY.get(type(node.op))
            if fn is None:raise ValueError('Unsupported arithmetic')
            if isinstance(node.op,ast.Mult) and (isinstance(a,(list,tuple)) or isinstance(b,(list,tuple))):
                seq,n=(a,b) if isinstance(a,(list,tuple)) else (b,a)
                if type(n) is not int or not 0<=n<=900 or len(seq)*n>900:raise ValueError('Sequence multiplication budget')
            elif not (type(a) in (int,bool) and type(b) in (int,bool)) and not (isinstance(node.op,ast.Add) and type(a)==type(b) and isinstance(a,(list,tuple))):
                raise ValueError('Unsupported binary operand types')
            return self.bound(fn(a,b))
        if isinstance(node,ast.Compare):
            a=self.expr(node.left,env)
            for op,right in zip(node.ops,node.comparators):
                b=self.expr(right,env)
                if not self.compare(op,a,b):return False
                a=b
            return True
        if isinstance(node,ast.BoolOp):
            if isinstance(node.op,ast.And):
                for x in node.values:
                    value=self.expr(x,env)
                    if not value:return value
            else:
                for x in node.values:
                    value=self.expr(x,env)
                    if value:return value
            return value
        if isinstance(node,ast.IfExp):return self.expr(node.body if self.expr(node.test,env) else node.orelse,env)
        if isinstance(node,ast.Slice):
            return slice(*(None if x is None else self.expr(x,env) for x in (node.lower,node.upper,node.step)))
        if isinstance(node,ast.Subscript):
            seq=self.expr(node.value,env);idx=self.expr(node.slice,env)
            if not isinstance(seq,(tuple,list)) or not isinstance(idx,(int,slice)):raise ValueError('Invalid indexing')
            return self.bound(seq[idx])
        if isinstance(node,ast.Call):
            args=[]
            for x in node.args:
                if isinstance(x,ast.Starred):
                    seq=self.expr(x.value,env)
                    if not isinstance(seq,(list,tuple)) or len(seq)>30:raise ValueError('Star expansion budget')
                    args.extend(seq)
                else:args.append(self.expr(x,env))
            if isinstance(node.func,ast.Attribute):
                seq=self.expr(node.func.value,env);name=node.func.attr
                if not isinstance(seq,list):raise ValueError('Only list receivers allowed')
                if name=='append' and len(args)==1:seq.append(args[0]);self.bound(seq);return None
                if name=='extend' and len(args)==1 and isinstance(args[0],(list,tuple)):
                    if len(seq)+len(args[0])>900:raise ValueError('Extension length budget')
                    seq.extend(args[0]);self.bound(seq);return None
                if name=='copy' and not args:return self.bound(list(seq))
                if name=='reverse' and not args:seq.reverse();return None
                if name in ('count','index') and len(args)==1:
                    count=0
                    for i,value in enumerate(seq):
                        if self.equal(value,args[0]):
                            if name=='index':return i
                            count+=1
                    if name=='index':raise ValueError('Value absent from sequence')
                    return count
                raise ValueError('Invalid approved list method arguments')
            name=node.func.id
            if name=='range':
                if not 1<=len(args)<=3 or any(type(a) is not int for a in args):raise ValueError('Integer range parameters required')
                r=range(*args)
                if len(r)>60:raise ValueError('Range length budget exceeded')
                return list(r)
            if name in ('zip','enumerate','reversed'):
                if any(not isinstance(a,(list,tuple)) for a in args):raise ValueError('Sequence operands required')
                fn={'zip':zip,'enumerate':enumerate,'reversed':reversed}[name]
                return self.bound(list(fn(*args)))
            if name in ('min','max','sorted'):
                values=args[0] if len(args)==1 else args
                if not isinstance(values,(list,tuple)):raise ValueError('Sequence selection required')
                ordered=sorted(values,key=cmp_to_key(self.ordering))
                if name=='sorted':return self.bound(ordered)
                if not ordered:raise ValueError('Empty selection')
                return ordered[0 if name=='min' else -1]
            functions={'len':len,'sum':sum,'abs':abs,'list':list,'all':all,'any':any}
            return self.bound(functions[name](*args))
        if isinstance(node,(ast.ListComp,ast.GeneratorExp)):
            result=[]
            def fill(index,scope):
                if index==len(node.generators):
                    result.append(self.expr(node.elt,scope));self.bound(result);return
                g=node.generators[index]
                if g.is_async:raise ValueError('No async comprehensions')
                seq=self.expr(g.iter,scope)
                if not isinstance(seq,(list,tuple)):raise ValueError('Comprehension requires bounded sequence')
                for value in seq:
                    local=dict(scope);self.assign(g.target,value,local)
                    if all(self.expr(c,local) for c in g.ifs):fill(index+1,local)
            fill(0,dict(env));return result
        raise ValueError('Unsupported expression '+type(node).__name__)
    def statements(self,nodes,env):
        for node in nodes:
            self.tick()
            if isinstance(node,ast.Return):raise ReturnValue(self.expr(node.value,env))
            if isinstance(node,ast.Assign):
                value=self.expr(node.value,env)
                for target in node.targets:self.assign(target,value,env)
            elif isinstance(node,ast.If):self.statements(node.body if self.expr(node.test,env) else node.orelse,env)
            elif isinstance(node,ast.For):
                seq=self.expr(node.iter,env)
                if not isinstance(seq,(list,tuple)):raise ValueError('For requires bounded sequence')
                for value in list(seq):
                    self.assign(node.target,value,env);self.statements(node.body,env)
                if node.orelse:self.statements(node.orelse,env)
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):self.expr(node.value,env)
            elif isinstance(node,ast.Pass):pass
            else:raise ValueError('Unsupported statement '+type(node).__name__)
    def run(self,source,g):
        fn=syntax(source);env={'x':[list(r) for r in grid(g)]}
        try:self.statements(fn.body,env)
        except ReturnValue as result:return grid(result.value)
        raise ValueError('Program did not return a grid')

def run(source,g):
    try:return Interpreter().run(source,g)
    except (TypeError,IndexError,KeyError,ZeroDivisionError,OverflowError,RecursionError) as exc:
        raise ValueError('Rejected grid program: '+str(exc)) from exc
