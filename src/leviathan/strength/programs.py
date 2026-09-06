"""Typed, bounded, pure Grid->Grid programs. Parsing never calls eval/exec.

Completeness is only relative to this finite operator vocabulary and search budget.
The model proposes programs; it never installs executable Python code.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
from itertools import product
import numpy as np
from . import grid as G
from .contracts import Grid, digest, grid

# Parameter types are checked at construction. Each program has one grid child;
# branching over hypotheses belongs to search, not another language model.
SCHEMA={
    'python_grid':('source',),
    'merge':('bg',), 'intersection':('color','bg'), 'xor_grids':('color','bg'),
    'kronecker':('bg',), 'paint_mask':('color','bg'),
    'x':(), 'rot90':(), 'rot180':(), 'rot270':(), 'flip_lr':(), 'flip_ud':(), 'transpose':(),
    'crop':('bg',), 'crop_color':('color',), 'recolor':('color','color'), 'color_map':('mapping',),
    'scale':('factor',), 'tile':('factor','factor'), 'subsample':('factor',), 'compress':(),
    'largest':('bg','conn','bool','bool'), 'smallest':('bg','conn','bool','bool'),
    'top_object':('bg','conn','bool','bool'), 'bottom_object':('bg','conn','bool','bool'),
    'fill_holes':('color','bg'), 'connect':('color','axis','bg'), 'gravity':('direction','bg'),
    'complete_symmetry':('symmetry','bg'), 'periodic':('bg',),
    'panel':('hv','bool'), 'overlay':('hv','logic','color','bg'),
    'mirror':('hv','bool'), 'border':('color','thickness'), 'trim':('thickness',),
    'translate':('shift','shift','bg'), 'mask':('color','color','bg'),
    'object_boxes':('color','bg','bool'), 'erase_color':('color','bg'),
    'count_objects':('color','hv','bg'), 'color_histogram':('hv','bg'),
}


BINARY={'merge','intersection','xor_grids','kronecker','paint_mask'}

def validate_param(kind,value):
    enums={'axis':('h','v','both'),'hv':('h','v'),'direction':('up','down','left','right'),
           'symmetry':('h','v','r','d'),'logic':('or','and','xor','diff','nor')}
    if kind=='source':
        from .safe_python import syntax
        syntax(value)
    elif kind in enums:
        if value not in enums[kind]:raise ValueError('Invalid symbolic parameter')
    elif kind=='mapping':
        if not isinstance(value,tuple) or len(value)!=10 or any(type(x) is not int or not 0<=x<=9 for x in value):
            raise ValueError('Color map must contain 10 symbols')
    else:
        lo,hi={'color':(0,9),'bg':(-1,9),'factor':(1,5),'conn':(4,8),'bool':(0,1),
               'thickness':(1,5),'shift':(-30,30)}[kind]
        if type(value) is not int or not lo<=value<=hi or (kind=='conn' and value not in (4,8)):
            raise ValueError('Invalid bounded integer parameter')

@dataclass(frozen=True)
class Program:
    op: str='x'
    child: 'Program | None'=None
    params: tuple=()
    other: 'Program | None'=None
    def __post_init__(self):
        if self.op not in SCHEMA or len(self.params)!=len(SCHEMA[self.op]):
            raise ValueError('Unknown operation or invalid parameter count')
        if self.op=='x' and self.child is not None:raise ValueError('x has no child')
        if self.op!='x' and not isinstance(self.child,Program):raise ValueError('Grid child required')
        if self.op in BINARY:
            if not isinstance(self.other,Program):raise ValueError('Second grid operand required')
        elif self.other is not None:raise ValueError('Unexpected second grid operand')
        for k,v in zip(SCHEMA[self.op],self.params):validate_param(k,v)
        if self.size>16:raise ValueError('Program size exceeds 16 bounded operations')
    @property
    def size(self): return 0 if self.op=='x' else 1+self.child.size+(self.other.size if self.other else 0)
    @property
    def id(self): return digest(self.as_dict())
    def as_dict(self):
        return {'op':self.op,'params':list(self.params),'child':None if self.child is None else self.child.as_dict(),
                'other':None if self.other is None else self.other.as_dict()}
    @classmethod
    def from_dict(cls,obj,depth=0):
        if depth>16 or not isinstance(obj,dict) or set(obj) not in ({'op','params','child'},{'op','params','child','other'}):raise ValueError('Invalid AST')
        params=tuple(tuple(x) if isinstance(x,list) else x for x in obj['params'])
        return cls(obj['op'],None if obj['child'] is None else cls.from_dict(obj['child'],depth+1),params,
                   None if obj.get('other') is None else cls.from_dict(obj['other'],depth+1))
    def __str__(self):
        if self.op=='x':return 'x'
        extra=''.join(', '+repr(p) for p in self.params)
        return f'{self.op}({self.child}'+(f', {self.other}' if self.other else '')+extra+')'
    def then(self,op,params=()):return Program(op,self,tuple(params))
    def run(self,value:Grid)->Grid:
        g=grid(value)
        if self.op=='x':return g
        if self.other is not None:return apply_binary(self.op,self.child.run(g),self.other.run(g),self.params)
        return apply(self.op,self.child.run(g),self.params)
    def replace_input(self,child):
        return child if self.op=='x' else Program(self.op,self.child.replace_input(child),self.params,
            self.other.replace_input(child) if self.other else None)


def parse(text:str)->Program:
    if not isinstance(text,str) or len(text)>4096:raise ValueError('Program text budget exceeded')
    text=text.strip()
    if text.startswith('```'):text='\n'.join(text.splitlines()[1:]).split('```')[0].strip()
    if text.startswith('Program:'):text=text[len('Program:'):].strip()
    if text.startswith('def transform('):
        return Program('python_grid',Program(),(text,))
    tree=ast.parse(text,mode='eval')
    if sum(1 for _ in ast.walk(tree))>160:raise ValueError('Syntax node budget exceeded')
    def literal(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,str):return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub) and isinstance(node.operand,ast.Constant) and type(node.operand.value) is int:
            return -node.operand.value
        if isinstance(node,(ast.Tuple,ast.List)):return tuple(literal(x) for x in node.elts)
        raise ValueError('Only bounded literal parameters accepted')
    def visit(node,depth=0):
        if depth>16:raise ValueError('AST depth exceeded')
        if isinstance(node,ast.Name) and node.id=='x':return Program()
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.keywords or not node.args:
            raise ValueError('Only whitelisted function calls over x allowed')
        if node.func.id not in SCHEMA or node.func.id=='x':raise ValueError('Unknown grid primitive')
        if node.func.id in BINARY:
            if len(node.args)<2:raise ValueError('Binary operation needs two grids')
            return Program(node.func.id,visit(node.args[0],depth+1),tuple(literal(x) for x in node.args[2:]),visit(node.args[1],depth+1))
        return Program(node.func.id,visit(node.args[0],depth+1),tuple(literal(x) for x in node.args[1:]))
    return visit(tree.body)


def apply(op,g,params=()):
    # Public entry also validates parameter types, preventing bypass of the AST parser.
    if op not in SCHEMA or len(params)!=len(SCHEMA[op]):raise ValueError('Invalid operator')
    for k,v in zip(SCHEMA[op],params):validate_param(k,v)
    a=G.array(g)
    if op=='x':return g
    if op=='python_grid':
        from .safe_python import run
        return run(params[0],g)
    if op in ('rot90','rot180','rot270'):return G.checked(np.rot90(a,{'rot90':1,'rot180':2,'rot270':3}[op]))
    if op=='flip_lr':return G.checked(np.fliplr(a))
    if op=='flip_ud':return G.checked(np.flipud(a))
    if op=='transpose':return G.checked(a.T)
    if op=='crop':return G.crop(g,*params)
    if op=='crop_color':
        positions=np.argwhere(a==params[0])
        if not len(positions):raise ValueError('Requested color absent')
        lo=positions.min(0);hi=positions.max(0)+1
        return G.checked(a[lo[0]:hi[0],lo[1]:hi[1]])
    if op=='recolor':return G.checked(np.where(a==params[0],params[1],a))
    if op=='color_map':return G.checked(np.asarray(params[0])[a])
    if op=='scale':
        n=params[0]
        if max(a.shape)*n>30:raise ValueError('Scale exceeds ARC dimensions')
        return G.checked(a.repeat(n,0).repeat(n,1))
    if op=='tile':
        h,w=params
        if a.shape[0]*h>30 or a.shape[1]*w>30:raise ValueError('Tile exceeds ARC dimensions')
        return G.checked(np.tile(a,(h,w)))
    if op=='subsample':return G.checked(a[::params[0],::params[0]])
    if op=='compress':return G.compress(g)
    if op in ('largest','smallest','top_object','bottom_object'):
        return G.select_object(g,{'top_object':'top','bottom_object':'bottom'}.get(op,op),*params)
    if op=='fill_holes':return G.holes(g,*params)
    if op=='connect':return G.connect(g,*params)
    if op=='gravity':return G.gravity(g,*params)
    if op=='complete_symmetry':return G.symmetry_complete(g,*params)
    if op=='periodic':return G.periodic_complete(g,*params)
    if op=='panel':return G.checked(G.panel_pair(g,params[0])[params[1]])
    if op=='overlay':return G.overlay(g,*params)
    if op=='mirror':
        axis,reverse=params;b=np.fliplr(a) if axis=='h' else np.flipud(a)
        halves=(b,a) if reverse else (a,b)
        return G.checked(np.concatenate(halves,axis=1 if axis=='h' else 0))
    if op=='border':return G.checked(np.pad(a,params[1],constant_values=params[0]))
    if op=='trim':
        n=params[0]
        if min(a.shape)<=2*n:raise ValueError('Trim removes entire grid')
        return G.checked(a[n:-n,n:-n])
    if op=='translate':
        dr,dc,bg=params;out=np.full(a.shape,G.resolve_bg(g,bg),dtype=int)
        for r in range(a.shape[0]):
            for c in range(a.shape[1]):
                if 0<=r+dr<a.shape[0] and 0<=c+dc<a.shape[1]:out[r+dr,c+dc]=a[r,c]
        return G.checked(out)
    if op=='mask':
        color,fg,bg=params;return G.checked(np.where(a==color,fg,G.resolve_bg(g,bg)))
    if op=='erase_color':return G.checked(np.where(a==params[0],G.resolve_bg(g,params[1]),a))
    if op=='object_boxes':
        color,bg,filled=params;out=a.copy()
        for ob in G.components(g,bg):
            r,c,h,w=G.bbox(ob)
            if filled:out[r:h,c:w]=color
            else:out[r,c:w]=color;out[h-1,c:w]=color;out[r:h,c]=color;out[r:h,w-1]=color
        return G.checked(out)
    if op=='count_objects':
        color,axis,bg=params;n=len(G.components(g,bg))
        if not 1<=n<=30:raise ValueError('Object count outside output dimension range')
        return G.checked(np.full((1,n) if axis=='h' else (n,1),color))
    if op=='color_histogram':
        axis,bg=params;bg=G.resolve_bg(g,bg);colors=sorted({x for r in g for x in r}-{bg})
        if not colors:raise ValueError('No foreground')
        counts=[sum(v==color for r in g for v in r) for color in colors]
        if max(counts)>30:raise ValueError('Count outside grid size')
        out=np.full((len(colors),max(counts)),bg,dtype=int)
        for i,(c,n) in enumerate(zip(colors,counts)):out[i,:n]=c
        return G.checked(out if axis=='h' else out.T)
    raise ValueError('Unimplemented operation')


def infer_color_map(inputs,targets):
    """Exact consistent symbol substitution inferred solely from demonstrations."""
    mapping={}
    for a,b in zip(inputs,targets):
        if G.shape(a)!=G.shape(b):return None
        for r,s in zip(a,b):
            for x,y in zip(r,s):
                if x in mapping and mapping[x]!=y:return None
                mapping[x]=y
    result=tuple(mapping.get(i,i) for i in range(10))
    return None if result==tuple(range(10)) else result


def vocabulary(examples):
    """Generic primitives with parameters suggested by visible support only.

    Negative candidates remain usable prefixes. This is not a table of ARC solutions.
    """
    inputs=[e.input for e in examples];outputs=[e.output for e in examples]
    colors=sorted({v for g in inputs+outputs for r in g for v in r})
    backgrounds=sorted({-1,0,*[G.background(g) for g in inputs]})
    # Tuple = operation, parameters, representation family. Families reserve beams.
    ops=[]
    def add(name,params=(),view='geometry'):ops.append((name,tuple(params),view))
    for n in ('rot90','rot180','rot270','flip_lr','flip_ud','transpose','compress'):add(n)
    for bg in backgrounds:
        add('crop',(bg,),'objects')
        for conn,multi,canvas in product((4,8),(0,1),(0,1)):
            for name in ('largest','smallest'):add(name,(bg,conn,multi,canvas),'objects')
        for name in ('top_object','bottom_object'):add(name,(bg,4,0,0),'objects')
        for direction in ('up','down','left','right'):add('gravity',(direction,bg),'topology')
        for s in ('h','v','r','d'):add('complete_symmetry',(s,bg),'topology')
        add('periodic',(bg,),'topology')
    for color in colors:
        add('crop_color',(color,),'objects')
        for bg in backgrounds:
            if color==bg:continue
            add('fill_holes',(color,bg),'topology')
            add('connect',(color,'both',bg),'topology')
            for axis in ('h','v'):add('connect',(color,axis,bg),'topology')
            for filled in (0,1):add('object_boxes',(color,bg,filled),'objects')
        for target in colors:
            if color!=target:add('recolor',(color,target),'color')
    for f in (2,3,4,5):add('scale',(f,));add('subsample',(f,))
    for axis in ('h','v'):
        for side in (0,1):add('panel',(axis,side),'panels');add('mirror',(axis,side),'geometry')
        for logic,color in product(('or','and','xor','diff','nor'),colors):
            add('overlay',(axis,logic,color,0),'panels')
    # Size relationships suggest scale/tile; no query labels are consulted.
    factors={(2,2),(3,3),(1,2),(2,1)}
    for x,y in zip(inputs,outputs):
        if len(y)%len(x)==0 and len(y[0])%len(x[0])==0:
            f=(len(y)//len(x),len(y[0])//len(x[0]))
            if all(1<=n<=5 for n in f):factors.add(f)
    for f in sorted(factors):add('tile',f)
    for n in (1,2):add('trim',(n,))
    for color in colors:add('border',(color,1))
    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1),(2,0),(0,2)):
        add('translate',(dr,dc,0),'geometry')
    for c,axis in product(colors,('h','v')):add('count_objects',(c,axis,-1),'count')
    for axis in ('h','v'):add('color_histogram',(axis,-1),'count')
    return tuple(dict.fromkeys(ops))


def apply_binary(op,left,right,params):
    if op not in BINARY or len(params)!=len(SCHEMA[op]):raise ValueError('Invalid binary operator')
    for kind,value in zip(SCHEMA[op],params):validate_param(kind,value)
    a=G.array(left);b=G.array(right);bg=G.resolve_bg(left,params[-1])
    if op=='kronecker':
        if a.shape[0]*b.shape[0]>30 or a.shape[1]*b.shape[1]>30:raise ValueError('Product grid exceeds ARC size')
        mask=np.repeat(np.repeat(a!=bg,b.shape[0],0),b.shape[1],1)
        pattern=np.tile(b,a.shape)
        return G.checked(np.where(mask,pattern,bg))
    if a.shape!=b.shape:raise ValueError('Binary grid shapes differ')
    if op=='merge':
        if np.any((a!=bg)&(b!=bg)&(a!=b)):raise ValueError('Conflicting overlay')
        return G.checked(np.where(a==bg,b,a))
    color=params[0]
    if op=='intersection':return G.checked(np.where((a!=bg)&(b!=bg),color,bg))
    if op=='xor_grids':return G.checked(np.where((a!=bg)^(b!=bg),color,bg))
    if op=='paint_mask':return G.checked(np.where(b!=bg,color,a))
    raise ValueError('Unknown binary operator')


def constant_hole_repairs(program:Program,examples,limit=64):
    """One-literal repairs of a neural program, constrained by visible task symbols.

    This is bounded program synthesis, not fitting neural parameters. A repair must
    still pass the solver's entire demonstration ledger. A wrong complete program
    remains usable as a structural sketch; no hidden test output is consulted.
    """
    if limit<1:return
    palette=sorted({v for ex in examples for g in (ex.input,ex.output) for row in g for v in row})
    count=0;seen={program.id}
    if program.op=='python_grid':
        import copy
        original=ast.parse(program.params[0])
        nodes=[n for n in ast.walk(original) if isinstance(n,ast.Constant) and type(n.value) is int]
        for index,node in enumerate(nodes[:8]):
            for value in palette:
                if value==node.value:continue
                trial=copy.deepcopy(original)
                constants=[n for n in ast.walk(trial) if isinstance(n,ast.Constant) and type(n.value) is int]
                constants[index].value=value
                try:repaired=Program('python_grid',program.child,(ast.unparse(trial),))
                except ValueError:continue
                if repaired.id in seen:continue
                seen.add(repaired.id);yield repaired;count+=1
                if count>=limit:return
    else:
        for index,(kind,value) in enumerate(zip(SCHEMA[program.op],program.params)):
            if kind not in ('color','bg','factor','thickness','bool','shift'):continue
            alternatives=palette if kind in ('color','bg') else tuple(range(0,6))
            for candidate in alternatives:
                if candidate==value:continue
                params=list(program.params);params[index]=candidate
                try:repaired=Program(program.op,program.child,tuple(params),program.other)
                except ValueError:continue
                if repaired.id in seen:continue
                seen.add(repaired.id);yield repaired;count+=1
                if count>=limit:return
