"""Small total, bounded grid primitives. No downloaded solver or solution table.

Objectness, topology, color, geometry and grouping are explicit inductive priors.
Composite programs are inferred from each task's demonstrations by search.
"""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
import numpy as np
from .contracts import Grid, grid

def array(g): return np.asarray(g, dtype=np.int16)
def checked(a): return grid(np.asarray(a).astype(int).tolist())
def shape(g): return (len(g), len(g[0]))
def background(g):
    counts=Counter(x for r in g for x in r)
    return min(counts, key=lambda c:(-counts[c], c))
def resolve_bg(g, value=-1): return background(g) if value == -1 else value

def components(g, bg=-1, connectivity=4, multicolor=False):
    a=array(g); bg=resolve_bg(g,bg); seen=set(); objects=[]
    offsets=((-1,0),(1,0),(0,-1),(0,1))
    if connectivity==8:offsets+=((-1,-1),(-1,1),(1,-1),(1,1))
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if (r,c) in seen or a[r,c]==bg:continue
            cells=[]; queue=deque([(r,c)]); seen.add((r,c)); color=int(a[r,c])
            while queue:
                i,j=queue.popleft();cells.append((i,j,int(a[i,j])))
                for dr,dc in offsets:
                    x,y=i+dr,j+dc
                    if 0<=x<a.shape[0] and 0<=y<a.shape[1] and (x,y) not in seen:
                        if a[x,y]!=bg and (multicolor or a[x,y]==color):
                            seen.add((x,y));queue.append((x,y))
            objects.append(tuple(cells))
    return tuple(objects)

def bbox(cells):
    if not cells:raise ValueError('Empty object')
    return min(x[0] for x in cells), min(x[1] for x in cells), max(x[0] for x in cells)+1,max(x[1] for x in cells)+1

def object_grid(g,cells,bg=-1):
    r,c,h,w=bbox(cells);out=np.full((h-r,w-c),resolve_bg(g,bg),dtype=int)
    for i,j,v in cells:out[i-r,j-c]=v
    return checked(out)

def crop(g,bg=-1):
    a=array(g);points=np.argwhere(a!=resolve_bg(g,bg))
    if not len(points):return g
    lo=points.min(0);hi=points.max(0)+1
    return checked(a[lo[0]:hi[0],lo[1]:hi[1]])

def select_object(g,which='largest',bg=-1,conn=4,multicolor=0,canvas=0):
    objs=components(g,bg,conn,bool(multicolor))
    if not objs:return g
    if which=='largest':chosen=min(objs,key=lambda x:(-len(x),bbox(x)))
    elif which=='smallest':chosen=min(objs,key=lambda x:(len(x),bbox(x)))
    elif which=='top':chosen=min(objs,key=lambda x:bbox(x))
    elif which=='bottom':chosen=max(objs,key=lambda x:bbox(x))
    else:raise ValueError('Unknown object selector')
    if not canvas:return object_grid(g,chosen,bg)
    a=np.full(shape(g),resolve_bg(g,bg),dtype=int)
    for r,c,v in chosen:a[r,c]=v
    return checked(a)

def holes(g,color,bg=-1):
    a=array(g);bg=resolve_bg(g,bg);h,w=a.shape;outside=set();queue=deque()
    for i in range(h):
        for j in range(w):
            if (i in (0,h-1) or j in (0,w-1)) and a[i,j]==bg:
                outside.add((i,j));queue.append((i,j))
    while queue:
        i,j=queue.popleft()
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            r,c=i+di,j+dj
            if 0<=r<h and 0<=c<w and a[r,c]==bg and (r,c) not in outside:
                outside.add((r,c));queue.append((r,c))
    for i in range(h):
        for j in range(w):
            if a[i,j]==bg and (i,j) not in outside:a[i,j]=color
    return checked(a)

def connect(g,color,axis='both',bg=-1):
    a=array(g);bg=resolve_bg(g,bg)
    if axis in ('h','both'):
        for row in a:
            pos=np.where(row==color)[0]
            if len(pos)>1:
                for j in range(pos[0],pos[-1]+1):
                    if row[j]==bg:row[j]=color
    if axis in ('v','both'):
        for j in range(a.shape[1]):
            pos=np.where(a[:,j]==color)[0]
            if len(pos)>1:
                for i in range(pos[0],pos[-1]+1):
                    if a[i,j]==bg:a[i,j]=color
    return checked(a)

def gravity(g,direction='down',bg=-1):
    a=array(g);bg=resolve_bg(g,bg)
    if direction in ('left','right'):a=a.T
    for j in range(a.shape[1]):
        values=a[:,j][a[:,j]!=bg];a[:,j]=bg
        if len(values):
            if direction in ('down','right'):a[-len(values):,j]=values
            else:a[:len(values),j]=values
    if direction in ('left','right'):a=a.T
    return checked(a)

def panel_pair(g,axis='h'):
    a=array(g)
    if axis=='v':a=a.T
    n=a.shape[1]
    if n%2==0:left,right=a[:,:n//2],a[:,n//2:]
    else:
        mid=n//2
        if len(set(a[:,mid].tolist()))!=1:raise ValueError('Not a uniform separator')
        left,right=a[:,:mid],a[:,mid+1:]
    if not left.size:raise ValueError('Empty panels')
    return (left.T,right.T) if axis=='v' else (left,right)

def overlay(g,axis,logic,color,bg=-1):
    bg=resolve_bg(g,bg);a,b=panel_pair(g,axis)
    pa,pb=a!=bg,b!=bg
    mask={'or':pa|pb,'and':pa&pb,'xor':pa^pb,'diff':pa&~pb,'nor':~(pa|pb)}[logic]
    return checked(np.where(mask,color,bg))

def symmetry_complete(g,axis='h',bg=-1):
    a=array(g);bg=resolve_bg(g,bg)
    if axis=='h':b=np.fliplr(a)
    elif axis=='v':b=np.flipud(a)
    elif axis=='r':b=np.rot90(a,2)
    elif axis=='d':
        if a.shape[0]!=a.shape[1]:raise ValueError('Diagonal symmetry requires square')
        b=a.T
    else:raise ValueError('Unknown symmetry')
    if np.any((a!=bg)&(b!=bg)&(a!=b)):raise ValueError('Conflicting symmetric evidence')
    return checked(np.where(a==bg,b,a))

def compress(g):
    a=array(g);a=a[np.r_[True,np.any(a[1:]!=a[:-1],axis=1)]]
    a=a[:,np.r_[True,np.any(a[:,1:]!=a[:,:-1],axis=0)]]
    return checked(a)

def periodic_complete(g,bg=-1):
    a=array(g);bg=resolve_bg(g,bg);h,w=a.shape
    choices=[]
    for ph in range(1,h+1):
        for pw in range(1,w+1):
            if ph*pw>=h*w:continue
            block=np.full((ph,pw),-1,dtype=int);ok=True
            for i,j in np.argwhere(a!=bg):
                v=a[i,j];r,c=i%ph,j%pw
                if block[r,c] not in (-1,v):ok=False;break
                block[r,c]=v
            if ok and (block>=0).all():choices.append((ph*pw,ph,pw,block))
    if not choices:raise ValueError('No fully specified periodic motif')
    _,ph,pw,b=min(choices,key=lambda x:x[:3])
    return checked(np.tile(b,((h+ph-1)//ph,(w+pw-1)//pw))[:h,:w])

def summarize(g):
    counts=Counter(x for row in g for x in row);bg=background(g)
    objs=components(g,bg)
    return {'shape':shape(g),'background_hypothesis':bg,'colors':dict(sorted(counts.items())),
            'objects4':[{'area':len(x),'box':bbox(x),'colors':sorted({p[2] for p in x})} for x in objs[:32]],
            'symmetric_lr':g==checked(np.fliplr(array(g))), 'symmetric_ud':g==checked(np.flipud(array(g)))}
