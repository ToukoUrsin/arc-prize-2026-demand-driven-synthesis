"""Frozen, pure unary grid DSL and query-aware breadth-first synthesis.

SPDX-License-Identifier: MIT-0
Inference consumes demonstration outputs and query inputs, never query labels.
All grid states must have dimensions 1..30. Undefined operations are absorbing.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import NamedTuple
import time
import numpy as np


def pack(a):
    return bytes(a.shape) + a.astype(np.uint8, copy=False).tobytes()


def unpack(s):
    return np.frombuffer(s[2:], dtype=np.uint8).reshape(s[0], s[1])


def background(a):
    return int(np.bincount(a.ravel(), minlength=10).argmax())


def bbox(a, mask):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return None
    return a[yy.min():yy.max()+1, xx.min():xx.max()+1]


@lru_cache(maxsize=128)
def components(s, bg, same_color):
    a = unpack(s)
    remaining = set(zip(*np.nonzero(a != bg)))
    out = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            y, x = stack.pop()
            for q in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if q in remaining and (not same_color or a[q] == a[start]):
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        out.append(tuple(sorted(cells)))
    return tuple(sorted(out, key=lambda c: (-len(c), c)))


def select_object(a, bg, same, smallest, crop):
    cs = components(pack(a), bg, same)
    if not cs:
        return None
    c = min(cs, key=lambda c: (len(c), c)) if smallest else cs[0]
    out = np.full_like(a, bg)
    for pos in c:
        out[pos] = a[pos]
    if crop:
        yy, xx = zip(*c)
        out = out[min(yy):max(yy)+1, min(xx):max(xx)+1]
    return out


def reduce_blocks(a, k):
    h, w = a.shape
    if h % k or w % k:
        return None
    blocks = a.reshape(h//k, k, w//k, k)
    out = blocks[:, 0, :, 0]
    if not np.all(blocks == out[:, None, :, None]):
        return None
    return out


def compress(a, axis):
    if axis == 0:
        keep = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
        return a[keep]
    return compress(a.T, 0).T


def remove_lines(a):
    rows = ~np.all(a == a[:, :1], axis=1)
    cols = ~np.all(a == a[:1, :], axis=0)
    return a[rows][:, cols]


def half(a, axis, side, gap):
    n = a.shape[axis]
    if (n - gap) % 2:
        return None
    k = (n-gap)//2
    if axis == 0:
        return a[:k] if side == 0 else a[k+gap:]
    return a[:, :k] if side == 0 else a[:, k+gap:]


class Operation(NamedTuple):
    name: str
    fn: object


def operations(train):
    """Fixed task grammar. Parameters depend only on demonstration inputs/labels."""
    ops = []
    def add(name, fn):
        ops.append(Operation(name, fn))
    for k in (1,2,3):
        add(f'rotate{k*90}', lambda a,k=k: np.rot90(a,k))
    add('flip_lr', np.fliplr)
    add('flip_ud', np.flipud)
    add('transpose', lambda a:a.T)
    add('anti_transpose', lambda a:np.rot90(a.T,2))
    add('crop_nonzero', lambda a:bbox(a,a!=0))
    add('crop_nonbackground', lambda a:bbox(a,a!=background(a)))
    add('compress_rows', lambda a:compress(a,0))
    add('compress_columns', lambda a:compress(a,1))
    add('remove_uniform_lines', remove_lines)
    for k in (2,3):
        add(f'upscale{k}',lambda a,k=k:np.repeat(np.repeat(a,k,0),k,1))
        add(f'downscale{k}',lambda a,k=k:reduce_blocks(a,k))
        add(f'tile{k}x{k}',lambda a,k=k:np.tile(a,(k,k)))
    for axis in (0,1):
        add(f'repeat_axis{axis}',lambda a,axis=axis:np.concatenate((a,a),axis=axis))
        add(f'mirror_axis{axis}',lambda a,axis=axis:np.concatenate((a,np.flip(a,axis)),axis=axis))
        for side in (0,1):
            for gap in (0,1):
                add(f'half{axis}_{side}_gap{gap}',lambda a,axis=axis,side=side,gap=gap:half(a,axis,side,gap))
    colors = sorted({c for t in train for row in t['input'] for c in row})
    for c in colors:
        add(f'crop_color{c}',lambda a,c=c:bbox(a,a==c))
        add(f'keep_color{c}',lambda a,c=c:np.where(a==c,a,0))
        add(f'erase_color{c}',lambda a,c=c:np.where(a==c,0,a))
    for bgmode in ('zero','mode'):
        for same in (False,True):
            for small in (False,True):
                for crop in (False,True):
                    add(f'object_{bgmode}_same{int(same)}_small{int(small)}_crop{int(crop)}',
                        lambda a,bgmode=bgmode,same=same,small=small,crop=crop:
                        select_object(a,0 if bgmode=='zero' else background(a),same,small,crop))
    # Infer a single consistent color map from demonstrations of equal shape.
    mapping, consistent = {}, True
    for pair in train:
        a,b = np.array(pair['input']),np.array(pair['output'])
        if a.shape != b.shape:
            consistent = False
            break
        for x,y in zip(a.ravel(),b.ravel()):
            x,y = int(x),int(y)
            if x in mapping and mapping[x] != y:
                consistent = False
                break
            mapping[x] = y
    if consistent:
        lut = np.array([mapping.get(c,c) for c in range(10)],dtype=np.uint8)
        add('demonstration_color_map',lambda a,lut=lut:lut[a])
    return ops


def apply(op, values):
    out = []
    for s in values:
        try:
            a = op.fn(unpack(s))
        except (ValueError,IndexError):
            return None
        if a is None or a.ndim != 2 or min(a.shape)<1 or max(a.shape)>30:
            return None
        out.append(pack(a))
    return tuple(out)


def search(task, mode='query', depth=2, max_states=0, ops_override=None):
    """Collect goals *before* merging, preserving final-step alternatives in both arms.

    max_states=0 is exhaustive quotient search. A positive cap is approximate.
    Grammar state is only remaining depth, and BFS stores minimum-depth states.
    """
    started = time.perf_counter()
    train, queries = task['train'], task['test']
    n = len(train)
    initial = tuple(pack(np.array(p['input'],dtype=np.uint8)) for p in train+queries)
    goal = tuple(pack(np.array(p['output'],dtype=np.uint8)) for p in train)
    ops = operations(train) if ops_override is None else ops_override
    signature = (lambda v:v[:n]) if mode=='train' else (lambda v:v)
    seen = {signature(initial): (initial, ())}
    frontier = [(initial,())]
    solutions, witnesses = {}, []
    metrics = Counter()
    def collect(v,path):
        if v[:n] == goal:
            predictions = v[n:]
            if predictions not in solutions:
                solutions[predictions] = path
    collect(initial,())
    layers=[]
    for d in range(1,depth+1):
        next_frontier=[]
        for values,path in frontier:
            for i,op in enumerate(ops):
                metrics['expansions'] += 1
                v = apply(op,values)
                if v is None:
                    metrics['undefined'] += 1
                    continue
                p = path+(i,)
                collect(v,p)
                sig = signature(v)
                if sig in seen:
                    metrics['merged'] += 1
                    old,oldpath=seen[sig]
                    if old[n:] != v[n:]:
                        metrics['query_divergent_merges'] += 1
                        if len(witnesses)<3:
                            witnesses.append({'kept':[ops[j].name for j in oldpath],
                                              'discarded':[ops[j].name for j in p],
                                              'at_depth':d})
                    continue
                # Final-depth states need not be stored or deduplicated: all goals
                # have already been collected. Avoid quadratic memory for evidence.
                if d < depth:
                    if max_states and len(next_frontier)>=max_states:
                        metrics['capped'] += 1
                        continue
                    seen[sig]=(v,p)
                    next_frontier.append((v,p))
        layers.append({'depth':d,'frontier':len(frontier),'next':len(next_frontier)})
        frontier=next_frontier
    # Deterministic shortest-program, then grammar-order tie break. No labels used.
    candidates=sorted(solutions.items(),key=lambda kv:(len(kv[1]),kv[1],kv[0]))
    out=[]
    for pred,path in candidates:
        out.append({'predictions':[unpack(s).tolist() for s in pred],
                    'program':[ops[i].name for i in path]})
    return {'candidates':out,'metrics':dict(metrics),'layers':layers,
            'merge_witnesses':witnesses,'num_operations':len(ops),
            'seconds':time.perf_counter()-started,'mode':mode,'depth':depth,
            'max_states':max_states}


def solve(task, mode='query', depth=2, max_states=0):
    """Kaggle-style two attempts per query, deduplicated independently per input."""
    result=search(task,mode,depth,max_states)
    answer=[]
    for i,q in enumerate(task['test']):
        grids=[]
        for candidate in result['candidates']:
            grid=candidate['predictions'][i]
            if grid not in grids:
                grids.append(grid)
            if len(grids)==2:
                break
        while len(grids)<2:
            grids.append(q['input'])
        answer.append({'attempt_1':grids[0],'attempt_2':grids[1]})
    return answer,result
