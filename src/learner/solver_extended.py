"""Structural search with a demonstration-fitted terminal color map.

SPDX-License-Identifier: MIT-0
This is a separately versioned grammar extension. Structural-prefix equivalence
still preserves query outputs; terminal map fitting is a deterministic function
of the complete demonstration state vector. Unknown colors map to themselves.
"""
from collections import Counter
import time
import numpy as np
from solver import operations,pack,unpack,apply


def fit_map(values,goal):
    if any(v[:2] != g[:2] for v,g in zip(values,goal)):
        return None
    lut=np.arange(10,dtype=np.uint8)
    known={}
    for v,g in zip(values,goal):
        a=np.frombuffer(v[2:],dtype=np.uint8)
        b=np.frombuffer(g[2:],dtype=np.uint8)
        pairs=np.unique(a.astype(np.uint16)*10+b)
        for pair in pairs:
            x,y=divmod(int(pair),10)
            if x in known and known[x]!=y:return None
            known[x]=y;lut[x]=y
    return lut


def search(task,mode='query',depth=3,max_states=0,ops_override=None):
    start=time.perf_counter();train=task['train'];n=len(train)
    initial=tuple(pack(np.array(p['input'],dtype=np.uint8)) for p in train+task['test'])
    goal=tuple(pack(np.array(p['output'],dtype=np.uint8)) for p in train)
    ops=operations(train) if ops_override is None else ops_override
    signature=(lambda v:v[:n]) if mode=='train' else (lambda v:v)
    seen={signature(initial):(initial,())};frontier=[(initial,())]
    solutions={};fit_cache={};metrics=Counter();witnesses=[];layers=[]
    def collect(v,path):
        if v[:n]==goal:
            predictions=v[n:];names=[ops[i].name for i in path]
            rank=(len(path),path,0)
        else:
            if len(path)>=depth:return
            key=v[:n]
            if key not in fit_cache:fit_cache[key]=fit_map(key,goal)
            lut=fit_cache[key]
            if lut is None:return
            predictions=tuple(pack(lut[unpack(s)]) for s in v[n:])
            names=[ops[i].name for i in path]+['fit_color_map:'+','.join(map(str,lut.tolist()))]
            rank=(len(path)+1,path,1)
        if predictions not in solutions or rank<solutions[predictions][0]:
            solutions[predictions]=(rank,names)
    collect(initial,())
    for d in range(1,depth+1):
        next_frontier=[]
        for values,path in frontier:
            for i,op in enumerate(ops):
                metrics['expansions']+=1
                v=apply(op,values)
                if v is None:
                    metrics['undefined']+=1;continue
                p=path+(i,);collect(v,p);sig=signature(v)
                if sig in seen:
                    metrics['merged']+=1;old,oldpath=seen[sig]
                    if old[n:]!=v[n:]:
                        metrics['query_divergent_merges']+=1
                        if len(witnesses)<3:witnesses.append({'kept':[ops[j].name for j in oldpath],'discarded':[ops[j].name for j in p],'at_depth':d})
                    continue
                if d<depth:
                    if max_states and len(next_frontier)>=max_states:
                        metrics['capped']+=1;continue
                    seen[sig]=(v,p);next_frontier.append((v,p))
        layers.append({'depth':d,'frontier':len(frontier),'next':len(next_frontier)})
        frontier=next_frontier
    out=[{'predictions':[unpack(s).tolist() for s in pred],'program':info[1]}
         for pred,info in sorted(solutions.items(),key=lambda kv:(kv[1][0],kv[0]))]
    return {'candidates':out,'metrics':dict(metrics),'layers':layers,'merge_witnesses':witnesses,
            'num_operations':len(ops),'seconds':time.perf_counter()-start,'mode':mode,
            'depth':depth,'max_states':max_states,'grammar':'structural_with_terminal_color_map'}


def solve(task,mode='query',depth=3,max_states=0):
    result=search(task,mode,depth,max_states)
    answers=[]
    for i,q in enumerate(task['test']):
        grids=[]
        for c in result['candidates']:
            grid=c['predictions'][i]
            if grid not in grids:grids.append(grid)
            if len(grids)==2:break
        while len(grids)<2:grids.append(q['input'])
        answers.append({'attempt_1':grids[0],'attempt_2':grids[1]})
    return answers,result
