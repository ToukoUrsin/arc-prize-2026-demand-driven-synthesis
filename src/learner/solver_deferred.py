"""Layered demonstration DAG with demand-driven query evaluation.
SPDX-License-Identifier: MIT-0
Pure fixed unary grammars only. No beam, syntax-dependent semantics or cost.
Each training-equivalent node retains all predecessor edges; no representative
prefix is discarded. Query vectors are materialized only for goal ancestors.
"""
from collections import Counter
from functools import lru_cache
import time
import numpy as np
from solver import pack,unpack,operations,apply
from solver_extended import fit_map


def search(task,depth=2,ops_override=None,apply_fn=apply,terminal_map=True):
    started=time.perf_counter()
    train=tuple(pack(np.array(p['input'],dtype=np.uint8)) for p in task['train'])
    queries=tuple(pack(np.array(p['input'],dtype=np.uint8)) for p in task['test'])
    goal=tuple(pack(np.array(p['output'],dtype=np.uint8)) for p in task['train'])
    ops=operations(task['train']) if ops_override is None else ops_override
    layers=[{train:[]}];metrics=Counter();terminals=[];fit_cache={}

    def terminal(d,values):
        if values==goal:return None,0
        if not terminal_map or d>=depth:return False,0
        if values not in fit_cache:fit_cache[values]=fit_map(values,goal)
        lut=fit_cache[values]
        return (lut,1) if lut is not None else (False,0)

    for d in range(depth+1):
        if d:
            current={}
            for parent in layers[-1]:
                for i,op in enumerate(ops):
                    metrics['demonstration_expansions']+=1
                    values=apply_fn(op,parent)
                    if values is None:
                        metrics['demonstration_undefined']+=1;continue
                    # Final layer requires only training-fitting terminal nodes.
                    if d==depth and values!=goal:continue
                    current.setdefault(values,[]).append((parent,i))
            layers.append(current)
        for values in layers[d]:
            lut,extra=terminal(d,values)
            if lut is not False:terminals.append((d,values,lut,extra))
        metrics['training_nodes']+=len(layers[d])
        metrics['provenance_edges']+=sum(map(len,layers[d].values()))

    @lru_cache(maxsize=None)
    def materialize(d,values):
        metrics['materialized_nodes']+=1
        if d==0:return {queries:()}
        outputs={}
        for parent,i in layers[d][values]:
            for q,path in materialize(d-1,parent).items():
                metrics['query_expansions']+=1
                pred=apply_fn(ops[i],q)
                if pred is None:
                    metrics['query_undefined']+=1;continue
                p=path+(i,)
                if pred not in outputs or p<outputs[pred]:outputs[pred]=p
        metrics['materialized_query_tuples']+=len(outputs)
        return outputs

    solutions={}
    for d,values,lut,extra in terminals:
        for pred,path in materialize(d,values).items():
            if lut is not None:
                pred=tuple(pack(lut[unpack(s)]) for s in pred)
                metrics['terminal_query_maps']+=len(pred)
            rank=(d+extra,path,extra)
            if pred not in solutions or rank<solutions[pred][0]:
                names=[ops[i].name for i in path]
                if extra:names+=['fit_color_map:'+','.join(map(str,lut.tolist()))]
                solutions[pred]=(rank,names)
    candidates=[{'predictions':[unpack(s).tolist() for s in pred],'program':info[1]}
                for pred,info in sorted(solutions.items(),key=lambda kv:(kv[1][0],kv[0]))]
    return {'candidates':candidates,'metrics':dict(metrics),'seconds':time.perf_counter()-started,
            'depth':depth,'mode':'deferred','max_states':0,'num_operations':len(ops),
            'grammar':'layered_demonstration_dag_terminal_map' if terminal_map else 'layered_demonstration_dag',
            'layers':[{'depth':d,'nodes':len(nodes)} for d,nodes in enumerate(layers)]}


def solve(task,depth=2,ops_override=None,apply_fn=apply,terminal_map=True):
    result=search(task,depth,ops_override,apply_fn,terminal_map);answers=[]
    for i,q in enumerate(task['test']):
        grids=[]
        for candidate in result['candidates']:
            grid=candidate['predictions'][i]
            if grid not in grids:grids.append(grid)
            if len(grids)==2:break
        while len(grids)<2:grids.append(q['input'])
        answers.append({'attempt_1':grids[0],'attempt_2':grids[1]})
    return answers,result
