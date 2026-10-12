"""Original depth3 plus panel-depth2 fitting programs, then native depth2.
SPDX-License-Identifier: MIT-0; native solver retains MIT.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import time
import numpy as np
from solver import pack
from solver_panels import operations
from solver_deferred import search
from portfolio_entry import select
from icecuber_entry import predict as native_predict


def merge(task, original, panels):
    indices = {op.name:i for i,op in enumerate(operations(task['train']))}
    candidates = {}
    for c in original + panels:
        terminal = bool(c['program'] and c['program'][-1].startswith('fit_color_map:'))
        names = c['program'][:-1] if terminal else c['program']
        path = tuple(indices[n] for n in names)
        rank = (len(c['program']),path,int(terminal))
        grids = tuple(pack(np.array(g,dtype=np.uint8)) for g in c['predictions'])
        if grids not in candidates or rank < candidates[grids][0]: candidates[grids] = (rank,c)
    return [v[1] for k,v in sorted(candidates.items(),key=lambda kv:(kv[1][0],kv[0]))]


def dag_one(item):
    name,task = item
    original = search(task,3,terminal_map=True)
    panels = search(task,2,ops_override=operations(task['train']),terminal_map=True)
    return name,original,panels,merge(task,original['candidates'],panels['candidates'])


def predict(challenges, native_root, timeout=1500, workers=2):
    blind = {name:{'train':t['train'],'test':[{'input':q['input']} for q in t['test']]}
             for name,t in sorted(challenges.items())}
    started = time.perf_counter()
    native,native_receipt = native_predict(blind,native_root,depth=2,timeout=timeout)
    assert native_receipt['exit_code'] == 0 and not native_receipt['missing_queries']
    rows = [];submission = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name,original,panels,candidates in pool.map(dag_one,blind.items(),chunksize=1):
            submission[name] = select(blind[name],candidates,native[name])
            rows.append({'task_id':name,'original_candidates':original['candidates'],
                         'panel_candidates':panels['candidates'],'candidates':candidates,
                         'native_attempts':native[name],'attempts':submission[name]})
    return submission,{'complete':True,'tasks':len(blind),'queries':sum(len(t['test']) for t in blind.values()),
                       'native_receipt':native_receipt,'rows':rows,'wall_seconds':time.perf_counter()-started,
                       'workers':workers,'policy':'Original-depth3 and panel-depth2 candidate union, ranked by the combined grammar shortest program/path order, distinct per-query predictions first then reset-native depth2. No task-ID/query-label routing.'}
