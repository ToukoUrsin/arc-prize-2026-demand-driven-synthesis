"""Fresh native + worker solvers. Two disclosed, uniform development policies.
SPDX-License-Identifier: MIT-0
"""
from concurrent.futures import ProcessPoolExecutor
import resource
import time
from solver_relational_v3 import clean_task, solve as relational_solve
from solver_periodic_v2 import operations as periodic_operations
from solver_deferred import search
from diversity_selection import select as diversity_select
from panel_portfolio_entry import predict as panel_predict
from icecuber_entry import predict as native_predict


def extra_one(item):
    name, task, baseline, profile = item
    started = time.process_time()
    relational = relational_solve(task, 'deferred')
    periodic = None
    if profile == 'conservative':
        candidates = relational['candidates']
        use = bool(candidates and candidates[0]['loo_exact'] == len(task['train']))
        answers = []
        for i, previous in enumerate(baseline):
            grids = [candidates[0]['predictions'][i]] if use else []
            for key in ('attempt_1', 'attempt_2'):
                if previous[key] not in grids:
                    grids.append(previous[key])
            while len(grids) < 2:
                grids.append(previous['attempt_1'])
            answers.append(dict(zip(('attempt_1', 'attempt_2'), grids[:2])))
    else:
        periodic = search(task, depth=1, ops_override=periodic_operations(task['train']), terminal_map=False)
        answers = diversity_select(baseline, periodic['candidates'], relational['candidates'], len(task['test']))
    return name, answers, {'task_id': name, 'relational': relational, 'periodic': periodic,
                           'baseline_attempts': baseline, 'attempts': answers,
                           'extra_cpu_seconds': time.process_time() - started}


def predict(challenges, native_root, profile='diversity', timeout=1500, workers=2):
    if profile not in ('conservative', 'diversity'):
        raise ValueError(profile)
    blind = {name: clean_task(task) for name, task in sorted(challenges.items())}
    start = time.perf_counter()
    cpu_start = time.process_time()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    if profile == 'conservative':
        baseline, baseline_receipt = panel_predict(blind, native_root, timeout, workers)
    else:
        baseline, native_receipt = native_predict(blind, native_root, depth=2, timeout=timeout)
        assert native_receipt['exit_code'] == 0 and not native_receipt['missing_queries']
        baseline_receipt = {'complete': True, 'native_receipt': native_receipt}
    baseline_wall = time.perf_counter() - start
    items = [(name, task, baseline[name], profile) for name, task in blind.items()]
    output, rows = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, attempts, row in pool.map(extra_one, items, chunksize=1):
            output[name] = attempts
            rows.append(row)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return output, {'complete': True, 'profile': profile, 'tasks': len(blind),
                    'queries': sum(len(t['test']) for t in blind.values()), 'rows': rows,
                    'baseline_receipt': baseline_receipt, 'baseline_wall_seconds': baseline_wall,
                    'wall_seconds': time.perf_counter() - start,
                    'cpu_seconds': time.process_time() - cpu_start + children.ru_utime + children.ru_stime
                                   - child_start.ru_utime - child_start.ru_stime,
                    'workers': workers, 'baseline_recomputed': True,
                    'policy': 'Conservative: panel/native then exact-LOO relational insertion. Diversity: native then periodic-first; otherwise native attempt1 plus top relational prediction. Profile is explicit, independent of task IDs and labels.'}
