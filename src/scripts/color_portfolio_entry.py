"""Native/panel/copy portfolio followed by isolated color-relation proposals.
SPDX-License-Identifier: MIT-0
"""
from concurrent.futures import ProcessPoolExecutor
import resource
import time
from solver_color_relations import clean_task, solve
from copy_portfolio_entry import predict as copy_predict


def merge(result, baseline, ntrain):
    candidates = result['candidates']
    use = bool(candidates and candidates[0]['loo_exact'] == ntrain)
    if not use:
        return baseline
    output = []
    for i, previous in enumerate(baseline):
        grids = [candidates[0]['predictions'][i]]
        for key in ('attempt_1', 'attempt_2'):
            if previous[key] not in grids:
                grids.append(previous[key])
        while len(grids) < 2:
            grids.append(previous['attempt_1'])
        output.append(dict(zip(('attempt_1', 'attempt_2'), grids[:2])))
    return output


def color_one(item):
    name, task, baseline = item
    blind = clean_task(task)
    result = solve(blind, 'deferred')
    return name, merge(result, baseline, len(blind['train'])), result


def predict(challenges, native_root, profile='color-conservative', timeout=1500, workers=2):
    if profile != 'color-conservative':
        raise ValueError(profile)
    blind = {name: clean_task(task) for name, task in sorted(challenges.items())}
    started = time.perf_counter()
    cpu = time.process_time()
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    baseline, baseline_receipt = copy_predict(blind, native_root, 'copy-conservative', timeout, workers)
    output, rows = {}, []
    jobs = [(name, task, baseline[name]) for name, task in blind.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, attempts, result in pool.map(color_one, jobs, chunksize=1):
            output[name] = attempts
            rows.append({'task_id': name, 'relational': result, 'attempts': attempts})
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return output, {'complete': True, 'profile': profile, 'tasks': len(blind),
                    'queries': sum(len(t['test']) for t in blind.values()),
                    'baseline_receipt': baseline_receipt, 'rows': rows,
                    'wall_seconds': time.perf_counter() - started,
                    'cpu_seconds': time.process_time() - cpu + usage.ru_utime + usage.ru_stime
                                   - children.ru_utime - children.ru_stime,
                    'policy': 'Uniform perfect leave-one-demonstration insertion after native/panel/copy. No query labels or task metadata.'}
