"""Demonstration-fitting DAG predictions, then distinct pinned-native attempts.
SPDX-License-Identifier: MIT-0; embedded native solver retains MIT.
"""
import hashlib
import json
import time
from pathlib import Path
from icecuber_entry import predict as native_predict, valid
from solver_deferred import search


def select(task, candidates, native):
    """Use two distinct grids per query, with no labels or task-ID routing."""
    assert len(native) == len(task['test'])
    answers = []
    for qi, query in enumerate(task['test']):
        grids = []
        ordered = [c['predictions'][qi] for c in candidates]
        ordered += [native[qi]['attempt_1'], native[qi]['attempt_2']]
        for grid in ordered:
            assert valid(grid)
            if grid not in grids:
                grids.append(grid)
            if len(grids) == 2:
                break
        while len(grids) < 2:
            grids.append(query['input'])
        answers.append({'attempt_1': grids[0], 'attempt_2': grids[1]})
    return answers


def predict(challenges, native_root, depth=3, timeout=1500):
    blind = {name: {'train': task['train'],
                    'test': [{'input': q['input']} for q in task['test']]}
             for name, task in sorted(challenges.items())}
    started = time.perf_counter()
    native, native_receipt = native_predict(blind, native_root, depth=2, timeout=timeout)
    assert native_receipt['exit_code'] == 0 and not native_receipt['missing_queries'], native_receipt
    submission = {}
    rows = []
    for name, task in blind.items():
        result = search(task, depth=depth, terminal_map=True)
        answers = select(task, result['candidates'], native[name])
        submission[name] = answers
        rows.append({'task_id': name, 'candidate_count': len(result['candidates']),
                     'candidates': result['candidates'], 'dag_seconds': result['seconds'],
                     'dag_metrics': result['metrics'], 'native_attempts': native[name],
                     'attempts': answers})
    return submission, {'complete': True, 'depth': depth, 'tasks': len(blind),
                        'queries': sum(len(t['test']) for t in blind.values()),
                        'native_receipt': native_receipt, 'rows': rows,
                        'wall_seconds': time.perf_counter() - started,
                        'policy': 'Ranked DAG candidate grids first; fill remaining distinct slots from reset native depth2. Input fallback only for empty slots. No query-label access.'}

