"""Portable object grammar / exact engine adapter. MIT-0."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
from grid_cache import GridEvaluator
from relational_grammar import RelationalGrammar
from lazy_grammar import LazyGrammar
from cached_grammar import CachedGrammar
import constructors  # Load source during setup, before guarded inference.
from packed_2d import PackedGrammar
from staged_solver import search as staged
from solver_deferred import search as deferred


def attempts(task, candidates):
    output = []
    for qi, query in enumerate(task['test']):
        distinct = []
        for candidate in candidates:
            grid = candidate['predictions'][qi]
            if grid not in distinct: distinct.append(grid)
            if len(distinct) == 2: break
        while len(distinct) < 2: distinct.append(query['input'])
        output.append(dict(zip(('attempt_1', 'attempt_2'), distinct)))
    return output


def solve(task, engine='staged', grammar='objects', depth=2, backend='cached'):
    cpu, wall = time.process_time(), time.perf_counter()
    blind = {'train': [{'input': p['input'], 'output': p['output']} for p in task['train']],
             'test': [{'input': p['input']} for p in task['test']]}
    factory = ({'lazy':LazyGrammar,'cached':CachedGrammar,'original':RelationalGrammar}[backend] if grammar == 'objects' else PackedGrammar)(blind['train'])
    factory.clear_cache()
    constructor_cpu = time.process_time()-cpu
    evaluator = GridEvaluator(factory.operations, factory.apply, True, 200000)
    result = {'staged': staged, 'deferred': deferred}[engine](
        blind, depth, factory.operations, evaluator, True)
    return dict(candidates=result['candidates'], attempts=attempts(blind, result['candidates']),
                metrics=result['metrics'], stages=result.get('stages'),
                work_counts=evaluator.metrics(), num_operations=len(factory.operations),
                screening_grid_evaluations=getattr(factory,'screening_grid_evaluations',0),
                proposed_operations=getattr(factory,'proposed_operations',0),
                admitted_operations=getattr(factory,'admitted_operations',0),
                learned_operations=getattr(factory,'learned_operations',0),
                constructor_metrics=getattr(factory,'constructor_metrics',{}),
                grammar_sha256=hashlib.sha256(json.dumps([op.name for op in factory.operations],separators=(',',':')).encode()).hexdigest(),
                feature_counts=getattr(factory,'feature_counts',{}),
                fit_counts=getattr(factory,'fit_counts',{}),
                constructor_cpu_seconds=constructor_cpu,
                cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall)


def run(challenges, engine='staged', grammar='objects', depth=2):
    submission, totals = {}, Counter()
    with gzip.open('tasks.jsonl.gz', 'wt') as stream:
        for task_id, task in challenges.items():
            row = solve(task, engine, grammar, depth)
            submission[task_id] = row['attempts']
            row['task_id'] = task_id
            stream.write(json.dumps(row, separators=(',', ':'))+'\n'); stream.flush()
            totals['tasks'] += 1
            totals['primitive_calls'] += row['work_counts']['primitive_grid_evaluations']
            totals['cpu_seconds'] += row['cpu_seconds']
    Path('submission.json').write_text(json.dumps(submission)+'\n')
    receipt = dict(complete=True, totals=dict(totals), engine=engine, grammar=grammar,
                   depth=depth, verified=False)
    Path('receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt
