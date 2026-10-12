"""Baseline-preserving anytime depth-three adapter. MIT-0."""
import time
import hashlib
import json
from adaptive import BudgetEvaluator, fill_slots, has_vacancy
from cached_grammar import CachedGrammar
from staged_solver import search as staged
from entry import attempts
from anytime_search import search

EXTRA_BUDGET = 50000
MAX_DEPTH = 3


def solve(task, extra_budget=EXTRA_BUDGET, max_depth=MAX_DEPTH):
    cpu, wall = time.process_time(), time.perf_counter()
    blind = {'train': [{'input':p['input'], 'output':p['output']} for p in task['train']],
             'test': [{'input':p['input']} for p in task['test']]}
    factory = CachedGrammar(blind['train']); factory.clear_cache()
    constructor_cpu = time.process_time()-cpu
    evaluator = BudgetEvaluator(factory.operations, factory.apply)
    base = staged(blind, 2, factory.operations, evaluator, True)
    base_answers = attempts(blind, base['candidates'])
    baseline = dict(candidates=base['candidates'], attempts=base_answers,
        metrics=base['metrics'], stages=base['stages'], work_counts=evaluator.metrics(),
        constructor_metrics=factory.constructor_metrics, fit_counts=factory.fit_counts,
        grammar_sha256=hashlib.sha256(json.dumps([op.name for op in factory.operations],separators=(',',':')).encode()).hexdigest(),
        constructor_cpu_seconds=constructor_cpu, cpu_seconds=time.process_time()-cpu)
    before = evaluator.counts['primitive_grid_evaluations']
    evaluator.ceiling = before + extra_budget
    extension = None
    selected = base_answers
    witnesses = [None]*len(base_answers)
    if extra_budget and has_vacancy(selected):
        extension = search(blind, max_depth, factory.operations, evaluator)
        selected, witnesses = fill_slots(selected, extension['candidates'])
    extra = evaluator.counts['primitive_grid_evaluations']-before
    assert 0 <= extra <= extra_budget
    return dict(baseline=baseline, extension=extension, attempts=selected, witnesses=witnesses,
        extra_budget=extra_budget, max_depth=max_depth, extra_primitive_calls=extra,
        work_counts=evaluator.metrics(), cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.perf_counter()-wall)


def run(challenges):
    import gzip
    from collections import Counter
    from pathlib import Path
    submission, totals = {}, Counter()
    with gzip.open('tasks.jsonl.gz', 'wt') as stream:
        for task_id, task in challenges.items():
            row = solve(task); row['task_id'] = task_id
            submission[task_id] = row['attempts']
            stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
            totals['tasks'] += 1; totals['cpu_seconds'] += row['cpu_seconds']
            totals['primitive_grid_evaluations'] += row['work_counts']['primitive_grid_evaluations']
    Path('submission.json').write_text(json.dumps(submission)+'\n')
    receipt = dict(complete=True, totals=dict(totals), verified=False,
                   extra_budget=EXTRA_BUDGET, max_depth=MAX_DEPTH)
    Path('receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
