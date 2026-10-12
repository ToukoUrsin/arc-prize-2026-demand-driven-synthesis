"""Monotone attempt extension using bounded, exact deeper searches. MIT-0."""
from collections import Counter
import hashlib
import json
import time
from grid_cache import GridEvaluator
from cached_grammar import CachedGrammar
from packed_2d import PackedGrammar
from staged_solver import search as staged
from solver_deferred import search as deferred
from entry import attempts

EXTRA_BUDGET = 50000
MAX_DEPTH = 4

class BudgetExpired(Exception):
    pass

class BudgetEvaluator(GridEvaluator):
    def __init__(self, operations, apply_fn):
        super().__init__(operations, apply_fn, True, 200000)
        self.ceiling = None

    def __call__(self, op, values):
        result = []
        index = self.operation_ids[id(op)]
        for packed in values:
            # Check before evaluation, including undefined outcomes. Cached
            # lookups remain free but cannot add another primitive execution.
            if (self.ceiling is not None and (index, packed) not in self.cache
                and self.counts['primitive_grid_evaluations'] >= self.ceiling):
                raise BudgetExpired()
            output = super().__call__(op, (packed,))
            if output is None:
                return None
            result.append(output[0])
        return tuple(result)


def fill_slots(prior, candidates):
    """Keep all existing grids; canonical candidates fill duplicate slots only."""
    result = []
    witnesses = []
    for qi, answer in enumerate(prior):
        first, second = answer['attempt_1'], answer['attempt_2']
        witness = None
        if first == second:
            for candidate in candidates:
                proposed = candidate['predictions'][qi]
                if proposed != first:
                    second = proposed
                    witness = candidate['program']
                    break
        result.append({'attempt_1': first, 'attempt_2': second})
        witnesses.append(witness)
    return result, witnesses


def has_vacancy(answers):
    return any(a['attempt_1'] == a['attempt_2'] for a in answers)


def solve(task, engine='staged', grammar='objects', depth=2,
          extra_budget=EXTRA_BUDGET, max_depth=MAX_DEPTH):
    cpu, wall = time.process_time(), time.perf_counter()
    blind = {'train': [{'input': p['input'], 'output': p['output']} for p in task['train']],
             'test': [{'input': p['input']} for p in task['test']]}
    factory = (CachedGrammar if grammar == 'objects' else PackedGrammar)(blind['train'])
    factory.clear_cache()
    constructor_cpu = time.process_time() - cpu
    evaluator = BudgetEvaluator(factory.operations, factory.apply)
    search = {'staged': staged, 'deferred': deferred}[engine]
    base = search(blind, depth, factory.operations, evaluator, True)
    base_answers = attempts(blind, base['candidates'])
    baseline = dict(candidates=base['candidates'], attempts=base_answers,
        metrics=base['metrics'], stages=base.get('stages'), work_counts=evaluator.metrics(),
        constructor_metrics=getattr(factory,'constructor_metrics',{}),
        fit_counts=getattr(factory,'fit_counts',{}),
        grammar_sha256=hashlib.sha256(json.dumps([op.name for op in factory.operations],separators=(',',':')).encode()).hexdigest(),
        constructor_cpu_seconds=constructor_cpu, cpu_seconds=time.process_time()-cpu)
    start_count = evaluator.counts['primitive_grid_evaluations']
    evaluator.ceiling = start_count + extra_budget
    selected = base_answers
    levels = []
    for next_depth in range(depth+1, max_depth+1):
        if not extra_budget or not has_vacancy(selected):
            break
        level_cpu = time.process_time()
        before = dict(evaluator.counts)
        try:
            result = search(blind, next_depth, factory.operations, evaluator, True)
        except BudgetExpired:
            levels.append(dict(depth=next_depth, complete=False, budget_expired=True,
                cpu_seconds=time.process_time()-level_cpu,
                work_counts=dict(evaluator.counts-Counter(before)), attempts=selected))
            break
        selected, witnesses = fill_slots(selected, result['candidates'])
        levels.append(dict(depth=next_depth, complete=True, budget_expired=False,
            cpu_seconds=time.process_time()-level_cpu,
            work_counts=dict(evaluator.counts-Counter(before)), attempts=selected,
            witnesses=witnesses, candidates=result['candidates'], metrics=result['metrics'],
            stages=result.get('stages')))
    calls = evaluator.counts['primitive_grid_evaluations']-start_count
    assert calls <= extra_budget
    return dict(attempts=selected, baseline=baseline, levels=levels,
        fallback_requested=bool(levels), budget_expired=any(x['budget_expired'] for x in levels),
        extra_primitive_calls=calls, extra_budget=extra_budget, max_depth=max_depth,
        work_counts=evaluator.metrics(), cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.perf_counter()-wall, engine=engine, grammar=grammar)


def run(challenges, engine='staged', grammar='objects', depth=2):
    import gzip
    from pathlib import Path
    submission, totals = {}, Counter()
    with gzip.open('tasks.jsonl.gz', 'wt') as stream:
        for task_id, task in challenges.items():
            row = solve(task, engine, grammar, depth)
            submission[task_id] = row['attempts']
            row['task_id'] = task_id
            stream.write(json.dumps(row,separators=(',',':'))+'\n'); stream.flush()
            totals['tasks'] += 1
            totals['primitive_calls'] += row['work_counts']['primitive_grid_evaluations']
            totals['extra_primitive_calls'] += row['extra_primitive_calls']
            totals['cpu_seconds'] += row['cpu_seconds']
    Path('submission.json').write_text(json.dumps(submission)+'\n')
    receipt=dict(complete=True,totals=dict(totals),engine=engine,grammar=grammar,
                 depth=depth,extra_budget=EXTRA_BUDGET,max_depth=MAX_DEPTH,verified=False)
    Path('receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
