"""Stream sound goals from an increasingly refined provenance DAG. MIT-0.

Unbounded enumeration has the same ranked semantics as staged_solver. Under
interruption only already validated candidates survive; search completeness
and final ranking over unseen programs are explicitly not claimed.
"""
from collections import Counter
from functools import lru_cache
import time
import numpy as np
from solver import pack, unpack
from solver_extended import fit_map
from adaptive import BudgetExpired


def search(task, depth, ops, evaluator, terminal_map=True):
    start = time.process_time()
    train = tuple(pack(np.asarray(p['input'], dtype=np.uint8)) for p in task['train'])
    goal = tuple(pack(np.asarray(p['output'], dtype=np.uint8)) for p in task['train'])
    queries = tuple(pack(np.asarray(p['input'], dtype=np.uint8)) for p in task['test'])
    if not train or depth < 0:
        raise ValueError('Expected demonstrations and nonnegative depth')
    layers = [{train[:1]: []}]
    refined = [{train: []}] + [{} for _ in range(depth)]
    metrics = Counter()
    solutions = {}
    fit_cache = {}

    def terminal(d, values, targets):
        if values == targets:
            return None, 0
        if not terminal_map or d >= depth:
            return False, 0
        key = (values, targets)
        if key not in fit_cache:
            fit_cache[key] = fit_map(values, targets)
        lut = fit_cache[key]
        return (lut, 1) if lut is not None else (False, 0)

    @lru_cache(maxsize=None)
    def refine(d, coarse):
        # Called only for a node in an already complete coarse layer. Every
        # incoming syntax edge is retained; this memo never sees a growing node.
        metrics['refined_nodes'] += 1
        if d == 0:
            return (train,)
        full_values = {}
        for parent_coarse, i in layers[d][coarse]:
            for parent in refine(d-1, parent_coarse):
                metrics['refinement_expansions'] += 1
                extra = evaluator(ops[i], parent[1:])
                if extra is None:
                    continue
                values = coarse + extra
                full_values.setdefault(values, []).append((parent, i))
        for values, edges in full_values.items():
            assert values not in refined[d]
            refined[d][values] = edges
        return tuple(full_values)

    @lru_cache(maxsize=None)
    def materialize(d, values):
        metrics['materialized_nodes'] += 1
        if d == 0:
            return {queries: ()}
        outputs = {}
        for parent, i in refined[d][values]:
            for query_values, path in materialize(d-1, parent).items():
                metrics['query_expansions'] += 1
                pred = evaluator(ops[i], query_values)
                if pred is None:
                    continue
                path = path + (i,)
                if pred not in outputs or path < outputs[pred]:
                    outputs[pred] = path
        return outputs

    def emit(d, pred, path, lut, extra):
        if lut is not None:
            pred = tuple(pack(lut[unpack(s)]) for s in pred)
            metrics['terminal_query_maps'] += len(pred)
        rank = (d+extra, path, extra)
        if pred not in solutions or rank < solutions[pred][0]:
            names = [ops[i].name for i in path]
            if extra:
                names += ['fit_color_map:' + ','.join(map(str, lut.tolist()))]
            solutions[pred] = (rank, names)
            metrics['candidate_updates'] += 1

    # Refine a terminal edge immediately, rather than waiting for the complete
    # deepest layer. Prefix DAGs are stable; the current layer is never memoized.
    def inspect(d, parent_coarse, i, coarse_value):
        lut, extra = terminal(d, coarse_value, goal[:1])
        if lut is False:
            return
        metrics['coarse_terminal_edges'] += 1
        for parent in refine(d-1, parent_coarse):
            more = evaluator(ops[i], parent[1:])
            metrics['terminal_refinement_expansions'] += 1
            if more is None:
                continue
            full = coarse_value + more
            lut, extra = terminal(d, full, goal)
            if lut is False:
                continue
            metrics['full_terminal_edges'] += 1
            for q, path in materialize(d-1, parent).items():
                metrics['terminal_query_expansions'] += 1
                pred = evaluator(ops[i], q)
                if pred is not None:
                    emit(d, pred, path+(i,), lut, extra)

    complete = True
    try:
        lut, extra = terminal(0, train, goal)
        if lut is not False:
            emit(0, queries, (), lut, extra)
        for d in range(1, depth+1):
            current = {}
            for parent in layers[-1]:
                for i, op in enumerate(ops):
                    metrics['pivot_expansions'] += 1
                    value = evaluator(op, parent)
                    if value is None:
                        continue
                    if d < depth:
                        current.setdefault(value, []).append((parent, i))
                    inspect(d, parent, i, value)
            layers.append(current)
            metrics['completed_layers'] += 1
    except BudgetExpired:
        complete = False
    candidates = [dict(predictions=[unpack(s).tolist() for s in pred], program=info[1])
                  for pred, info in sorted(solutions.items(), key=lambda kv: (kv[1][0], kv[0]))]
    return dict(candidates=candidates, complete=complete, budget_expired=not complete,
                depth=depth, metrics=dict(metrics), cpu_seconds=time.process_time()-start)
