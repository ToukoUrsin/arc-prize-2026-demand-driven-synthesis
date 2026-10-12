"""Exact, successive demonstration refinement of a provenance DAG.
SPDX-License-Identifier: MIT-0

The grammar is constructed from all demonstrations before search. Only evaluation
is staged: first demonstrate potential, then refine the productive graph with
each remaining demonstration, then materialize query values. Every incoming
edge is retained, including paths whose earlier query evaluation would fail.
"""
from collections import Counter
from functools import lru_cache
import time
import numpy as np
from solver import pack, unpack, operations, apply
from solver_extended import fit_map


def search(task, depth=2, ops_override=None, apply_fn=apply, terminal_map=True):
    started = time.perf_counter()
    train = tuple(pack(np.array(p['input'], dtype=np.uint8)) for p in task['train'])
    goal = tuple(pack(np.array(p['output'], dtype=np.uint8)) for p in task['train'])
    queries = tuple(pack(np.array(p['input'], dtype=np.uint8)) for p in task['test'])
    if not train or depth < 0:
        raise ValueError('At least one demonstration and nonnegative depth required')
    ops = operations(task['train']) if ops_override is None else ops_override
    metrics = Counter()
    stages = []

    def prune(layers, targets):
        """Retain precisely ancestors of possible goals on the known examples."""
        fit_cache = {}
        terminals = []
        needed = [set() for _ in layers]
        for d, layer in enumerate(layers):
            for values in layer:
                if values == targets:
                    lut, extra = None, 0
                elif terminal_map and d < depth:
                    if values not in fit_cache:
                        fit_cache[values] = fit_map(values, targets)
                    lut, extra = fit_cache[values], 1
                    if lut is None:
                        continue
                else:
                    continue
                terminals.append((d, values, lut, extra))
                needed[d].add(values)
        for d in range(depth, 0, -1):
            for values in needed[d]:
                for parent, _ in layers[d][values]:
                    needed[d - 1].add(parent)
        metrics['nodes_before_pruning'] += sum(map(len, layers))
        metrics['edges_before_pruning'] += sum(len(e) for layer in layers for e in layer.values())
        kept = [{v: edges for v, edges in layer.items() if v in needed[d]}
                for d, layer in enumerate(layers)]
        stages.append({'demonstrations': len(targets),
                       'nodes_before': sum(map(len, layers)),
                       'nodes_after': sum(map(len, kept)),
                       'edges_after': sum(len(e) for layer in kept for e in layer.values()),
                       'terminals': len(terminals)})
        return kept, terminals

    # The coarse DAG records all first-demonstration-defined syntax paths.
    layers = [{train[:1]: []}]
    for d in range(1, depth + 1):
        current = {}
        for parent in layers[-1]:
            for i, op in enumerate(ops):
                metrics['pivot_expansions'] += 1
                values = apply_fn(op, parent)
                if values is None:
                    metrics['demonstration_undefined'] += 1
                    continue
                if d == depth and values != goal[:1]:
                    continue
                current.setdefault(values, []).append((parent, i))
        layers.append(current)
    layers, terminals = prune(layers, goal[:1])

    # Refine only coarse paths that can reach a coarse goal. Distinct additional
    # demonstration values split a node; all predecessor edges survive the split.
    for n in range(1, len(train)):
        refined = [{train[:n + 1]: []} if layers[0] else {}]
        by_coarse = {train[:n]: [train[:n + 1]]} if layers[0] else {}
        for d in range(1, depth + 1):
            current = {}
            next_by_coarse = {}
            for coarse_values, edges in layers[d].items():
                for coarse_parent, i in edges:
                    for parent in by_coarse.get(coarse_parent, ()):
                        metrics['refinement_expansions'] += 1
                        extra = apply_fn(ops[i], parent[n:])
                        if extra is None:
                            metrics['demonstration_undefined'] += 1
                            continue
                        values = coarse_values + extra
                        if d == depth and values != goal[:n + 1]:
                            continue
                        if values not in current:
                            current[values] = []
                            next_by_coarse.setdefault(coarse_values, []).append(values)
                        current[values].append((parent, i))
            refined.append(current)
            by_coarse = next_by_coarse
        layers, terminals = prune(refined, goal[:n + 1])

    @lru_cache(maxsize=None)
    def materialize(d, values):
        metrics['materialized_nodes'] += 1
        if d == 0:
            return {queries: ()}
        outputs = {}
        for parent, i in layers[d][values]:
            for query_values, path in materialize(d - 1, parent).items():
                metrics['query_expansions'] += 1
                pred = apply_fn(ops[i], query_values)
                if pred is None:
                    metrics['query_undefined'] += 1
                    continue
                new_path = path + (i,)
                if pred not in outputs or new_path < outputs[pred]:
                    outputs[pred] = new_path
        metrics['materialized_query_tuples'] += len(outputs)
        return outputs

    solutions = {}
    for d, values, lut, extra in terminals:
        for pred, path in materialize(d, values).items():
            if lut is not None:
                pred = tuple(pack(lut[unpack(s)]) for s in pred)
                metrics['terminal_query_maps'] += len(pred)
            rank = (d + extra, path, extra)
            if pred not in solutions or rank < solutions[pred][0]:
                names = [ops[i].name for i in path]
                if extra:
                    names += ['fit_color_map:' + ','.join(map(str, lut.tolist()))]
                solutions[pred] = rank, names
    candidates = [{'predictions': [unpack(s).tolist() for s in pred], 'program': info[1]}
                  for pred, info in sorted(solutions.items(), key=lambda kv: (kv[1][0], kv[0]))]
    return {'candidates': candidates, 'metrics': dict(metrics), 'stages': stages,
            'seconds': time.perf_counter() - started, 'depth': depth, 'max_states': 0,
            'mode': 'staged', 'num_operations': len(ops),
            'grammar': 'successive_demonstration_refinement'}
