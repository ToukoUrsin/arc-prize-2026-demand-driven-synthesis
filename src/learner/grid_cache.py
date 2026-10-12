"""Task-local, bounded memoization of pure unary grid transformations.
SPDX-License-Identifier: MIT-0

The cache stores packed immutable results, including undefined results. It never
merges search nodes, chooses programs, or reads labels. Operations are keyed by
identity, not name: same-name functions may implement different transformations.
"""
from collections import Counter


class GridEvaluator:
    def __init__(self, operations, apply_fn, cache=True, max_entries=200000):
        self.apply_fn = apply_fn
        self.operation_ids = {id(op): i for i, op in enumerate(operations)}
        self.enabled = cache
        self.max_entries = max_entries
        self.cache = {}
        self.counts = Counter()

    def __call__(self, op, values):
        output = []
        index = self.operation_ids[id(op)]
        for packed in values:
            self.counts['logical_grid_requests'] += 1
            key = (index, packed)
            if self.enabled and key in self.cache:
                self.counts['cache_hits'] += 1
                value = self.cache[key]
            else:
                self.counts['primitive_grid_evaluations'] += 1
                self.counts['primitive_input_cells'] += packed[0] * packed[1]
                transformed = self.apply_fn(op, (packed,))
                value = None if transformed is None else transformed[0]
                if self.enabled and len(self.cache) < self.max_entries:
                    self.cache[key] = value
                elif self.enabled:
                    self.counts['uncached_at_capacity'] += 1
            if value is None:
                return None
            output.append(value)
        return tuple(output)

    def metrics(self):
        return dict(self.counts, cache_entries=len(self.cache),
                    cache_payload_bytes=sum(len(k[1]) + (0 if v is None else len(v))
                                            for k, v in self.cache.items()))
