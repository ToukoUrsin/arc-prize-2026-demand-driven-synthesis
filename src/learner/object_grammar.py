"""Finite object/relational grammar for exact staged search. MIT-0.

Only demonstration pairs construct the grammar. Runtime operations are pure
functions of one packed grid. No identifiers, query labels, stored answers,
external files, libraries beyond NumPy, or nondeterministic ties.
"""
from collections import Counter
from functools import lru_cache
from solver import Operation
from packed_2d import PackedGrammar as BaseGrammar


def background(s, role):
    return 0 if role == 'zero' else max(range(10), key=s[2:].count)


def neighbors(p, h, w, diagonal=False):
    y, x = divmod(p, w)
    for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w:
            yield yy*w + xx
    if diagonal:
        for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                yield yy*w + xx


def normalized_shape(points, w):
    coords = [divmod(p, w) for p in points]
    # D4-invariant binary object shape; color and location do not enter.
    variants = []
    for swap in (False, True):
        for sy, sx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = [(sy*(x if swap else y), sx*(y if swap else x)) for y, x in coords]
            y0, x0 = min(y for y, x in q), min(x for y, x in q)
            variants.append(tuple(sorted((y-y0, x-x0) for y, x in q)))
    return min(variants)


@lru_cache(maxsize=4096)
def objects(s, role, kind):
    h, w, b = s[0], s[1], s[2:]
    bg = background(s, role)
    remaining = {p for p, c in enumerate(b) if c != bg}
    groups = []
    if kind == 'colors':
        groups = [tuple(p for p in sorted(remaining) if b[p] == c)
                  for c in sorted(set(b)-{bg})]
    else:
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            queue, group = [start], [start]
            while queue:
                for p in neighbors(queue.pop(), h, w, kind == 'mono8'):
                    if p in remaining and (kind == 'fg4' or b[p] == b[start]):
                        remaining.remove(p)
                        queue.append(p)
                        group.append(p)
            groups.append(tuple(sorted(group)))
    result = []
    for cells in groups:
        ys, xs = zip(*(divmod(p, w) for p in cells))
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        box = (y1-y0+1)*(x1-x0+1)
        # Internal empty cells not 4-connected to the bounding-box exterior.
        empty = {y*w+x for y in range(y0, y1+1) for x in range(x0, x1+1)} - set(cells)
        border = {p for p in empty if p//w in (y0, y1) or p%w in (x0, x1)}
        queue = list(border)
        while queue:
            for p in neighbors(queue.pop(), h, w):
                if p in empty and p not in border:
                    border.add(p)
                    queue.append(p)
        shape = normalized_shape(cells, w)
        result.append((cells, y0, y1, x0, x1, box, len(empty-border),
                       len({b[p] for p in cells}), shape))
    return tuple(result)


@lru_cache(maxsize=16384)
def select(s, role, kind, criterion):
    objs = objects(s, role, kind)
    if not objs:
        return ()
    frequencies = Counter(o[8] for o in objs)
    if criterion == 'all':
        return objs
    if criterion == 'rare_shape':
        values = [frequencies[o[8]] for o in objs]
        target = min(values)
    elif criterion == 'common_shape':
        values = [frequencies[o[8]] for o in objs]
        target = max(values)
    else:
        key, direction = criterion.rsplit('_', 1)
        f = {'area': lambda o: len(o[0]), 'box': lambda o: o[5],
             'holes': lambda o: o[6], 'colors': lambda o: o[7],
             'row': lambda o: o[1], 'col': lambda o: o[3],
             'density': lambda o: len(o[0])/o[5]}[key]
        values = [f(o) for o in objs]
        target = (min if direction == 'min' else max)(values)
    return tuple(o for o, value in zip(objs, values) if value == target)


def object_transform(s, role, kind, criterion, action, color=None):
    chosen = select(s, role, kind, criterion)
    if not chosen:
        return None
    h, w, b = s[0], s[1], s[2:]
    bg = background(s, role)
    out = bytearray([bg])*(h*w) if action in ('crop', 'keep') else bytearray(b)
    cells = sorted({p for o in chosen for p in o[0]})
    for p in cells:
        out[p] = bg if action == 'erase' else (color if action == 'paint' else b[p])
    if action == 'crop':
        y0, y1 = cells[0]//w, cells[-1]//w
        x0, x1 = min(p%w for p in cells), max(p%w for p in cells)
        return bytes((y1-y0+1, x1-x0+1)) + b''.join(
            out[y*w+x0:y*w+x1+1] for y in range(y0, y1+1))
    if action in ('boxes', 'outlines'):
        for o in chosen:
            own = b[o[0][0]] if color is None else color
            for y in range(o[1], o[2]+1):
                for x in range(o[3], o[4]+1):
                    if action == 'boxes' or y in (o[1], o[2]) or x in (o[3], o[4]):
                        if out[y*w+x] == bg:
                            out[y*w+x] = own
    return s[:2] + bytes(out)


def connect(s, role, direction, mode):
    h, w, b = s[0], s[1], s[2:]
    bg = background(s, role)
    out, claims = bytearray(b), {}
    # Same-colored endpoints extend only through background. Conflicting colors
    # make an operation undefined, avoiding an arbitrary color-order overwrite.
    for color in sorted(set(b)-{bg}):
        points = [divmod(p, w) for p, c in enumerate(b) if c == color]
        groups = {}
        for y, x in points:
            key = y if direction == 'row' else x if direction == 'col' else y-x if direction == 'diag' else y+x
            groups.setdefault(key, []).append((y, x))
        for group in groups.values():
            if len(group) < 2:
                continue
            group.sort()
            if direction == 'row': group.sort(key=lambda p: p[1])
            y0, x0 = group[0]; y1, x1 = group[-1]
            dy, dx = (y1 > y0)-(y1 < y0), (x1 > x0)-(x1 < x0)
            n = max(abs(y1-y0), abs(x1-x0))
            for k in range(n+1):
                p = (y0+k*dy)*w+x0+k*dx
                if b[p] != bg and b[p] != color:
                    if mode == 'clear': return None
                    continue
                if p in claims and claims[p] != color: return None
                claims[p] = color
    for p, c in claims.items(): out[p] = c
    return s[:2]+bytes(out)


def holes(s, role, color):
    h, w, b = s[0], s[1], s[2:]
    bg = background(s, role)
    empty = {p for p, c in enumerate(b) if c == bg}
    reached = {p for p in empty if p//w in (0, h-1) or p%w in (0, w-1)}
    queue = list(reached)
    while queue:
        for p in neighbors(queue.pop(), h, w):
            if p in empty and p not in reached:
                reached.add(p); queue.append(p)
    out = bytearray(b)
    for p in empty-reached: out[p] = color
    return s[:2]+bytes(out)


def symmetry(s, role, axis):
    h, w, b = s[0], s[1], s[2:]
    if axis in ('transpose', 'anti') and h != w: return None
    bg, out = background(s, role), bytearray(b)
    for p, c in enumerate(b):
        y, x = divmod(p, w)
        yy, xx = {'lr': (y, w-1-x), 'ud': (h-1-y, x),
                  'rot': (h-1-y, w-1-x), 'transpose': (x, y),
                  'anti': (w-1-x, h-1-y)}[axis]
        q = yy*w+xx
        if c != bg:
            if b[q] != bg and b[q] != c: return None
            out[q] = c
    return s[:2]+bytes(out)


class ObjectGrammar:
    def __init__(self, train):
        self.base = BaseGrammar(train)
        self.operations = list(self.base.operations)
        self.functions = {}
        def add(name, fn):
            op = Operation('rel_'+name, None)
            self.operations.append(op); self.functions[id(op)] = fn
        # Color literals are derived only from demonstration output palettes.
        colors = sorted({c for pair in train for row in pair['output'] for c in row})
        selectors = ('area_min', 'area_max', 'box_min', 'box_max', 'holes_max',
                     'colors_max', 'rare_shape', 'common_shape', 'density_min',
                     'row_min', 'row_max', 'col_min', 'col_max')
        paint_selectors = ('area_min', 'area_max', 'holes_max', 'rare_shape', 'common_shape')
        for role in ('zero', 'mode'):
            for kind in ('fg4', 'mono4', 'mono8'):
                for criterion in selectors:
                    for action in ('crop', 'keep', 'erase'):
                        add(f'{role}_{kind}_{criterion}_{action}',
                            lambda s, r=role, k=kind, c=criterion, a=action:
                            object_transform(s, r, k, c, a))
                for criterion in paint_selectors:
                    for color in colors:
                        add(f'{role}_{kind}_{criterion}_paint{color}',
                            lambda s, r=role, k=kind, c=criterion, v=color:
                            object_transform(s, r, k, c, 'paint', v))
            for kind in ('mono4', 'colors'):
                for action in ('boxes', 'outlines'):
                    add(f'{role}_{kind}_{action}', lambda s, r=role, k=kind, a=action:
                        object_transform(s, r, k, 'all', a))
            for direction in ('row', 'col', 'diag', 'anti'):
                for mode in ('clear', 'skip'):
                    add(f'{role}_connect_{direction}_{mode}',
                        lambda s, r=role, d=direction, m=mode: connect(s, r, d, m))
            for color in colors:
                add(f'{role}_holes{color}', lambda s, r=role, c=color: holes(s, r, c))
            for axis in ('lr', 'ud', 'rot', 'transpose', 'anti'):
                add(f'{role}_symmetry_{axis}', lambda s, r=role, a=axis: symmetry(s, r, a))
        # Imported after this module is initialized to share the pure object
        # representation without constructor state in primitive evaluation.
        from constructors import construct, apply_rule, name
        self.rules = construct(train)
        for spec in self.rules:
            add(name(spec), lambda s, spec=spec: apply_rule(s, spec))

    def clear_cache(self):
        self.base.clear_cache(); objects.cache_clear(); select.cache_clear()

    def apply(self, op, values):
        fn = self.functions.get(id(op))
        if fn is None: return self.base.apply(op, values)
        out = []
        for s in values:
            value = fn(s)
            if value is None: return None
            out.append(value)
        return tuple(out)
