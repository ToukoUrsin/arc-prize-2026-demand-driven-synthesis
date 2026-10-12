"""Exact packed-byte evaluator for the frozen solver.operations factory. MIT-0.

Contract: packed immutable grids, dimensions 1..30, palette 0..9. Dispatch is
bound to the identities created by this grammar, not caller-supplied names.
"""
from functools import lru_cache
from solver import operations


def _grid(h, w, data):
    if not (1 <= h <= 30 and 1 <= w <= 30):
        return None
    return bytes((h, w)) + data


def _transpose(s):
    h, w, b = s[0], s[1], s[2:]
    return bytes((w, h)) + b''.join(b[x::w] for x in range(w))


def _crop(s, positions):
    if not positions:
        return None
    h, w, b = s[0], s[1], s[2:]
    y0, y1 = positions[0] // w, positions[-1] // w
    x0 = min(p % w for p in positions)
    x1 = max(p % w for p in positions)
    return _grid(y1-y0+1, x1-x0+1,
                 b''.join(b[y*w+x0:y*w+x1+1] for y in range(y0, y1+1)))


def _background(b):
    return max(range(10), key=b.count)  # First maximum: lowest color wins ties.


@lru_cache(maxsize=128)
def _components(s, bg, same):
    h, w, b = s[0], s[1], s[2:]
    remaining = {i for i, c in enumerate(b) if c != bg}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            p = stack.pop()
            y, x = divmod(p, w)
            neighbors = []
            if y: neighbors.append(p-w)
            if y+1 < h: neighbors.append(p+w)
            if x: neighbors.append(p-1)
            if x+1 < w: neighbors.append(p+1)
            for q in neighbors:
                if q in remaining and (not same or b[q] == b[start]):
                    remaining.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return tuple(sorted(result, key=lambda c: (-len(c), c)))


def _object(s, bgmode, same, small, crop):
    h, w, b = s[0], s[1], s[2:]
    bg = 0 if bgmode == 'zero' else _background(b)
    cs = _components(s, bg, same)
    if not cs:
        return None
    cells = min(cs, key=lambda c: (len(c), c)) if small else cs[0]
    data = bytearray([bg]) * (h*w)
    for p in cells:
        data[p] = b[p]
    value = bytes((h, w)) + bytes(data)
    return _crop(value, cells) if crop else value


def _transform(name, s):
    h, w, b = s[0], s[1], s[2:]
    if name == 'rotate90':
        return _grid(w, h, b''.join(b[x::w] for x in range(w-1, -1, -1)))
    if name == 'rotate180':
        return s[:2] + b[::-1]
    if name == 'rotate270':
        return _grid(w, h, b''.join(b[x::w][::-1] for x in range(w)))
    if name == 'flip_lr':
        return s[:2] + b''.join(b[y*w:(y+1)*w][::-1] for y in range(h))
    if name == 'flip_ud':
        return s[:2] + b''.join(b[y*w:(y+1)*w] for y in range(h-1, -1, -1))
    if name == 'transpose':
        return _transpose(s)
    if name == 'anti_transpose':
        return _grid(w, h, b''.join(b[x::w][::-1] for x in range(w-1, -1, -1)))
    if name in ('crop_nonzero', 'crop_nonbackground'):
        bg = 0 if name == 'crop_nonzero' else _background(b)
        return _crop(s, [i for i, c in enumerate(b) if c != bg])
    if name == 'compress_rows':
        rows = [b[y*w:(y+1)*w] for y in range(h)]
        kept = [row for i, row in enumerate(rows) if not i or row != rows[i-1]]
        return _grid(len(kept), w, b''.join(kept))
    if name == 'compress_columns':
        return _transpose(_transform('compress_rows', _transpose(s)))
    if name == 'remove_uniform_lines':
        rows = [y for y in range(h) if b[y*w:(y+1)*w] != bytes((b[y*w],))*w]
        cols = [x for x in range(w) if b[x::w] != bytes((b[x],))*h]
        return _grid(len(rows), len(cols), bytes(b[y*w+x] for y in rows for x in cols))
    if name.startswith('upscale'):
        k = int(name[-1])
        if h*k > 30 or w*k > 30:
            return None
        return _grid(h*k, w*k,
                     b''.join(b''.join(bytes((c,))*k for c in b[y*w:(y+1)*w])*k for y in range(h)))
    if name.startswith('downscale'):
        k = int(name[-1])
        if h % k or w % k:
            return None
        rows = []
        for y in range(0, h, k):
            row = b[y*w:(y+1)*w:k]
            expanded = b''.join(bytes((c,))*k for c in row)
            if any(b[(y+j)*w:(y+j+1)*w] != expanded for j in range(k)):
                return None
            rows.append(row)
        return _grid(h//k, w//k, b''.join(rows))
    if name.startswith('tile'):
        k = int(name[4])
        if h*k > 30 or w*k > 30:
            return None
        return _grid(h*k, w*k, b''.join(b[y*w:(y+1)*w]*k for y in range(h))*k)
    if name.startswith(('repeat_axis', 'mirror_axis')):
        axis, mirror = int(name[-1]), name.startswith('mirror')
        if (h if axis == 0 else w)*2 > 30:
            return None
        rows = [b[y*w:(y+1)*w] for y in range(h)]
        if axis == 0:
            return _grid(h*2, w, b + b''.join(reversed(rows) if mirror else rows))
        return _grid(h, w*2, b''.join(row+(row[::-1] if mirror else row) for row in rows))
    if name.startswith('half'):
        axis, side, gap = int(name[4]), int(name[6]), int(name[-1])
        n = h if axis == 0 else w
        if (n-gap) % 2:
            return None
        k = (n-gap)//2
        if k < 1:
            return None
        start = k+gap if side else 0
        if axis == 0:
            return _grid(k, w, b[start*w:(start+k)*w])
        return _grid(h, k, b''.join(b[y*w+start:y*w+start+k] for y in range(h)))
    if name.startswith('crop_color'):
        color = int(name[10:])
        return _crop(s, [i for i, c in enumerate(b) if c == color])
    raise ValueError('Unsupported frozen operation: ' + name)


class PackedGrammar:
    def __init__(self, train):
        self.operations = operations(train)
        self._functions = {}
        for op in self.operations:
            name = op.name
            if name.startswith('object_'):
                _, bg, same, small, crop = name.split('_')
                params = (bg, bool(int(same[-1])), bool(int(small[-1])), bool(int(crop[-1])))
                fn = lambda s, p=params: _object(s, *p)
            elif name.startswith(('keep_color', 'erase_color')):
                keep = name.startswith('keep')
                color = int(name[10:] if keep else name[11:])
                table = bytes((c if c == color else 0) if keep else (0 if c == color else c)
                              for c in range(256))
                fn = lambda s, t=table: s[:2] + s[2:].translate(t)
            elif name == 'demonstration_color_map':
                # The frozen factory stores the learned ten-entry array here.
                table = op.fn.__defaults__[0].tobytes() + bytes(range(10, 256))
                fn = lambda s, t=table: s[:2] + s[2:].translate(t)
            else:
                fn = lambda s, n=name: _transform(n, s)
            self._functions[id(op)] = fn

    def apply(self, op, values):
        fn = self._functions[id(op)]
        out = []
        for value in values:
            result = fn(value)
            if result is None:
                return None
            out.append(result)
        return tuple(out)

    @staticmethod
    def clear_cache():
        _components.cache_clear()
