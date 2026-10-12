"""Input-only color equality/role features for finite learned copy proposals.
SPDX-License-Identifier: MIT-0. Constructor and fitter: adjacent frozen v4.
"""
from collections import Counter
from itertools import combinations
import numpy as np
from scipy import ndimage as ndi
import solver_color_base as base

VERSION = 'color-relations-v1'
ORIGINAL_FEATURES = base.features
ORIGINAL_SOURCES = base.output_sources


def features(a):
    a = np.asarray(a, dtype=np.int16)
    X, names, tags = ORIGINAL_FEATURES(a)
    values = [X]
    names, tags = list(names), list(tags)
    byname = {n: X[:, i] for i, n in enumerate(names)}
    h, w = a.shape
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    present = list(map(int, np.flatnonzero(counts)))
    appearance = list(dict.fromkeys(map(int, a.ravel())))
    orderings = {
        'frequency': sorted(present, key=lambda c: (-counts[c], appearance.index(c))),
        'rare': sorted(present, key=lambda c: (counts[c], appearance.index(c))),
        'appearance': appearance,
        'foreground': sorted([c for c in present if c != bg],
                             key=lambda c: (-counts[c], appearance.index(c))),
    }
    def add(name, v, tag):
        if np.isscalar(v): v = np.full(a.size, v)
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        assert len(v) == a.size
        values.append(v[:, None]); names.append(name); tags.append(tag)
        byname[name] = v

    for role, colors in orderings.items():
        rank = np.full(10, 10)
        for i, c in enumerate(colors): rank[c] = i
        add('role_rank_' + role, rank[a], 'color')
        for i in range(3):
            c = colors[i] if len(colors) > i else bg
            add(f'role_color_{role}_{i}', c, 'base')
            add(f'role_is_{role}_{i}', a == c, 'color')
    add('own_color_count', counts[a], 'color')
    add('own_color_unique', counts[a] == 1, 'color')
    add('palette_size', len(present), 'color')

    # Equality edges form short color-independent conditions instead of numeric
    # color-ID thresholds. The finite fixed list is independent of task/labels.
    local = ['color'] + [f'neighbor_{dy}_{dx}' for dy, dx in
        [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]]
    mirrors = ['color', 'mirror_ud', 'mirror_lr', 'mirror_both',
               'transpose', 'antitranspose']
    rays = [f'ray{dy}_{dx}_color' for dy, dx in
        [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]]
    differing = [r[:-5] + 'diffcolor' for r in rays]
    for group, sources, tag in [('local', local, 'local'),
                                ('mirrors', mirrors, 'local'),
                                ('rays', ['color'] + rays + differing, 'rays')]:
        for left, right in combinations(sources, 2):
            x, y = byname[left], byname[right]
            add('equal_' + group + '_' + left + '_' + right,
                (x == y) & (x < 10) & (y < 10), tag)
        for name in sources:
            color = byname[name].astype(int)
            valid = color < 10
            safe = np.clip(color, 0, 9)
            add('source_is_bg_' + group + '_' + name,
                valid & (color == bg), tag)
            add('source_count_' + group + '_' + name,
                np.where(valid, counts[safe], -1), tag)

    # Candidate decoders can select the nearest nonbackground/different color
    # without learning its direction or literal color identifier.
    for kind, sources in [('ray', rays), ('different', differing)]:
        distances = np.stack([byname[s[:-5] + 'distance'] for s in sources], axis=1)
        colors = np.stack([byname[s] for s in sources], axis=1).astype(int)
        minimum = distances.min(axis=1)
        chosen = distances.argmin(axis=1)
        copied = colors[np.arange(a.size), chosen]
        add('closest_color_' + kind, np.where(copied < 10, copied, a.ravel()), 'rays')
        add('closest_distance_' + kind, minimum, 'rays')
        add('closest_unique_' + kind, (distances == minimum[:, None]).sum(axis=1) == 1, 'rays')

    # Global nearest foreground color also works away from the eight rays.
    for bgval, label in [(0, 'zero'), (bg, 'mode')]:
        mask = a != bgval
        if mask.any():
            idx = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            copied = a[tuple(idx)]
        else: copied = a
        add('closest_color_euclidean_' + label, copied, 'color')

    return np.concatenate(values, axis=1), names, tags


def output_sources(X, names):
    S, sources = ORIGINAL_SOURCES(X, names)
    ix = [i for i, name in enumerate(names)
          if name.startswith(('role_color_', 'closest_color_'))]
    if ix:
        S = np.concatenate((S, X[:, ix].astype(np.int16)), axis=1)
        sources += [names[i] for i in ix]
    return S, sources


base.features = features
base.output_sources = output_sources
clean_task = base.clean_task
solve = base.solve
replay = base.replay
transform = base.transform
