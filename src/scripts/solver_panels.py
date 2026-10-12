"""Separate general 2D grammar extension: Boolean composition of equal panels.
SPDX-License-Identifier: MIT-0
All parameters are fixed; demonstration-fitted terminal color maps are separate.
No task IDs, category or query labels enter operations.
"""
import numpy as np
from solver import Operation, operations as base_operations, background, half


def combine(a, axis, gap, bgmode, kind):
    left = half(a, axis, 0, gap)
    right = half(a, axis, 1, gap)
    if left is None or right is None or left.shape != right.shape or min(left.shape) < 1:
        return None
    bg = 0 if bgmode == 'zero' else background(a)
    x, y = left != bg, right != bg
    if kind == 'union':
        mask = x | y
    elif kind == 'intersection':
        mask = x & y
    elif kind == 'xor':
        mask = x ^ y
    elif kind == 'left_only':
        mask = x & ~y
    elif kind == 'right_only':
        mask = y & ~x
    else:
        raise ValueError(kind)
    # For overlap the first panel's foreground wins. A terminal color map can
    # consistently recolor the structural result using all demonstrations.
    colors = np.where(x, left, right)
    return np.where(mask, colors, bg).astype(np.uint8)


def operations(train):
    ops = base_operations(train)
    for axis in (0, 1):
        for gap in (0, 1):
            for bgmode in ('zero', 'mode'):
                for kind in ('union', 'intersection', 'xor', 'left_only', 'right_only'):
                    ops.append(Operation(f'panel_axis{axis}_gap{gap}_{bgmode}_{kind}',
                        lambda a, axis=axis, gap=gap, bgmode=bgmode, kind=kind:
                        combine(a, axis, gap, bgmode, kind)))
    return ops
