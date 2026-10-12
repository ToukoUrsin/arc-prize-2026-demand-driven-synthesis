"""Exact component feature lookup with per-property demand. MIT-0.

Component labels are grid-local. Tables include a zero-valued background row;
no label numbering is exposed to the grammar. Rectangular component shortcuts
are identities for the reference's fill, erosion and symmetry definitions.
"""
import numpy as np
from scipy import ndimage as ndi
from fast_features import FeatureGrid as BaseFeatureGrid, NAMES, TAGS


class ComponentTable:
    def __init__(self, a, conn, bg):
        self.a = a
        self.h, self.w = a.shape
        labels = np.zeros(a.shape, dtype=np.int32)
        n = 0
        structure = ndi.generate_binary_structure(2, conn)
        for color in (range(10) if bg is None else (None,)):
            part, count = ndi.label(a == color if bg is None else a != bg, structure)
            selected = part != 0
            labels[selected] = part[selected] + n
            n += count
        self.labels, self.n = labels, n
        flat = labels.ravel()
        self.area = np.bincount(flat, minlength=n + 1)
        y, x = np.indices(a.shape)
        self.y, self.x = y, x
        self.y0, self.x0 = np.full(n+1, self.h), np.full(n+1, self.w)
        self.y1, self.x1 = np.zeros(n+1, dtype=int), np.zeros(n+1, dtype=int)
        np.minimum.at(self.y0, flat, y.ravel())
        np.minimum.at(self.x0, flat, x.ravel())
        np.maximum.at(self.y1, flat, y.ravel())
        np.maximum.at(self.x1, flat, x.ravel())
        self.height, self.width = self.y1-self.y0+1, self.x1-self.x0+1
        self.full = self.area == self.height*self.width
        self.cache = {}

    def __getitem__(self, name):
        if name not in self.cache:
            values = self.property(name)
            # The reference initializes all background feature cells to zero.
            if np.ndim(values) == 2:
                result = np.where(self.labels, values, 0)
            else:
                values = np.array(values, dtype=np.int16, copy=True)
                values[0] = 0
                result = values[self.labels]
            self.cache[name] = result
        return self.cache[name]

    def property(self, name):
        n = self.n
        if name == 'area': return self.area
        if name == 'height': return self.height
        if name == 'width': return self.width
        if name == 'top': return self.y0
        if name == 'left': return self.x0
        if name == 'bottom': return self.h-1-self.y1
        if name == 'right': return self.w-1-self.x1
        if name == 'row': return self.y-self.y0[self.labels]
        if name == 'col': return self.x-self.x0[self.labels]
        if name == 'edge': return (self.y0 == 0) | (self.x0 == 0) | (self.y1 == self.h-1) | (self.x1 == self.w-1)
        if name == 'fill': return self.full
        if name == 'rankarea': return np.searchsorted(np.sort(self.area[1:]), self.area)
        if name == 'nobjects': return np.full(n+1, n)
        if name == 'rankrow': return np.searchsorted(np.unique(self.y0[1:]), self.y0)
        if name == 'rankcol': return np.searchsorted(np.unique(self.x0[1:]), self.x0)
        if name in ('colors', 'color'):
            if 'color_counts' not in self.__dict__:
                joint = self.labels.ravel()*10 + self.a.ravel()
                self.color_counts = np.bincount(joint, minlength=(n+1)*10).reshape(n+1, 10)
            return (self.color_counts > 0).sum(1) if name == 'colors' else self.color_counts.argmax(1)
        values = np.zeros(n+1, dtype=np.int16)
        if name == 'perimeter':
            values[1:] = self.area[1:] - np.maximum(self.height[1:]-2, 0)*np.maximum(self.width[1:]-2, 0)
        elif name in ('sym_ud', 'sym_lr'):
            values[1:] = 1
        elif name != 'holes':
            raise KeyError(name)
        for label in np.flatnonzero(~self.full[1:])+1:
            sl = (slice(self.y0[label], self.y1[label]+1), slice(self.x0[label], self.x1[label]+1))
            mask = self.labels[sl] == label
            if name == 'holes': value = (ndi.binary_fill_holes(mask) & ~mask).sum()
            elif name == 'perimeter': value = (mask & ~ndi.binary_erosion(mask)).sum()
            elif name == 'sym_ud': value = np.array_equal(mask, mask[::-1])
            else: value = np.array_equal(mask, mask[:, ::-1])
            values[label] = value
        return values


class FeatureGrid(BaseFeatureGrid):
    def component_maps(self, conn, bg):
        return ComponentTable(self.a, conn, bg)


def features(a):
    grid = FeatureGrid(a)
    return np.stack([grid.column(name) for name in NAMES], axis=1).astype(np.float32), list(NAMES), list(TAGS)
