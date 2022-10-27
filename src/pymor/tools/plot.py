# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import math

import numpy as np

from pymor.core.base import BasicObject
from pymor.core.defaults import defaults


@defaults('initial_num', 'max_num', 'angle_tol', 'min_rel_dist', 'aspect_ratio', 'xscale', 'yscale')
def adaptive(f, a, b, initial_num=10, max_num=2000, angle_tol=2, min_rel_dist=1e-2,
             aspect_ratio=4/3, xscale='linear', yscale='linear'):
    """Adaptive sampling of a |NumPy array|-valued function.

    Samples the function such that the piecewise linear interpolation looks "smooth".

    If the function is complex-valued, it is assumed that the magnitude and phase should be plotted.

    Parameters
    ----------
    f
        The function to sample.
    a
        The left bound.
    b
        The right bound.
    initial_num
        Initial number of linearly spaced sampling points.
    max_num
        Maximum number of sampling points.
    angle_tol
        Tolerance for the maximum pointwise angle in degrees away from 180°.
    min_rel_dist
        Minimum distance between two neighboring points relative to the width of the plot.
    aspect_ratio
        Ratio between width and height of the plot, used in calculating angles and distances.
    xscale
        Whether to use linear (`'linear'`) or logarithmic (`'log'`) scale in the x-axis.
    yscale
        Whether to use linear (`'linear'`) or logarithmic (`'log'`) scale in the y-axis.
        If the function is complex-valued, yscale only refers to the magnitude
        (phase is always assumed to be in a linear scale).

    Returns
    -------
    points
        A 1D |NumPy array| of sampled points.
    fvals
        A |NumPy array| of function values.
    """
    plot = Adaptive(f, a, b, initial_num=initial_num, max_num=max_num,
                    angle_tol=angle_tol, min_rel_dist=min_rel_dist,
                    aspect_ratio=aspect_ratio, xscale=xscale, yscale=yscale)
    return plot.compute()


class Adaptive(BasicObject):

    def __init__(self, f, a, b, initial_num, max_num, angle_tol, min_rel_dist,
                 aspect_ratio, xscale, yscale):
        assert a < b
        assert initial_num >= 3
        assert max_num > initial_num
        assert 0 < angle_tol < 90
        assert 0 < min_rel_dist < 1
        assert aspect_ratio > 0
        assert xscale in ('linear', 'log')
        assert yscale in ('linear', 'log')

        angle_tol *= np.pi / 180
        self.__auto_init(locals())

        if xscale == 'linear':
            self.points = list(np.linspace(a, b, initial_num))
            self.x = self.points
            self.x_range = b - a
        else:
            points_array = np.geomspace(a, b, initial_num)
            self.points = list(points_array)
            self.x = list(np.log2(points_array))
            self.x_range = self.x[-1] - self.x[0]
        self.fvals = [f(p) for p in self.points]
        self.is_complex = any(np.iscomplexobj(fval) for fval in self.fvals)
        if not self.is_complex:
            if yscale == 'linear':
                self.y = self.fvals
            else:
                self.y = [np.log2(fval) for fval in self.fvals]
        else:
            if yscale == 'linear':
                self.y = list(np.stack(
                    (np.abs(self.fvals), np.unwrap(np.angle(self.fvals), axis=0)),
                    axis=-1,
                ))
            else:
                self.y = list(np.stack(
                    (np.log2(np.abs(self.fvals)), np.unwrap(np.angle(self.fvals), axis=0)),
                    axis=-1,
                ))
        self.y_min = np.min(self.y, axis=0, keepdims=True)
        self.y_max = np.max(self.y, axis=0, keepdims=True)

    def _p_mean(self, p1, p2):
        if self.xscale == 'linear':
            return (p1 + p2) / 2
        else:
            # geometric mean, but trying to avoid overflow or underflow
            return 2**((math.log2(p1) + math.log2(p2)) / 2)

    def _angles_and_dists(self):
        y_range = self.y_max - self.y_min
        y_range[y_range == 0] = 1
        x = np.array(self.x) / self.x_range
        y = np.array(self.y) / (y_range * self.aspect_ratio)
        dx = x[:-1] - x[1:]
        dy = y[:-1] - y[1:]
        dx = dx.reshape(dx.shape + (dy.ndim - 1) * (1,))
        dists = np.sqrt(dx**2 + dy**2)
        inner_products = -(dx[:-1] * dx[1:] + dy[:-1] * dy[1:])
        inner_products_normed = inner_products / (dists[:-1] * dists[1:])
        angles = np.arccos(np.clip(inner_products_normed, -1, 1))
        return angles, dists

    def _insert(self, index):
        p = self._p_mean(self.points[index - 1], self.points[index])
        self.points.insert(index, p)
        self.fvals.insert(index, self.f(p))
        if self.xscale == 'log':
            self.x.insert(index, math.log2(p))
        if not self.is_complex:
            if self.yscale == 'log':
                self.y.insert(index, np.log2(self.fvals[index]))
        else:
            if self.yscale == 'linear':
                self.y.insert(index, np.stack(
                    (np.abs(self.fvals[index]), np.angle(self.fvals[index])),
                    axis=-1,
                ))
            else:
                self.y.insert(index, np.stack(
                    (np.log2(np.abs(self.fvals[index])), np.angle(self.fvals[index])),
                    axis=-1,
                ))
            unwrapped_angle = np.unwrap([y[..., 1] for y in self.y], axis=0)
            for y, new_angle in zip(self.y, unwrapped_angle):
                y[..., 1] = new_angle
        self.y_min = np.min(self.y, axis=0, keepdims=True)
        self.y_max = np.max(self.y, axis=0, keepdims=True)

    def _loop(self):
        while len(self.points) < self.max_num:
            angles, dists = self._angles_and_dists()

            # ignore points where both incident segments are short
            dists_pair_max = np.max(np.stack((dists[:-1], dists[1:])), axis=0)
            angles[dists_pair_max <= self.min_rel_dist] = np.pi

            # find the point with the sharpest angle
            idx = np.unravel_index(np.argmin(angles), angles.shape)
            if np.pi - angles[idx] <= self.angle_tol:
                break

            # sample at the longer segment
            idx_1 = (idx[0] + 1,) + idx[1:]
            if dists[idx] < dists[idx_1]:
                self._insert(idx[0] + 2)
            else:
                self._insert(idx[0] + 1)

    def compute(self):
        self._loop()
        return np.array(self.points), np.array(self.fvals)
