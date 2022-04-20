# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np


def adaptive(f, a, b, initial_num=10, max_num=2000, tol=2, min_rel_dist=1e-3,
             aspect_ratio=4/3):
    """Adaptive sampling of a |NumPy array|-valued function.

    Samples the function such that the piecewise linear interpolation looks
    "smooth".

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
    tol
        Tolerance for the maximum pointwise angle in degrees away from 180°.
    min_rel_dist
        Minimum distance between two neighboring points relative to the width
        of the plot.
    aspect_ratio
        Ratio between width and height of the plot, used in calculating
        angles and distances.

    Returns
    -------
    points
        A 1D |NumPy array| of sampled points.
    fvals
        A |NumPy array| of function values.
    """
    tol *= np.pi / 180
    points = list(np.linspace(a, b, initial_num))
    fvals = [f(p) for p in points]
    while len(points) < max_num:
        angles, dists = _angles_and_dists(points, fvals, aspect_ratio)
        dists_pair_max = np.max(np.stack((dists[:-1], dists[1:])), axis=0)
        angles[dists_pair_max <= min_rel_dist] = np.pi
        idx = np.unravel_index(np.argmin(angles), angles.shape)
        if np.pi - angles[idx] <= tol:
            break
        idx_1 = (idx[0] + 1,) + idx[1:]
        if dists[idx_1] > min_rel_dist:
            p2 = (points[idx[0] + 1] + points[idx[0] + 2]) / 2
            points.insert(idx[0] + 2, p2)
            fvals.insert(idx[0] + 2, f(p2))
        if dists[idx] > min_rel_dist and len(points) < max_num:
            p1 = (points[idx[0]] + points[idx[0] + 1]) / 2
            points.insert(idx[0] + 1, p1)
            fvals.insert(idx[0] + 1, f(p1))
    return np.array(points), np.array(fvals)


def _angles_and_dists(x, y, aspect_ratio):
    x_range = x[-1] - x[0]
    y_range = np.max(y, axis=0, keepdims=True) - np.min(y, axis=0, keepdims=True)
    y_range[y_range == 0] = 1
    x = np.array(x) / x_range * aspect_ratio
    y = np.array(y) / y_range
    dx = x[:-1] - x[1:]
    dy = y[:-1] - y[1:]
    dx = dx.reshape(dx.shape + (dy.ndim - 1) * (1,))
    dists = np.sqrt(dx**2 + dy**2)
    inner_products = -(dx[:-1] * dx[1:] + dy[:-1] * dy[1:])
    ip_norm = inner_products / (dists[:-1] * dists[1:])
    angles = np.arccos(np.clip(ip_norm, -1, 1))
    return angles, dists
