# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import math as m
from hypothesis import strategies as hyst
import numpy as np
import pytest

from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.builtin.grids.subgrid import SubGrid
from pymor.discretizers.builtin.grids.tria import TriaGrid
from pymor.discretizers.builtin.grids.unstructured import UnstructuredTriangleGrid


# TODO let domain be drawn too
def hy_rect_tria_kwargs(draw):
    identify_left_right = draw(hyst.booleans())
    identify_bottom_top = draw(hyst.booleans())
    interval_i = hyst.integers(min_value=1, max_value=42)
    num_intervals = draw(hyst.tuples(interval_i.filter(lambda x: (not identify_left_right) or x > 1),
                                     interval_i.filter(lambda y: (not identify_bottom_top) or y > 1)))
    # domain_value = hyst.floats(allow_infinity=False, allow_nan=False)
    # domain_point = hyst.tuples(domain_value, domain_value)
    # domain = draw(hyst.tuples(domain_point, domain_point).filter(lambda d: d[0][0] < d[1][0] and d[0][1] < d[1][1]))
    domain = ((0,0), (1,1))
    return {"num_intervals": num_intervals, "domain": domain, "identify_left_right": identify_left_right,
            "identify_bottom_top": identify_bottom_top}


@hyst.composite
def hy_rect_grid(draw):
    return RectGrid(**hy_rect_tria_kwargs(draw))


@hyst.composite
def hy_tria_grid(draw):
    return RectGrid(**hy_rect_tria_kwargs(draw))


# TODO negative Domain values produce centers outside bounding box
@hyst.composite
def hy_oned_grid(draw):
    identify_left_right = draw(hyst.booleans())
    interval_i = hyst.integers(min_value=1, max_value=10000)
    num_intervals = draw(interval_i.filter(lambda x: (not identify_left_right) or x > 1))
    # domain points are limited to allow their norm2 computations
    domain_point = hyst.floats(allow_infinity=False, allow_nan=False, min_value=0,
                               max_value=np.math.sqrt(np.finfo(float).max))
    domain = draw(hyst.tuples(domain_point, domain_point).filter(lambda d: d[0] < d[1]))
    return OnedGrid(num_intervals=num_intervals, domain=domain, identify_left_right=identify_left_right)


# TODO re-use other grid strategies
@hyst.composite
def hy_subgrid(draw):
    grid = draw(hyst.sampled_from([RectGrid((1, 1)), TriaGrid((1, 1)), RectGrid((8, 8)), TriaGrid((24, 24))]))
    neq = draw(hyst.sampled_from([0, 2, 4]))
    if neq == 0:
        return SubGrid(grid, np.arange(grid.size(0), dtype=np.int32))
    else:
        random = draw(hyst.randoms())
        sample = random.sample(range(grid.size(0)), max(int(m.floor(grid.size(0) / neq)), 1))
        return SubGrid(grid, np.array(sample))


hy_grid = hy_rect_grid() | hy_tria_grid() | hy_oned_grid() | hy_subgrid()
hy_rect_or_tria_grid = hy_rect_grid() | hy_tria_grid()
hy_unstructured_grid = hyst.just(UnstructuredTriangleGrid.from_vertices(
    np.array([[0, 0], [-1, -1], [1, -1], [1, 1], [-1, 1]]), np.array([[0, 1, 2], [0, 3, 4], [0, 4, 1]])))
hy_grid_with_orthogonal_centers = hy_rect_grid() | hy_oned_grid()
hy_grids_with_visualize = hy_rect_grid() | hy_tria_grid() | hy_oned_grid() | hy_unstructured_grid
