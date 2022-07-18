# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import math as m

from hypothesis import strategies as hyst
import numpy as np

from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.builtin.grids.subgrid import SubGrid
from pymor.discretizers.builtin.grids.tria import TriaGrid
from pymor.discretizers.builtin.grids.unstructured import UnstructuredTriangleGrid


def _hy_domain_bounds(draw, grid_type):
    # domain points are limited to allow their norm2 computations
    max_val = grid_type.MAX_DOMAIN_WIDTH / 2
    min_val = -grid_type.MAX_DOMAIN_WIDTH / 2
    domain_point = hyst.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False,
                               min_value=min_val, max_value=max_val)

    ll = draw(hyst.tuples(*[domain_point] * grid_type.dim))

    def _filter(d):
        return (all(l < r and abs(r - l) > grid_type.MIN_DOMAIN_WIDTH for l, r in zip(ll, d))
                and grid_type._check_domain((ll, d)))

    rr = draw(hyst.tuples(*[domain_point] * grid_type.dim).filter(_filter))
    return ll, rr


def _hy_rect_tria_kwargs(draw, grid_type):
    identify_left_right = draw(hyst.booleans())
    identify_bottom_top = draw(hyst.booleans())
    interval_i = hyst.integers(min_value=1, max_value=42)

    lambda x: (not identify_left_right) or x > 1
    num_intervals = draw(hyst.tuples(interval_i.map(lambda x: x if not identify_left_right else max(2, x)),
                                     interval_i.map(lambda y: y if not identify_bottom_top else max(2, y))))

    domain = _hy_domain_bounds(draw, grid_type=grid_type)
    return {"num_intervals": num_intervals, "domain": domain, "identify_left_right": identify_left_right,
            "identify_bottom_top": identify_bottom_top}


@hyst.composite
def hy_rect_grid(draw):
    return RectGrid(**_hy_rect_tria_kwargs(draw, RectGrid))


@hyst.composite
def hy_tria_grid(draw):
    return TriaGrid(**_hy_rect_tria_kwargs(draw, TriaGrid))


@hyst.composite
def hy_oned_grid(draw):
    identify_left_right = draw(hyst.booleans())
    interval_i = hyst.integers(min_value=1, max_value=10000)
    num_intervals = draw(interval_i.filter(lambda x: (not identify_left_right) or x > 1))
    domain = _hy_domain_bounds(draw, grid_type=OnedGrid)
    return OnedGrid(num_intervals=num_intervals, domain=[domain[0][0], domain[1][0]],
                    identify_left_right=identify_left_right)


@hyst.composite
def hy_subgrid(draw):
    grid = draw(hy_rect_grid() | hy_tria_grid())
    neq = draw(hyst.sampled_from([0, 2, 4]))
    if neq == 0:
        return SubGrid(grid, np.arange(grid.size(0), dtype=np.int32))
    else:
        random = draw(hyst.randoms())
        sample = random.sample(range(grid.size(0)), max(int(m.floor(grid.size(0) / neq)), 1))
        return SubGrid(grid, np.array(sample))


hy_unstructured_grid = hyst.just(UnstructuredTriangleGrid.from_vertices(
    np.array([[0, 0], [-1, -1], [1, -1], [1, 1], [-1, 1]]), np.array([[0, 1, 2], [0, 3, 4], [0, 4, 1]])))


hy_grid = hy_rect_grid() | hy_tria_grid() | hy_oned_grid() | hy_subgrid() | hy_unstructured_grid
hy_rect_or_tria_grid = hy_rect_grid() | hy_tria_grid()
hy_grid_with_orthogonal_centers = hy_rect_grid() | hy_oned_grid()
hy_grids_with_visualize = hy_rect_grid() | hy_tria_grid() | hy_oned_grid() | hy_unstructured_grid


@hyst.composite
def hy_grid_and_dim_range_product(draw):
    grid = draw(hy_grid)
    dim = hyst.integers(min_value=0, max_value=grid.dim)
    return grid, draw(dim), draw(dim)


@hyst.composite
def hy_grid_and_dim_range_product_and_s_max_en(draw):
    g, e, n = draw(hy_grid_and_dim_range_product())
    s = hyst.integers(min_value=max(e, n), max_value=g.dim)
    return g, e, n, draw(s)


@hyst.composite
def hy_grid_and_dim_range_product_and_s(draw):
    g, e, n = draw(hy_grid_and_dim_range_product())
    s = hyst.integers(min_value=e, max_value=g.dim)
    return g, e, n, draw(s)


@hyst.composite
def hy_grid_and_dim_range_product_and_s_to_e(draw):
    g, e, n = draw(hy_grid_and_dim_range_product())
    s = hyst.integers(min_value=0, max_value=max(e-1, 0))
    return g, e, n, draw(s)


@hyst.composite
def hy_grid_and_codim_product_and_entity_index(draw):
    grid = draw(hy_grid)
    codim = draw(hyst.integers(min_value=0, max_value=grid.dim-1))
    index = hyst.integers(min_value=0, max_value=grid.size(codim)-1)
    return grid, codim, draw(index)
