# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import math as m
import random

import numpy as np
import pytest

from pymor.grids.oned import OnedGrid
from pymor.grids.rect import RectGrid
from pymor.grids.subgrid import SubGrid
from pymor.grids.tria import TriaGrid
from pymor.grids.unstructured import UnstructuredTriangleGrid


rect_grid_generators = [lambda arg=arg, kwargs=kwargs: RectGrid(arg, **kwargs) for arg, kwargs in
                        [((2, 4), {}),
                         ((1, 1), {}),
                         ((42, 42), {}),
                         ((2, 4), dict(identify_left_right=True)),
                         ((2, 4), dict(identify_bottom_top=True)),
                         ((2, 4), dict(identify_left_right=True, identify_bottom_top=True)),
                         ((2, 1), dict(identify_left_right=True)),
                         ((1, 2), dict(identify_bottom_top=True)),
                         ((2, 2), dict(identify_left_right=True, identify_bottom_top=True)),
                         ((42, 30), dict(identify_left_right=True)),
                         ((42, 30), dict(identify_bottom_top=True)),
                         ((42, 30), dict(identify_left_right=True, identify_bottom_top=True))]]


tria_grid_generators = [lambda arg=arg, kwargs=kwargs: TriaGrid(arg, **kwargs) for arg, kwargs in
                        [((2, 4), {}),
                         ((1, 1), {}),
                         ((42, 42), {}),
                         ((2, 4), dict(identify_left_right=True)),
                         ((2, 4), dict(identify_bottom_top=True)),
                         ((2, 4), dict(identify_left_right=True, identify_bottom_top=True)),
                         ((2, 1), dict(identify_left_right=True)),
                         ((1, 2), dict(identify_bottom_top=True)),
                         ((2, 2), dict(identify_left_right=True, identify_bottom_top=True)),
                         ((42, 30), dict(identify_left_right=True)),
                         ((42, 30), dict(identify_bottom_top=True)),
                         ((42, 30), dict(identify_left_right=True, identify_bottom_top=True))]]


oned_grid_generators = [lambda kwargs=kwargs: OnedGrid(**kwargs) for kwargs in
                        [dict(domain=np.array((-2, 2)), num_intervals=10),
                         dict(domain=np.array((-4, -2)), num_intervals=100),
                         dict(domain=np.array((-4, -2)), num_intervals=100, identify_left_right=True),
                         dict(domain=np.array((2, 3)), num_intervals=10),
                         dict(domain=np.array((2, 3)), num_intervals=10, identify_left_right=True),
                         dict(domain=np.array((1, 2)), num_intervals=10000)]]

unstructured_grid_generators = [lambda: UnstructuredTriangleGrid(np.array([[0, 0], [-1, -1], [1, -1], [1, 1], [-1, 1]]),
                                                                 np.array([[0, 1, 2], [0, 3, 4], [0, 4, 1]]))]


def subgrid_factory(grid_generator, neq, seed):
    np.random.seed(seed)
    g = grid_generator()
    if neq == 0:
        return SubGrid(g, np.arange(g.size(0), dtype=np.int32))
    else:
        return SubGrid(g, np.array(random.sample(xrange(g.size(0)), max(int(m.floor(g.size(0) / neq)), 1))))


subgrid_generators = [lambda args=args: subgrid_factory(*args) for args in
                      [(lambda: RectGrid((1, 1)), 0, 123),
                       (lambda: RectGrid((1, 1)), 2, 123),
                       (lambda: RectGrid((1, 1)), 4, 123),
                       (lambda: TriaGrid((1, 1)), 0, 123),
                       (lambda: TriaGrid((1, 1)), 2, 123),
                       (lambda: TriaGrid((1, 1)), 4, 123),
                       (lambda: RectGrid((8, 8)), 0, 123),
                       (lambda: RectGrid((8, 8)), 2, 123),
                       (lambda: RectGrid((8, 8)), 4, 123),
                       (lambda: TriaGrid((24, 24)), 0, 123),
                       (lambda: TriaGrid((24, 24)), 2, 123),
                       (lambda: TriaGrid((24, 24)), 4, 123)]]


@pytest.fixture(params=(rect_grid_generators + tria_grid_generators + oned_grid_generators + subgrid_generators
                        + unstructured_grid_generators))
def grid(request):
    return request.param()

@pytest.fixture(params=(rect_grid_generators + tria_grid_generators))
def rect_or_tria_grid(request):
    return request.param()

@pytest.fixture(params=(rect_grid_generators + oned_grid_generators))
def grid_with_orthogonal_centers(request):
    return request.param()
