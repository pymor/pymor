# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pickle import dumps, loads
from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings

from pymor.core.exceptions import QtMissing
from pymor.discretizers.builtin.gui.qt import stop_gui_processes
from pymortests.fixtures.grid import hy_grids_with_visualize, hy_grid
from pymortests.pickling import assert_picklable_without_dumps_function


# monkey np.testing.assert_allclose to behave the same as np.allclose
# for some reason, the default atol of np.testing.assert_allclose is 0
# while it is 1e-8 for np.allclose

real_assert_allclose = np.testing.assert_allclose


def monkey_allclose(a, b, rtol=1.e-5, atol=1.e-8):
    real_assert_allclose(a, b, rtol=rtol, atol=atol)
np.testing.assert_allclose = monkey_allclose


@given(hy_grid)
def test_dim(grid):
    g = grid
    assert isinstance(g.dim, int)
    assert g.dim >= 0


@given(hy_grid)
def test_size(grid):
    g = grid
    for d in range(g.dim + 1):
        assert g.size(d) >= 0
    with pytest.raises(AssertionError):
        g.size(-1)
    with pytest.raises(AssertionError):
        g.size(g.dim + 1)


@given(hy_grid)
def test_subentities_wrong_arguments(grid):
    g = grid
    for e in range(g.dim + 1):
        with pytest.raises(AssertionError):
            g.subentities(e, g.dim + 1)
        with pytest.raises(AssertionError):
            g.subentities(e, e - 1)
    with pytest.raises(AssertionError):
        g.subentities(g.dim + 1, 0)
    with pytest.raises(AssertionError):
        g.subentities(-1, 0)


@given(hy_grid)
def test_subentities_shape(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e, g.dim + 1):
            assert g.subentities(e, s).ndim == 2
            assert g.subentities(e, s).shape[0] == g.size(e)


@given(hy_grid)
def test_subentities_dtype(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e, g.dim + 1):
            assert g.subentities(e, s).dtype == np.dtype('int32')


@given(hy_grid)
def test_subentities_entry_value_range(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e, g.dim + 1):
            np.testing.assert_array_less(g.subentities(e, s), g.size(s))
            np.testing.assert_array_less(-2, g.subentities(e, s))


@settings(deadline=None)
@given(hy_grid)
def test_subentities_entry_values_unique(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e, g.dim + 1):
            for S in g.subentities(e, s):
                S = S[S >= 0]
                assert S.size == np.unique(S).size


@given(hy_grid)
def test_subentities_codim_d_codim_d(grid):
    g = grid
    for d in range(g.dim + 1):
        assert g.subentities(d, d).shape == (g.size(d), 1)
        np.testing.assert_array_equal(g.subentities(d, d).ravel(), np.arange(g.size(d)))


@given(hy_grid)
def test_subentities_transitivity(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e + 1, g.dim + 1):
            for ss in range(s + 1, g.dim + 1):
                SE = g.subentities(e, s)
                SSE = g.subentities(e, ss)
                SESE = g.subentities(s, ss)
                for i in range(SE.shape[0]):
                    for j in range(SE.shape[1]):
                        if SE[i, j] != -1:
                            assert set(SESE[SE[i, j]]) <= set(SSE[i]).union((-1,))


@given(hy_grid)
def test_superentities_wrong_arguments(grid):
    g = grid
    for e in range(g.dim + 1):
        with pytest.raises(AssertionError):
            g.superentities(e, -1)
        with pytest.raises(AssertionError):
            g.superentities(e, e + 1)
    with pytest.raises(AssertionError):
        g.superentities(g.dim + 1, 0)
    with pytest.raises(AssertionError):
        g.superentities(-1, 0)
    with pytest.raises(AssertionError):
        g.superentities(-1, -2)


@given(hy_grid)
def test_superentities_shape(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            assert g.superentities(e, s).ndim == 2
            assert g.superentities(e, s).shape[0] == g.size(e)
            assert g.superentities(e, s).shape[1] > 0
    assert g.superentities(1, 0).shape[1] <= 2


@given(hy_grid)
def test_superentities_dtype(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            assert g.superentities(e, s).dtype == np.dtype('int32')


@given(hy_grid)
def test_superentities_entry_value_range(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            np.testing.assert_array_less(g.superentities(e, s), g.size(s))
            np.testing.assert_array_less(-2, g.superentities(e, s))


@given(hy_grid)
def test_superentities_entry_values_unique(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            for S in g.superentities(e, s):
                S = S[S >= 0]
                assert S.size == np.unique(S).size


@given(hy_grid)
def test_superentities_entries_sorted(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            for S in g.superentities(e, s):
                i = 0
                while (i + 1 < len(S)) and (S[i] < S[i + 1]):
                    i += 1
                assert (i + 1 == len(S)) or (S[i + 1] == -1)
                if i + 1 < len(S):
                    np.testing.assert_array_equal(S[i + 1:], -1)


@given(hy_grid)
def test_superentities_codim_d_codim_d(grid):
    g = grid
    for d in range(g.dim + 1):
        assert g.superentities(d, d).shape == (g.size(d), 1)
        np.testing.assert_array_equal(g.superentities(d, d).ravel(), np.arange(g.size(d)))


@given(hy_grid)
def test_superentities_each_entry_superentity(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            SE = g.superentities(e, s)
            SUBE = g.subentities(s, e)
            for i in range(SE.shape[0]):
                for se in SE[i]:
                    if se != -1:
                        assert i in SUBE[se]


@given(hy_grid)
def test_superentities_each_superentity_has_entry(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            SE = g.superentities(e, s)
            SUBE = g.subentities(s, e)
            for i in range(SUBE.shape[0]):
                for se in SUBE[i]:
                    if se != -1:
                        assert i in SE[se]


@given(hy_grid)
def test_superentity_indices_wrong_arguments(grid):
    g = grid
    for e in range(g.dim + 1):
        with pytest.raises(AssertionError):
            g.superentity_indices(e, -1)
        with pytest.raises(AssertionError):
            g.superentity_indices(e, e + 1)
    with pytest.raises(AssertionError):
        g.superentity_indices(g.dim + 1, 0)
    with pytest.raises(AssertionError):
        g.superentity_indices(-1, 0)
    with pytest.raises(AssertionError):
        g.superentity_indices(-1, -2)


@given(hy_grid)
def test_superentity_indices_shape(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            assert g.superentity_indices(e, s).shape == g.superentities(e, s).shape


@given(hy_grid)
def test_superentity_indices_dtype(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            assert g.superentity_indices(e, s).dtype == np.dtype('int32')


@given(hy_grid)
def test_superentity_indices_valid_entries(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e):
            SE = g.superentities(e, s)
            SEI = g.superentity_indices(e, s)
            SUBE = g.subentities(s, e)
            for index, superentity in np.ndenumerate(SE):
                if superentity > -1:
                    assert SUBE[superentity, SEI[index]] == index[0]


@given(hy_grid)
def test_neighbours_wrong_arguments(grid):
    g = grid
    for e in range(g.dim + 1):
        for n in range(g.dim + 1):
            with pytest.raises(AssertionError):
                g.neighbours(e, n, -1)
            with pytest.raises(AssertionError):
                g.neighbours(e, n, g.dim + 1)
            with pytest.raises(AssertionError):
                g.neighbours(e, n, e - 1)
            with pytest.raises(AssertionError):
                g.neighbours(e, n, n - 1)
        with pytest.raises(AssertionError):
            g.neighbours(e, g.dim + 1, g.dim)
        with pytest.raises(AssertionError):
            g.neighbours(e, -1, g.dim)
    with pytest.raises(AssertionError):
        g.neighbours(g.dim + 1, g.dim, g.dim)
    with pytest.raises(AssertionError):
        g.neighbours(-1, 0, g.dim)


# TODO product -> strategy
@settings(deadline=None)
@given(hy_grid)
def test_neighbours_shape(grid):
    g = grid
    for e, n in product(range(g.dim + 1), range(g.dim + 1)):
        for s in range(max(e, n), g.dim + 1):
            assert g.neighbours(e, n, s).ndim == 2
            assert g.neighbours(e, n, s).shape[0] == g.size(e)


# TODO product -> strategy
@settings(deadline=None)
@given(hy_grid)
def test_neighbours_dtype(grid):
    g = grid
    for e, n in product(range(g.dim + 1), range(g.dim + 1)):
        for s in range(max(e, n), g.dim + 1):
            assert g.neighbours(e, n, s).dtype == np.dtype('int32')


# TODO product -> strategy
@settings(deadline=None)
@given(hy_grid)
def test_neighbours_entry_value_range(grid):
    g = grid
    for e, n in product(range(g.dim + 1), range(g.dim + 1)):
        for s in range(max(e, n), g.dim + 1):
            np.testing.assert_array_less(g.neighbours(e, n, s), g.size(n))
            np.testing.assert_array_less(-2, g.neighbours(e, n, s))


# TODO product -> strategy
@settings(deadline=None)
@given(hy_grid)
def test_neighbours_entry_values_unique(grid):
    g = grid
    for e, n in product(range(g.dim + 1), range(g.dim + 1)):
        for s in range(max(e, n), g.dim + 1):
            for S in g.neighbours(e, n, s):
                S = S[S >= 0]
                assert S.size == np.unique(S).size


# TODO product -> strategy
@settings(deadline=None)
@given(hy_grid)
def test_neighbours_each_entry_neighbour(grid):
    g = grid
    for e, n in product(range(g.dim + 1), range(g.dim + 1)):
        for s in range(max(e, n), g.dim + 1):
            N = g.neighbours(e, n, s)
            ESE = g.subentities(e, s)
            NSE = g.subentities(n, s)
            for index, neigh in np.ndenumerate(N):
                if neigh > -1:
                    inter = set(ESE[index[0]]).intersection(set(NSE[neigh]))
                    if -1 in inter:
                        assert len(inter) > 1
                    else:
                        assert len(inter) > 0


# TODO product -> strategy
@settings(deadline=None)
@given(hy_grid)
def test_neighbours_each_neighbour_has_entry(grid):
    g = grid
    for e, n in product(range(g.dim + 1), range(g.dim + 1)):
        for s in range(max(e, n), g.dim + 1):
            N = g.neighbours(e, n, s)
            SUE = g.superentities(s, e)
            SUN = g.superentities(s, n)
            if e != n:
                for si in range(SUE.shape[0]):
                    for ei, ni in product(SUE[si], SUN[si]):
                        if ei != -1 and ni != -1:
                            assert ni in N[ei],\
                                f'Failed for\n{g}\ne={e}, n={n}, s={s}, ei={ei}, ni={ni}'
            else:
                for si in range(SUE.shape[0]):
                    for ei, ni in product(SUE[si], SUN[si]):
                        if ei != ni and ei != -1 and ni != -1:
                            assert ni in N[ei],\
                                f'Failed for\n{g}\ne={e}, n={n}, s={s}, ei={ei}, ni={ni}'


@settings(deadline=None)
@given(hy_grid)
def test_neighbours_not_neighbour_of_itself(grid):
    g = grid
    for e in range(g.dim + 1):
        for s in range(e, g.dim + 1):
            N = g.neighbours(e, e, s)
            for ei, E in enumerate(N):
                assert ei not in E,\
                    f'Failed for\n{g}\ne={e}, s={s}, ei={ei}, E={E}'


@given(hy_grid)
def test_boundary_mask_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.boundary_mask(-1)
    with pytest.raises(AssertionError):
        g.boundary_mask(g.dim + 1)


@given(hy_grid)
def test_boundary_mask_shape(grid):
    g = grid
    for d in range(g.dim + 1):
        assert g.boundary_mask(d).shape == (g.size(d),)


@given(hy_grid)
def test_boundary_mask_dtype(grid):
    g = grid
    for d in range(g.dim + 1):
        g.boundary_mask(d).dtype == np.dtype('bool')


@given(hy_grid)
def test_boundary_mask_entries_codim1(grid):
    g = grid
    BM = g.boundary_mask(1)
    SE = g.superentities(1, 0)
    for ei, b in enumerate(BM):
        E = SE[ei]
        assert (E[E > -1].size <= 1) == b


@given(hy_grid)
def test_boundary_mask_entries_codim0(grid):
    g = grid
    BM0 = g.boundary_mask(0)
    BM1 = g.boundary_mask(1)
    SE = g.subentities(0, 1)
    for ei, b in enumerate(BM0):
        S = SE[ei]
        assert np.any(BM1[S[S > -1]]) == b


@given(hy_grid)
def test_boundary_mask_entries_codim_d(grid):
    g = grid
    for d in range(2, g.dim + 1):
        BMD = g.boundary_mask(d)
        BM1 = g.boundary_mask(1)
        SE = g.superentities(d, 1)
        for ei, b in enumerate(BMD):
            S = SE[ei]
            assert np.any(BM1[S[S > -1]]) == b


@given(hy_grid)
def test_boundaries_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.boundaries(-1)
    with pytest.raises(AssertionError):
        g.boundaries(g.dim + 1)


@given(hy_grid)
def test_boundaries_shape(grid):
    g = grid
    for d in range(g.dim + 1):
        assert len(g.boundaries(d).shape) == 1


@given(hy_grid)
def test_boundaries_dtype(grid):
    g = grid
    for d in range(g.dim + 1):
        assert g.boundaries(d).dtype == np.dtype('int32')


@given(hy_grid)
def test_boundaries_entry_value_range(grid):
    g = grid
    for d in range(g.dim + 1):
        np.testing.assert_array_less(g.boundaries(d), g.size(d))
        np.testing.assert_array_less(-1, g.boundaries(d))


@given(hy_grid)
def test_boundaries_entries(grid):
    g = grid
    for d in range(g.dim + 1):
        np.testing.assert_array_equal(np.where(g.boundary_mask(d))[0], g.boundaries(d))


@given(hy_grid)
def test_pickle(grid):
    assert_picklable_without_dumps_function(grid)


@settings(deadline=None)
@given(hy_grids_with_visualize)
def test_visualize(grids_with_visualize):
    import sys
    sys._called_from_test = True

    def nop(*args, **kwargs):
        pass
    try:
        from matplotlib import pyplot
        pyplot.show = nop
    except ImportError:
        pass

    try:
        g = grids_with_visualize
        U = np.ones(g.size(g.dim))
        g.visualize(U, g.dim)
    except QtMissing:
        pytest.xfail("Qt missing")
    finally:
        stop_gui_processes()
