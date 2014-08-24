# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip

import numpy as np
import pytest

from pymor.grids.interfaces import ReferenceElementInterface
from pymortests.fixtures.grid import grid, grid_with_orthogonal_centers

# monkey np.testing.assert_allclose to behave the same as np.allclose
# for some reason, the default atol of np.testing.assert_allclose is 0
# while it is 1e-8 for np.allclose

real_assert_allclose = np.testing.assert_allclose


def monkey_allclose(a, b, rtol=1.e-5, atol=1.e-8):
    real_assert_allclose(a, b, rtol=rtol, atol=atol)
np.testing.assert_allclose = monkey_allclose


def test_dim_outer(grid):
    g = grid
    assert isinstance(g.dim_outer, int)
    assert g.dim_outer >= g.dim


def test_reference_element_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.reference_element(-1)
    with pytest.raises(AssertionError):
        g.reference_element(g.dim + 1)


def test_reference_element_type(grid):
    g = grid
    for d in xrange(g.dim + 1):
        assert isinstance(g.reference_element(d), ReferenceElementInterface)


def test_reference_element_transitivity(grid):
    g = grid
    for d in xrange(1, g.dim + 1):
        assert g.reference_element(d) is g.reference_element(0).sub_reference_element(d)


def test_embeddings_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.embeddings(-1)
    with pytest.raises(AssertionError):
        g.embeddings(g.dim + 1)


def test_embeddings_shape(grid):
    g = grid
    for d in xrange(g.dim + 1):
        RES = g.embeddings(d)
        assert len(RES) == 2
        A, B = RES
        assert A.shape == (g.size(d), g.dim_outer, g.dim - d)
        assert B.shape == (g.size(d), g.dim_outer)


def test_embeddings_transitivity(grid):
    g = grid
    for d in xrange(1, g.dim + 1):
        AD1, BD1 = g.embeddings(d - 1)
        AD, BD = g.embeddings(d)
        SE = g.superentities(d, d - 1)
        SEI = g.superentity_indices(d, d - 1)
        ASUB, BSUB = g.reference_element(d - 1).subentity_embedding(1)
        for e in xrange(g.size(d)):
            np.testing.assert_allclose(AD[e], np.dot(AD1[SE[e, 0]], ASUB[SEI[e, 0]]))
            np.testing.assert_allclose(BD[e], np.dot(AD1[SE[e, 0]], BSUB[SEI[e, 0]]) + BD1[SE[e, 0]])


def test_jacobian_inverse_transposed_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.jacobian_inverse_transposed(-1)
    with pytest.raises(AssertionError):
        g.jacobian_inverse_transposed(g.dim + 1)
    with pytest.raises(AssertionError):
        g.jacobian_inverse_transposed(g.dim)


def test_jacobian_inverse_transposed_shape(grid):
    g = grid
    for d in xrange(g.dim):
        assert g.jacobian_inverse_transposed(d).shape == (g.size(d), g.dim_outer, g.dim - d)


def test_jacobian_inverse_transposed_values(grid):
    g = grid
    for d in xrange(g.dim):
        JIT = g.jacobian_inverse_transposed(d)
        A, _ = g.embeddings(d)
        for e in xrange(g.size(d)):
            np.testing.assert_allclose(JIT[e], np.linalg.pinv(A[e]).T)


def test_integration_elements_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.integration_elements(-1)
    with pytest.raises(AssertionError):
        g.integration_elements(g.dim + 1)


def test_integration_elements_shape(grid):
    g = grid
    for d in xrange(g.dim):
        assert g.integration_elements(d).shape == (g.size(d),)


def test_integration_elements_values(grid):
    g = grid
    for d in xrange(g.dim - 1):
        IE = g.integration_elements(d)
        A, _ = g.embeddings(d)
        for e in xrange(g.size(d)):
            np.testing.assert_allclose(IE[e], np.sqrt(np.linalg.det(np.dot(A[e].T, A[e]))))
    np.testing.assert_allclose(g.integration_elements(g.dim), 1)


def test_volumes_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.volumes(-1)
    with pytest.raises(AssertionError):
        g.volumes(g.dim + 1)


def test_volumes_shape(grid):
    g = grid
    for d in xrange(g.dim):
        assert g.volumes(d).shape == (g.size(d),)


def test_volumes_values(grid):
    g = grid
    for d in xrange(g.dim - 1):
        V = g.volumes(d)
        IE = g.integration_elements(d)
        np.testing.assert_allclose(V, IE * g.reference_element(d).volume)


def test_volumes_inverse_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.volumes_inverse(-1)
    with pytest.raises(AssertionError):
        g.volumes_inverse(g.dim + 1)


def test_volumes_inverse_shape(grid):
    g = grid
    for d in xrange(g.dim):
        assert g.volumes_inverse(d).shape == (g.size(d),)


def test_volumes_inverse_values(grid):
    g = grid
    for d in xrange(g.dim - 1):
        VI = g.volumes_inverse(d)
        V = g.volumes(d)
        np.testing.assert_allclose(VI, np.reciprocal(V))


def test_unit_outer_normals_shape(grid):
    g = grid
    SE = g.subentities(0, 1)
    assert g.unit_outer_normals().shape == SE.shape + (g.dim_outer,)


def test_unit_outer_normals_normed(grid):
    g = grid
    UON = g.unit_outer_normals()
    np.testing.assert_allclose(np.sum(UON ** 2, axis=-1), 1)


def test_unit_outer_normals_normal(grid):
    g = grid
    SE = g.subentities(0, 1)
    A, _ = g.embeddings(1)
    SEE = A[SE, ...]
    UON = g.unit_outer_normals()
    np.testing.assert_allclose(np.sum(SEE * UON[..., np.newaxis], axis=-2), 0)


def test_unit_outer_normals_neighbours(grid):
    g = grid
    UON = g.unit_outer_normals()
    SE = g.superentities(1, 0)
    SEI = g.superentity_indices(1, 0)
    if SE.shape[1] < 2:
        return
    for se, sei in izip(SE, SEI):
        if se[0] == -1 or se[1] == -1:
            continue
        np.testing.assert_allclose(UON[se[0], sei[0]], -UON[se[1], sei[1]])


def test_centers_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.centers(-1)
    with pytest.raises(AssertionError):
        g.centers(g.dim + 1)


def test_centers_shape(grid):
    g = grid
    for d in xrange(g.dim):
        assert g.centers(d).shape == (g.size(d), g.dim_outer)


def test_centers_values(grid):
    g = grid
    for d in xrange(g.dim):
        A, B = g.embeddings(d)
        np.testing.assert_allclose(g.centers(d), B + A.dot(g.reference_element(d).center()))


def test_diameters_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.diameters(-1)
    with pytest.raises(AssertionError):
        g.diameters(g.dim + 1)


def test_diameters_shape(grid):
    g = grid
    for d in xrange(g.dim):
        assert g.diameters(d).shape == (g.size(d),)


def test_diameters_non_negative(grid):
    g = grid
    for d in xrange(g.dim - 1):
        assert np.min(g.diameters(d)) >= 0


def test_diameters_values(grid):
    g = grid
    for d in xrange(g.dim - 1):
        A, _ = g.embeddings(d)
        np.testing.assert_allclose(g.diameters(d), g.reference_element(d).mapped_diameter(A))


def test_quadrature_points_wrong_arguments(grid):
    g = grid
    for d in xrange(g.dim):
        with pytest.raises(Exception):
            g.quadrature_points(d, order=1, npoints=1)
        with pytest.raises(Exception):
            g.quadrature_points(d)
        os, ps = g.reference_element(d).quadrature_info()
        for t in os.keys():
            with pytest.raises(Exception):
                g.quadrature_points(d, order=max(os[t]) + 1, quadrature_type=t)
            with pytest.raises(Exception):
                g.quadrature_points(d, npoints=max(ps[t]) + 1, quadrature_type=t)


def test_quadrature_points_shape(grid):
    g = grid
    for d in xrange(g.dim):
        os, ps = g.reference_element(d).quadrature_info()
        for t in os.keys():
            for o, p in izip(os[t], ps[t]):
                assert g.quadrature_points(d, order=o, quadrature_type=t).shape == (g.size(d), p, g.dim_outer)
                assert g.quadrature_points(d, npoints=p, quadrature_type=t).shape == (g.size(d), p, g.dim_outer)


def test_quadrature_points_values(grid):
    g = grid
    for d in xrange(g.dim):
        A, B = g.embeddings(d)
        os, ps = g.reference_element(d).quadrature_info()
        for t in os.keys():
            for o, p in izip(os[t], ps[t]):
                Q = g.quadrature_points(d, order=o, quadrature_type=t)
                q, _ = g.reference_element(d).quadrature(order=o, quadrature_type=t)
                np.testing.assert_allclose(Q, g.quadrature_points(d, npoints=p, quadrature_type=t))
                np.testing.assert_allclose(Q, B[:, np.newaxis, :] + np.einsum('eij,qj->eqi', A, q))


def test_orthogonal_centers(grid_with_orthogonal_centers):
    g = grid_with_orthogonal_centers
    C = g.orthogonal_centers()
    SUE = g.superentities(1, 0)
    if SUE.shape[1] != 2:
        return
    EMB = g.embeddings(1)[0].swapaxes(1, 2)
    for s in xrange(g.size(1)):
        if -1 in SUE[s]:
            continue
        SEGMENT = C[SUE[s, 0]] - C[SUE[s, 1]]
        SPROD = EMB[s].dot(SEGMENT)
        np.testing.assert_allclose(SPROD, 0)
