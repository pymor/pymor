# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from hypothesis import given, settings

from pymor.discretizers.builtin.grids.interfaces import ReferenceElement
from pymor.tools.floatcmp import almost_less
from pymortests.base import runmodule
from pymortests.fixtures.grid import hy_grid, hy_grid_with_orthogonal_centers, \
    hy_grid_and_codim_product_and_entity_index


def _scale_tols_if_domain_bad(g, atol=1e-05, rtol=1e-08):
    # "badly" shaped domains produce excessive errors
    # same for large differences in absolute coord values
    bbox = g.bounding_box()
    scale = 1.0
    lower_left, upper_right = bbox[0], bbox[1]
    magic_downscale = 1e-3
    if g.dim == 2:
        upper_left = np.array([lower_left[0], upper_right[1]])
        lower_right = np.array([lower_left[1], upper_right[0]])
        h = np.linalg.norm(upper_left - lower_left)
        w = np.linalg.norm(lower_right - lower_left)
        min_l = min(w, h)
        max_l = max(w, h)
        ll, rr = np.linalg.norm(lower_left), np.linalg.norm(upper_right)
        scale = max(max_l / min_l, abs(rr - ll) * magic_downscale)
    if g.dim == 1:
        ratio = abs(upper_right) / abs(lower_left)
        if not np.isfinite(ratio):
            ratio = abs(upper_right)*magic_downscale
        scale = max(ratio, abs(upper_right-lower_left)*magic_downscale)[0]
    if scale > 10:
        rtol *= scale
        atol *= scale
    assert np.isfinite(atol)
    assert np.isfinite(rtol)
    return atol, rtol


@given(hy_grid)
def test_reference_element_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.reference_element(-1)
    with pytest.raises(AssertionError):
        g.reference_element(g.dim + 1)


@given(hy_grid)
def test_reference_element_type(grid):
    g = grid
    for d in range(g.dim + 1):
        assert isinstance(g.reference_element(d), ReferenceElement)


@given(hy_grid)
def test_reference_element_transitivity(grid):
    g = grid
    for d in range(1, g.dim + 1):
        assert g.reference_element(d) is g.reference_element(0).sub_reference_element(d)


@given(hy_grid)
def test_embeddings_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.embeddings(-1)
    with pytest.raises(AssertionError):
        g.embeddings(g.dim + 1)


@given(hy_grid)
def test_embeddings_shape(grid):
    g = grid
    for d in range(g.dim + 1):
        RES = g.embeddings(d)
        assert len(RES) == 2
        A, B = RES
        assert A.shape == (g.size(d), g.dim, g.dim - d)
        assert B.shape == (g.size(d), g.dim)


@settings(deadline=None)
@given(hy_grid)
def test_embeddings_transitivity(grid):
    g = grid
    for d in range(1, g.dim + 1):
        AD1, BD1 = g.embeddings(d - 1)
        AD, BD = g.embeddings(d)
        SE = g.superentities(d, d - 1)
        SEI = g.superentity_indices(d, d - 1)
        ASUB, BSUB = g.reference_element(d - 1).subentity_embedding(1)
        for e in range(g.size(d)):
            np.testing.assert_allclose(AD[e], np.dot(AD1[SE[e, 0]], ASUB[SEI[e, 0]]))
            np.testing.assert_allclose(BD[e], np.dot(AD1[SE[e, 0]], BSUB[SEI[e, 0]]) + BD1[SE[e, 0]])


@given(hy_grid)
def test_jacobian_inverse_transposed_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.jacobian_inverse_transposed(-1)
    with pytest.raises(AssertionError):
        g.jacobian_inverse_transposed(g.dim + 1)
    with pytest.raises(AssertionError):
        g.jacobian_inverse_transposed(g.dim)


@settings(deadline=None)
@given(hy_grid)
def test_jacobian_inverse_transposed_shape(grid):
    g = grid
    for d in range(g.dim):
        assert g.jacobian_inverse_transposed(d).shape == (g.size(d), g.dim, g.dim - d)


@settings(deadline=None)
@given(hy_grid_and_codim_product_and_entity_index())
def test_jacobian_inverse_transposed_values(grid_and_dims):
    g, d, e = grid_and_dims
    atol, rtol = _scale_tols_if_domain_bad(g)
    JIT = g.jacobian_inverse_transposed(d)
    A, _ = g.embeddings(d)
    np.testing.assert_allclose(JIT[e], np.linalg.pinv(A[e]).T, atol=atol, rtol=rtol)


@given(hy_grid)
def test_integration_elements_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.integration_elements(-1)
    with pytest.raises(AssertionError):
        g.integration_elements(g.dim + 1)


@given(hy_grid)
def test_integration_elements_shape(grid):
    g = grid
    for d in range(g.dim):
        assert g.integration_elements(d).shape == (g.size(d),)


@given(hy_grid_and_codim_product_and_entity_index())
def test_integration_elements_values(grid_and_dims):
    g, d, e = grid_and_dims
    atol, rtol = _scale_tols_if_domain_bad(g)
    IE = g.integration_elements(d)
    A, _ = g.embeddings(d)
    np.testing.assert_allclose(IE[e], np.sqrt(np.linalg.det(np.dot(A[e].T, A[e]))),
                               atol=atol, rtol=rtol)
    np.testing.assert_allclose(g.integration_elements(g.dim), 1,
                               atol=atol, rtol=rtol)


@given(hy_grid)
def test_volumes_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.volumes(-1)
    with pytest.raises(AssertionError):
        g.volumes(g.dim + 1)


@given(hy_grid)
def test_volumes_shape(grid):
    g = grid
    for d in range(g.dim):
        assert g.volumes(d).shape == (g.size(d),)


@given(hy_grid)
def test_volumes_values(grid):
    g = grid
    for d in range(g.dim - 1):
        V = g.volumes(d)
        IE = g.integration_elements(d)
        np.testing.assert_allclose(V, IE * g.reference_element(d).volume)


@given(hy_grid)
def test_volumes_inverse_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.volumes_inverse(-1)
    with pytest.raises(AssertionError):
        g.volumes_inverse(g.dim + 1)


@given(hy_grid)
def test_volumes_inverse_shape(grid):
    g = grid
    for d in range(g.dim):
        assert g.volumes_inverse(d).shape == (g.size(d),)


@given(hy_grid)
def test_volumes_inverse_values(grid):
    g = grid
    for d in range(g.dim - 1):
        VI = g.volumes_inverse(d)
        V = g.volumes(d)
        np.testing.assert_allclose(VI, np.reciprocal(V))


@given(hy_grid)
@settings(deadline=None)
def test_unit_outer_normals_shape(grid):
    g = grid
    SE = g.subentities(0, 1)
    assert g.unit_outer_normals().shape == SE.shape + (g.dim,)


@settings(deadline=None)
@given(hy_grid)
def test_unit_outer_normals_normed(grid):
    g = grid
    UON = g.unit_outer_normals()
    np.testing.assert_allclose(np.sum(UON ** 2, axis=-1), 1)


@settings(deadline=None)
@given(hy_grid)
def test_unit_outer_normals_normal(grid):
    g = grid
    SE = g.subentities(0, 1)
    A, _ = g.embeddings(1)
    SEE = A[SE, ...]
    UON = g.unit_outer_normals()
    np.testing.assert_allclose(np.sum(SEE * UON[..., np.newaxis], axis=-2), 0, atol=1e7)


@settings(deadline=None)
@given(hy_grid)
def test_unit_outer_normals_neighbours(grid):
    g = grid
    UON = g.unit_outer_normals()
    SE = g.superentities(1, 0)
    SEI = g.superentity_indices(1, 0)
    if SE.shape[1] < 2:
        return
    for se, sei in zip(SE, SEI):
        if se[0] == -1 or se[1] == -1:
            continue
        np.testing.assert_allclose(UON[se[0], sei[0]], -UON[se[1], sei[1]])


@given(hy_grid)
def test_centers_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.centers(-1)
    with pytest.raises(AssertionError):
        g.centers(g.dim + 1)


@given(hy_grid)
def test_centers_shape(grid):
    g = grid
    for d in range(g.dim):
        assert g.centers(d).shape == (g.size(d), g.dim)


@given(hy_grid)
def test_centers_values(grid):
    g = grid
    for d in range(g.dim):
        A, B = g.embeddings(d)
        np.testing.assert_allclose(g.centers(d), B + A.dot(g.reference_element(d).center()))


@given(hy_grid)
def test_diameters_wrong_arguments(grid):
    g = grid
    with pytest.raises(AssertionError):
        g.diameters(-1)
    with pytest.raises(AssertionError):
        g.diameters(g.dim + 1)


@given(hy_grid)
def test_diameters_shape(grid):
    g = grid
    for d in range(g.dim):
        assert g.diameters(d).shape == (g.size(d),)


@given(hy_grid)
def test_diameters_non_negative(grid):
    g = grid
    for d in range(g.dim - 1):
        assert np.min(g.diameters(d)) >= 0


@given(hy_grid)
def test_diameters_values(grid):
    g = grid
    for d in range(g.dim - 1):
        A, _ = g.embeddings(d)
        np.testing.assert_allclose(g.diameters(d), g.reference_element(d).mapped_diameter(A))


@given(hy_grid)
def test_quadrature_points_wrong_arguments(grid):
    g = grid
    for d in range(g.dim):
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


@given(hy_grid)
def test_quadrature_points_shape(grid):
    g = grid
    for d in range(g.dim):
        os, ps = g.reference_element(d).quadrature_info()
        for t in os.keys():
            for o, p in zip(os[t], ps[t]):
                assert g.quadrature_points(d, order=o, quadrature_type=t).shape == (g.size(d), p, g.dim)
                assert g.quadrature_points(d, npoints=p, quadrature_type=t).shape == (g.size(d), p, g.dim)


@given(hy_grid)
def test_quadrature_points_values(grid):
    g = grid
    for d in range(g.dim):
        A, B = g.embeddings(d)
        os, ps = g.reference_element(d).quadrature_info()
        for t in os.keys():
            for o, p in zip(os[t], ps[t]):
                Q = g.quadrature_points(d, order=o, quadrature_type=t)
                q, _ = g.reference_element(d).quadrature(order=o, quadrature_type=t)
                np.testing.assert_allclose(Q, g.quadrature_points(d, npoints=p, quadrature_type=t))
                np.testing.assert_allclose(Q, B[:, np.newaxis, :] + np.einsum('eij,qj->eqi', A, q))


@given(hy_grid)
def test_bounding_box(grid):
    g = grid
    bbox = g.bounding_box()
    assert bbox.shape == (2, g.dim)
    assert np.all(bbox[0] <= bbox[1])
    # compare with tolerance is necessary with very large domain boundaries values
    # where the relative error in the centers computation introduces enough error to fail the test
    # otherwise
    rtol, atol = _scale_tols_if_domain_bad(g, rtol=2e-12, atol=2e-12)
    assert np.all(almost_less(bbox[0], g.centers(g.dim), rtol=rtol, atol=atol))
    assert np.all(almost_less(g.centers(g.dim), bbox[1], rtol=rtol, atol=atol))


@settings(deadline=None)
@given(hy_grid_with_orthogonal_centers)
def test_orthogonal_centers(grid_with_orthogonal_centers):
    g = grid_with_orthogonal_centers
    C = g.orthogonal_centers()
    SUE = g.superentities(1, 0)
    if SUE.shape[1] != 2:
        return
    EMB = g.embeddings(1)[0].swapaxes(1, 2)
    for s in range(g.size(1)):
        if -1 in SUE[s]:
            continue
        SEGMENT = C[SUE[s, 0]] - C[SUE[s, 1]]
        SPROD = EMB[s].dot(SEGMENT)
        np.testing.assert_allclose(SPROD, 0)


if __name__ == "__main__":
    runmodule(filename=__file__)
