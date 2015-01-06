# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from pymor.la.numpyvectorarray import NumpyVectorArray
from pymortests.algorithms import MonomOperator
from pymortests.fixtures.operator import operator, operator_with_arrays, operator_with_arrays_and_products
from pymortests.vectorarray import valid_inds
from pymortests.pickle import assert_picklable, assert_picklable_without_dumps_function


def test_lincomb_op():
    p1 = MonomOperator(1)
    p2 = MonomOperator(2)
    p12 = p1 + p2
    p0 = p1 - p1
    x = np.linspace(-1., 1., num=3)
    vx = NumpyVectorArray(x[:, np.newaxis])
    assert np.allclose(p0.apply(vx).data, [0.])
    assert np.allclose(p12.apply(vx).data, (x * x + x)[:, np.newaxis])
    assert np.allclose((p1 * 2.).apply(vx).data, (x * 2.)[:, np.newaxis])
    assert p2.jacobian(vx).apply(vx).almost_equal(p1.apply(vx) * 2.).all()
    assert p0.jacobian(vx).apply(vx).almost_equal(vx * 0.).all()
    with pytest.raises(TypeError):
        p2.as_vector()
    p1.as_vector()
    assert p1.as_vector().almost_equal(p1.apply(NumpyVectorArray(1.)))

    basis = NumpyVectorArray([1.])
    for p in (p1, p2, p12):
        projected = p.projected(basis, basis)
        pa = projected.apply(vx)
        assert pa.almost_equal(p.apply(vx)).all()


def test_pickle(operator):
    assert_picklable(operator)


def test_pickle_without_dumps_function(operator):
    assert_picklable_without_dumps_function(operator)


def test_apply(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    V = op.apply(U, mu=mu)
    assert V in op.range
    assert len(V) == len(U)
    for ind in list(valid_inds(U, 3)) + [[]]:
        Vind = op.apply(U, mu=mu, ind=ind)
        assert np.all(Vind.almost_equal(V, o_ind=ind))
        assert np.all(Vind.almost_equal(op.apply(U.copy(ind=ind), mu=mu)))


def test_apply_adjoint(operator_with_arrays):
    op, mu, _, V = operator_with_arrays
    try:
        U = op.apply_adjoint(V, mu=mu)
    except NotImplementedError:
        return
    assert U in op.source
    assert len(V) == len(U)
    for ind in list(valid_inds(V, 3)) + [[]]:
        Uind = op.apply_adjoint(V, mu=mu, ind=ind)
        assert np.all(Uind.almost_equal(U, o_ind=ind))
        assert np.all(Uind.almost_equal(op.apply_adjoint(V.copy(ind=ind), mu=mu)))


def test_apply_adjoint_2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    try:
        ATV = op.apply_adjoint(V, mu=mu)
    except NotImplementedError:
        return
    assert np.allclose(V.dot(op.apply(U, mu=mu), pairwise=False), ATV.dot(U, pairwise=False))


def test_apply_adjoint_2_with_products(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    try:
        ATV = op.apply_adjoint(V, mu=mu, source_product=sp, range_product=rp)
    except NotImplementedError:
        return
    assert np.allclose(rp.apply2(V, op.apply(U, mu=mu), pairwise=False),
                       sp.apply2(ATV, U, pairwise=False))


def test_projected(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    op_UV = op.projected(U, V)
    np.random.seed(4711 + U.dim + len(V))
    coeffs = np.random.random(len(U))
    X = op_UV.apply(NumpyVectorArray(coeffs, copy=False), mu=mu)
    Y = NumpyVectorArray(V.dot(op.apply(U.lincomb(coeffs), mu=mu), pairwise=False).T, copy=False)
    assert np.all(X.almost_equal(Y))


def test_projected_2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    op_U = op.projected(U, None)
    op_V = op.projected(None, V)
    op_U_V = op_U.projected(None, V)
    op_V_U = op_V.projected(U, None)
    op_UV = op.projected(U, V)
    np.random.seed(4711 + U.dim + len(V))
    W = NumpyVectorArray(np.random.random(len(U)), copy=False)
    Y0 = op_UV.apply(W, mu=mu)
    Y1 = op_U_V.apply(W, mu=mu)
    Y2 = op_V_U.apply(W, mu=mu)
    assert np.all(Y0.almost_equal(Y1))
    assert np.all(Y0.almost_equal(Y2))


def test_projected_with_product(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    op_UV = op.projected(U, V, product=rp)
    np.random.seed(4711 + U.dim + len(V))
    coeffs = np.random.random(len(U))
    X = op_UV.apply(NumpyVectorArray(coeffs, copy=False), mu=mu)
    Y = NumpyVectorArray(rp.apply2(op.apply(U.lincomb(coeffs), mu=mu), V, pairwise=False), copy=False)
    assert np.all(X.almost_equal(Y))


def test_projected_with_product_2(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    op_U = op.projected(U, None)
    op_V = op.projected(None, V, product=rp)
    op_U_V = op_U.projected(None, V, product=rp)
    op_V_U = op_V.projected(U, None)
    op_UV = op.projected(U, V, product=rp)
    np.random.seed(4711 + U.dim + len(V))
    W = NumpyVectorArray(np.random.random(len(U)), copy=False)
    Y0 = op_UV.apply(W, mu=mu)
    Y1 = op_U_V.apply(W, mu=mu)
    Y2 = op_V_U.apply(W, mu=mu)
    assert np.all(Y0.almost_equal(Y1))
    assert np.all(Y0.almost_equal(Y2))
