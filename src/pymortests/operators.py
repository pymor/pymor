# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Andreas Buhr <andreas@andreasbuhr.de>

from __future__ import absolute_import, division, print_function

from itertools import chain
import numpy as np
import pytest

from pymor.algorithms.basic import almost_equal
from pymor.core.exceptions import InversionError
from pymor.operators.constructions import SelectionOperator
from pymor.parameters.base import ParameterType
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.tools.floatcmp import float_cmp_all
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymortests.algorithms.stuff import MonomOperator
from pymortests.fixtures.operator import operator, operator_with_arrays, operator_with_arrays_and_products
from pymortests.pickle import assert_picklable, assert_picklable_without_dumps_function
from pymortests.vectorarray import valid_inds, valid_inds_of_same_length, invalid_inds


def test_selection_op():
    p1 = MonomOperator(1)
    select_rhs_functional = GenericParameterFunctional(
        lambda x: round(float(x["nrrhs"])), 
        ParameterType({"nrrhs" : tuple()})
    )
    s1 = SelectionOperator(
        operators = [p1], 
        boundaries = [], 
        parameter_functional = select_rhs_functional,
        name = "foo"
    )
    x = np.linspace(-1., 1., num=3)
    vx = NumpyVectorArray(x[:, np.newaxis])
    assert np.allclose(p1.apply(vx,mu=0).data, s1.apply(vx,mu=0).data)

    s2 = SelectionOperator(
        operators = [p1,p1,p1,p1],
        boundaries = [-3, 3, 7],
        parameter_functional = select_rhs_functional,
        name = "Bar"
    )

    assert s2._get_operator_number({"nrrhs":-4}) == 0
    assert s2._get_operator_number({"nrrhs":-3}) == 0
    assert s2._get_operator_number({"nrrhs":-2}) == 1
    assert s2._get_operator_number({"nrrhs":3}) == 1
    assert s2._get_operator_number({"nrrhs":4}) == 2
    assert s2._get_operator_number({"nrrhs":7}) == 2
    assert s2._get_operator_number({"nrrhs":9}) == 3

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
    assert almost_equal(p2.jacobian(vx).apply(vx), p1.apply(vx) * 2.).all()
    assert almost_equal(p0.jacobian(vx).apply(vx), vx * 0.).all()
    with pytest.raises(TypeError):
        p2.as_vector()
    p1.as_vector()
    assert almost_equal(p1.as_vector(), p1.apply(NumpyVectorArray(1.)))

    basis = NumpyVectorArray([1.])
    for p in (p1, p2, p12):
        projected = p.projected(basis, basis)
        pa = projected.apply(vx)
        assert almost_equal(pa, p.apply(vx)).all()


def test_pickle(operator):
    assert_picklable(operator)


def test_pickle_without_dumps_function(operator):
    assert_picklable_without_dumps_function(operator)


def test_apply(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    V = op.apply(U, mu=mu)
    assert V in op.range
    assert len(V) == len(U)
    for ind in valid_inds(U):
        Vind = op.apply(U, mu=mu, ind=ind)
        assert np.all(almost_equal(Vind, V, V_ind=ind))
        assert np.all(almost_equal(Vind, op.apply(U.copy(ind=ind), mu=mu)))


def test_apply2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    for U_ind in valid_inds(U):
        for V_ind in valid_inds(V):
            M = op.apply2(V, U, U_ind=U_ind, V_ind=V_ind, mu=mu)
            assert M.shape == (V.len_ind(V_ind), U.len_ind(U_ind))
            M2 = V.dot(op.apply(U, ind=U_ind, mu=mu), ind=V_ind)
            assert np.allclose(M, M2)


def test_apply2_with_product(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    for U_ind in valid_inds(U):
        for V_ind in valid_inds(V):
            M = op.apply2(V, U, U_ind=U_ind, V_ind=V_ind, mu=mu, product=rp)
            assert M.shape == (V.len_ind(V_ind), U.len_ind(U_ind))
            M2 = V.dot(rp.apply(op.apply(U, ind=U_ind, mu=mu)), ind=V_ind)
            assert np.allclose(M, M2)


def test_pairwise_apply2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    for U_ind, V_ind in valid_inds_of_same_length(U, V):
        M = op.pairwise_apply2(V, U, U_ind=U_ind, V_ind=V_ind, mu=mu)
        assert M.shape == (V.len_ind(V_ind),)
        M2 = V.pairwise_dot(op.apply(U, ind=U_ind, mu=mu), ind=V_ind)
        assert np.allclose(M, M2)


def test_pairwise_apply2_with_product(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    for U_ind, V_ind in valid_inds_of_same_length(U, V):
        M = op.pairwise_apply2(V, U, U_ind=U_ind, V_ind=V_ind, mu=mu, product=rp)
        assert M.shape == (V.len_ind(V_ind),)
        M2 = V.pairwise_dot(rp.apply(op.apply(U, ind=U_ind, mu=mu)), ind=V_ind)
        assert np.allclose(M, M2)


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
        assert np.all(almost_equal(Uind, U, V_ind=ind))
        assert np.all(almost_equal(Uind, op.apply_adjoint(V.copy(ind=ind), mu=mu)))


def test_apply_adjoint_2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    try:
        ATV = op.apply_adjoint(V, mu=mu)
    except NotImplementedError:
        return
    assert np.allclose(V.dot(op.apply(U, mu=mu)), ATV.dot(U))


def test_apply_adjoint_2_with_products(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    try:
        ATV = op.apply_adjoint(V, mu=mu, source_product=sp, range_product=rp)
    except NotImplementedError:
        return
    assert np.allclose(rp.apply2(V, op.apply(U, mu=mu)), sp.apply2(ATV, U))


def test_apply_inverse(operator_with_arrays):
    op, mu, _, V = operator_with_arrays
    for ind in valid_inds(V):
        try:
            U = op.apply_inverse(V, mu=mu, ind=ind)
        except InversionError:
            return
        assert U in op.source
        assert len(U) == V.len_ind(ind)
        VV = op.apply(U, mu=mu)
        assert np.all(almost_equal(VV, V, V_ind=ind, atol=1e-10, rtol=1e-3))


def test_apply_inverse_adjoint(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    for ind in valid_inds(U):
        try:
            V = op.apply_inverse_adjoint(U, mu=mu, ind=ind)
        except InversionError:
            return
        assert V in op.range
        assert len(V) == U.len_ind(ind)
        UU = op.apply_adjoint(V, mu=mu)
        assert np.all(almost_equal(UU, U, V_ind=ind, atol=1e-10, rtol=1e-3))


def test_apply_inverse_adjoint_with_products(operator_with_arrays_and_products):
    op, mu, U, _, sp, rp = operator_with_arrays_and_products
    for ind in valid_inds(U):
        try:
            V = op.apply_inverse_adjoint(U, mu=mu, ind=ind, source_product=sp, range_product=rp)
        except InversionError:
            return
        assert V in op.range
        assert len(V) == U.len_ind(ind)
        UU = op.apply_adjoint(V, mu=mu, source_product=sp, range_product=rp)
        assert np.all(almost_equal(UU, U, V_ind=ind, atol=1e-10, rtol=1e-3))

def test_projected(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    op_UV = op.projected(V, U)
    np.random.seed(4711 + U.dim + len(V))
    coeffs = np.random.random(len(U))
    X = op_UV.apply(NumpyVectorArray(coeffs, copy=False), mu=mu)
    Y = NumpyVectorArray(V.dot(op.apply(U.lincomb(coeffs), mu=mu)).T, copy=False)
    assert np.all(almost_equal(X, Y))


def test_projected_2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    op_U = op.projected(None, U)
    op_V = op.projected(V, None)
    op_U_V = op_U.projected(V, None)
    op_V_U = op_V.projected(None, U)
    op_UV = op.projected(V, U)
    np.random.seed(4711 + U.dim + len(V))
    W = NumpyVectorArray(np.random.random(len(U)), copy=False)
    Y0 = op_UV.apply(W, mu=mu)
    Y1 = op_U_V.apply(W, mu=mu)
    Y2 = op_V_U.apply(W, mu=mu)
    assert np.all(almost_equal(Y0, Y1))
    assert np.all(almost_equal(Y0, Y2))


def test_projected_with_product(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    op_UV = op.projected(V, U, product=rp)
    np.random.seed(4711 + U.dim + len(V))
    coeffs = np.random.random(len(U))
    X = op_UV.apply(NumpyVectorArray(coeffs, copy=False), mu=mu)
    Y = NumpyVectorArray(rp.apply2(op.apply(U.lincomb(coeffs), mu=mu), V), copy=False)
    assert np.all(almost_equal(X, Y))


def test_projected_with_product_2(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    op_U = op.projected(None, U)
    op_V = op.projected(V, None, product=rp)
    op_U_V = op_U.projected(V, None, product=rp)
    op_V_U = op_V.projected(None, U)
    op_UV = op.projected(V, U, product=rp)
    np.random.seed(4711 + U.dim + len(V))
    W = NumpyVectorArray(np.random.random(len(U)), copy=False)
    Y0 = op_UV.apply(W, mu=mu)
    Y1 = op_U_V.apply(W, mu=mu)
    Y2 = op_V_U.apply(W, mu=mu)
    assert np.all(almost_equal(Y0, Y1))
    assert np.all(almost_equal(Y0, Y2))


def test_jacobian(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    if len(U) == 0:
        return
    try:
        j = op.jacobian(U.copy(ind=0), mu=mu)
    except NotImplementedError:
        return
    assert j.linear
    assert op.source == j.source
    assert op.range == j.range


def test_assemble(operator_with_arrays):
    op, mu, _, _ = operator_with_arrays
    aop = op.assemble(mu=mu)
    assert op.source == aop.source
    assert op.range == aop.range


########################################################################################################################


def test_apply_wrong_ind(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    for ind in invalid_inds(U):
        with pytest.raises(Exception):
            op.apply(U, mu=mu, ind=ind)


def test_apply2_wrong_ind(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    for ind in invalid_inds(U):
        with pytest.raises(Exception):
            op.apply2(U, V, mu=mu, ind=ind)
    for ind in invalid_inds(V):
        with pytest.raises(Exception):
            op.apply2(U, V, mu=mu, ind=ind)


def test_apply_adjoint_wrong_ind(operator_with_arrays):
    op, mu, _, V = operator_with_arrays
    for ind in invalid_inds(V):
        with pytest.raises(Exception):
            op.apply_adjoint(V, mu=mu, ind=ind)


def test_apply_inverse_wrong_ind(operator_with_arrays):
    op, mu, _, V = operator_with_arrays
    for ind in invalid_inds(V):
        with pytest.raises(Exception):
            op.apply_inverse(V, mu=mu, ind=ind)

