# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.exceptions import InversionError, LinAlgError
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import (SelectionOperator, InverseOperator, InverseAdjointOperator, IdentityOperator,
                                           LincombOperator, VectorArrayOperator)
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import ParameterType
from pymor.parameters.functionals import GenericParameterFunctional, ExpressionParameterFunctional
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.fixtures.operator import (operator, operator_with_arrays, operator_with_arrays_and_products,
                                          picklable_operator, MonomOperator)
from pymortests.pickling import assert_picklable, assert_picklable_without_dumps_function
from pymortests.vectorarray import valid_inds, valid_inds_of_same_length


def test_selection_op():
    p1 = MonomOperator(1)
    select_rhs_functional = GenericParameterFunctional(
        lambda x: round(float(x["nrrhs"])), 
        ParameterType({"nrrhs": ()})
    )
    s1 = SelectionOperator(
        operators=[p1],
        boundaries=[],
        parameter_functional=select_rhs_functional,
        name="foo"
    )
    x = np.linspace(-1., 1., num=3)
    vx = p1.source.make_array(x[:, np.newaxis])
    assert np.allclose(p1.apply(vx,mu=0).to_numpy(), s1.apply(vx,mu=0).to_numpy())

    s2 = SelectionOperator(
        operators=[p1,p1,p1,p1],
        boundaries=[-3, 3, 7],
        parameter_functional=select_rhs_functional,
        name="Bar"
    )

    assert s2._get_operator_number({"nrrhs": -4}) == 0
    assert s2._get_operator_number({"nrrhs": -3}) == 0
    assert s2._get_operator_number({"nrrhs": -2}) == 1
    assert s2._get_operator_number({"nrrhs": 3}) == 1
    assert s2._get_operator_number({"nrrhs": 4}) == 2
    assert s2._get_operator_number({"nrrhs": 7}) == 2
    assert s2._get_operator_number({"nrrhs": 9}) == 3


def test_lincomb_op():
    p1 = MonomOperator(1)
    p2 = MonomOperator(2)
    p12 = p1 + p2
    p0 = p1 - p1
    x = np.linspace(-1., 1., num=3)
    vx = p1.source.make_array((x[:, np.newaxis]))
    assert np.allclose(p0.apply(vx).to_numpy(), [0.])
    assert np.allclose(p12.apply(vx).to_numpy(), (x * x + x)[:, np.newaxis])
    assert np.allclose((p1 * 2.).apply(vx).to_numpy(), (x * 2.)[:, np.newaxis])
    assert almost_equal(p2.jacobian(vx).apply(vx), p1.apply(vx) * 2.).all()
    assert almost_equal(p0.jacobian(vx).apply(vx), vx * 0.).all()
    with pytest.raises(TypeError):
        p2.as_vector()
    p1.as_vector()
    assert almost_equal(p1.as_vector(), p1.apply(p1.source.make_array([1.])))

    basis = p1.source.make_array([1.])
    for p in (p1, p2, p12):
        projected = project(p, basis, basis)
        pa = projected.apply(vx)
        assert almost_equal(pa, p.apply(vx)).all()


def test_lincomb_adjoint():
    op = LincombOperator([NumpyMatrixOperator(np.eye(10)), NumpyMatrixOperator(np.eye(10))],
                         [1+3j, ExpressionParameterFunctional('c + 3', {'c': ()})])
    mu = op.parse_parameter(1j)
    U = op.range.random()
    V = op.apply_adjoint(U, mu=mu)
    VV = op.H.apply(U, mu=mu)
    assert np.all(almost_equal(V, VV))
    VVV = op.apply(U, mu=mu).conj()
    assert np.all(almost_equal(V, VVV))


def test_identity_lincomb():
    space = NumpyVectorSpace(10)
    identity = IdentityOperator(space)
    ones = space.ones()
    idid = (identity + identity)
    assert almost_equal(ones * 2, idid.apply(ones))
    assert almost_equal(ones * 2, idid.apply_adjoint(ones))
    assert almost_equal(ones * 0.5, idid.apply_inverse(ones))
    assert almost_equal(ones * 0.5, idid.apply_inverse_adjoint(ones))


def test_identity_numpy_lincomb():
    n = 2
    space = NumpyVectorSpace(n)
    identity = IdentityOperator(space)
    numpy_operator = NumpyMatrixOperator(np.ones((n, n)))
    for alpha in [-1, 0, 1]:
        for beta in [-1, 0, 1]:
            idop = alpha * identity + beta * numpy_operator
            mat1 = alpha * np.eye(n) + beta * np.ones((n, n))
            mat2 = to_matrix(idop.assemble(), format='dense')
            assert np.array_equal(mat1, mat2)


def test_block_identity_lincomb():
    space = NumpyVectorSpace(10)
    space2 = BlockVectorSpace([space, space])
    identity = BlockDiagonalOperator([IdentityOperator(space), IdentityOperator(space)])
    identity2 = IdentityOperator(space2)
    ones = space.ones()
    ones2 = space2.make_array([ones, ones])
    idid = identity + identity2
    assert almost_equal(ones2 * 2, idid.apply(ones2))
    assert almost_equal(ones2 * 2, idid.apply_adjoint(ones2))
    assert almost_equal(ones2 * 0.5, idid.apply_inverse(ones2))
    assert almost_equal(ones2 * 0.5, idid.apply_inverse_adjoint(ones2))


def test_pickle(operator):
    assert_picklable(operator)


def test_pickle_without_dumps_function(picklable_operator):
    assert_picklable_without_dumps_function(picklable_operator)


def test_apply(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    V = op.apply(U, mu=mu)
    assert V in op.range
    assert len(V) == len(U)
    for ind in valid_inds(U):
        Vind = op.apply(U[ind], mu=mu)
        assert np.all(almost_equal(Vind, V[ind]))
        assert np.all(almost_equal(Vind, op.apply(U[ind], mu=mu)))


def test_mul(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    V = op.apply(U, mu=mu)
    for a in (0., 1., -1., 0.3):
        assert np.all(almost_equal(V * a, (op * a).apply(U, mu=mu)))


def test_rmul(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    V = op.apply(U, mu=mu)
    for a in (0., 1., -1., 0.3):
        assert np.all(almost_equal(a * V, (op * a).apply(U, mu=mu)))


def test_neg(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    V = op.apply(U, mu=mu)
    assert np.all(almost_equal(-V, (-op).apply(U, mu=mu)))


def test_apply2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    for U_ind in valid_inds(U):
        for V_ind in valid_inds(V):
            M = op.apply2(V[V_ind], U[U_ind], mu=mu)
            assert M.shape == (V.len_ind(V_ind), U.len_ind(U_ind))
            M2 = V[V_ind].dot(op.apply(U[U_ind], mu=mu))
            assert np.allclose(M, M2)


def test_pairwise_apply2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    for U_ind, V_ind in valid_inds_of_same_length(U, V):
        M = op.pairwise_apply2(V[V_ind], U[U_ind], mu=mu)
        assert M.shape == (V.len_ind(V_ind),)
        M2 = V[V_ind].pairwise_dot(op.apply(U[U_ind], mu=mu))
        assert np.allclose(M, M2)


def test_apply_adjoint(operator_with_arrays):
    op, mu, _, V = operator_with_arrays
    if not op.linear:
        return
    try:
        U = op.apply_adjoint(V, mu=mu)
    except (NotImplementedError, LinAlgError):
        return
    assert U in op.source
    assert len(V) == len(U)
    for ind in list(valid_inds(V, 3)) + [[]]:
        Uind = op.apply_adjoint(V[ind], mu=mu)
        assert np.all(almost_equal(Uind, U[ind]))
        assert np.all(almost_equal(Uind, op.apply_adjoint(V[ind], mu=mu)))


def test_apply_adjoint_2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    if not op.linear:
        return
    try:
        ATV = op.apply_adjoint(V, mu=mu)
    except NotImplementedError:
        return
    assert np.allclose(V.dot(op.apply(U, mu=mu)), ATV.dot(U))


def test_H(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    if not op.linear:
        return
    try:
        op.H.apply(V, mu=mu)
    except NotImplementedError:
        return
    assert np.allclose(V.dot(op.apply(U, mu=mu)), op.H.apply(V, mu=mu).dot(U))


def test_apply_inverse(operator_with_arrays):
    op, mu, _, V = operator_with_arrays
    for ind in valid_inds(V):
        try:
            U = op.apply_inverse(V[ind], mu=mu)
        except InversionError:
            return
        assert U in op.source
        assert len(U) == V.len_ind(ind)
        VV = op.apply(U, mu=mu)
        assert np.all(almost_equal(VV, V[ind], atol=1e-10, rtol=1e-3))


def test_apply_inverse_adjoint(operator_with_arrays):
    op, mu, U, _ = operator_with_arrays
    if not op.linear:
        return
    for ind in valid_inds(U):
        if len(U[ind]) == 0:
            continue
        try:
            V = op.apply_inverse_adjoint(U[ind], mu=mu)
        except (InversionError, LinAlgError):
            return
        assert V in op.range
        assert len(V) == U.len_ind(ind)
        UU = op.apply_adjoint(V, mu=mu)
        assert np.all(almost_equal(UU, U[ind], atol=1e-10, rtol=1e-3))



def test_project(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    op_UV = project(op, V, U)
    np.random.seed(4711 + U.dim + len(V))
    coeffs = np.random.random(len(U))
    X = op_UV.apply(op_UV.source.make_array(coeffs), mu=mu)
    Y = op_UV.range.make_array(V.dot(op.apply(U.lincomb(coeffs), mu=mu)).T)
    assert np.all(almost_equal(X, Y))


def test_project_2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    op_U = project(op, None, U)
    op_V = project(op, V, None)
    op_U_V = project(op_U, V, None)
    op_V_U = project(op_V, None, U)
    op_UV = project(op, V, U)
    np.random.seed(4711 + U.dim + len(V))
    W = op_UV.source.make_array(np.random.random(len(U)))
    Y0 = op_UV.apply(W, mu=mu)
    Y1 = op_U_V.apply(W, mu=mu)
    Y2 = op_V_U.apply(W, mu=mu)
    assert np.all(almost_equal(Y0, Y1))
    assert np.all(almost_equal(Y0, Y2))


def test_project_with_product(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    op_UV = project(op, V, U, product=rp)
    np.random.seed(4711 + U.dim + len(V))
    coeffs = np.random.random(len(U))
    X = op_UV.apply(op_UV.source.make_array(coeffs), mu=mu)
    Y = op_UV.range.make_array(rp.apply2(op.apply(U.lincomb(coeffs), mu=mu), V))
    assert np.all(almost_equal(X, Y))


def test_project_with_product_2(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    op_U = project(op, None, U)
    op_V = project(op, V, None, product=rp)
    op_U_V = project(op_U, V, None, product=rp)
    op_V_U = project(op_V, None, U)
    op_UV = project(op, V, U, product=rp)
    np.random.seed(4711 + U.dim + len(V))
    W = op_UV.source.make_array(np.random.random(len(U)))
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
        j = op.jacobian(U[0], mu=mu)
    except NotImplementedError:
        return
    assert j.linear
    assert op.source == j.source
    assert op.range == j.range


def test_assemble(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    aop = op.assemble(mu=mu)
    assert op.source == aop.source
    assert op.range == aop.range
    assert np.all(almost_equal(aop.apply(U), op.apply(U, mu=mu)))

    try:
        ATV = op.apply_adjoint(V, mu=mu)
    except (NotImplementedError, LinAlgError):
        ATV = None
    if ATV is not None:
        assert np.all(almost_equal(aop.apply_adjoint(V), ATV))

    try:
        AIV = op.apply_inverse(V, mu=mu)
    except InversionError:
        AIV = None
    if AIV is not None:
        assert np.all(almost_equal(aop.apply_inverse(V), AIV))

    try:
        AITU = op.apply_inverse_adjoint(U, mu=mu)
    except (InversionError, LinAlgError):
        AITU = None
    if AITU is not None:
        assert np.all(almost_equal(aop.apply_inverse_adjoint(U), AITU))


def test_restricted(operator_with_arrays):
    op, mu, U, _, = operator_with_arrays
    if op.range.dim == 0:
        return
    np.random.seed(4711 + U.dim)
    for num in [0, 1, 3, 7]:
        dofs = np.random.randint(0, op.range.dim, num)
        try:
            rop, source_dofs = op.restricted(dofs)
        except NotImplementedError:
            return
        op_U = rop.range.make_array(op.apply(U, mu=mu).dofs(dofs))
        rop_U = rop.apply(rop.source.make_array(U.dofs(source_dofs)), mu=mu)
        assert np.all(almost_equal(op_U, rop_U))


def test_InverseOperator(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    inv = InverseOperator(op)
    rtol = atol = 1e-12
    try:
        assert np.all(almost_equal(inv.apply(V, mu=mu), op.apply_inverse(V, mu=mu), rtol=rtol, atol=atol))
    except InversionError:
        pass
    try:
        assert np.all(almost_equal(inv.apply_inverse(U, mu=mu), op.apply(U, mu=mu), rtol=rtol, atol=atol))
    except InversionError:
        pass
    if op.linear:
        try:
            assert np.all(almost_equal(inv.apply_adjoint(U, mu=mu), op.apply_inverse_adjoint(U, mu=mu),
                                       rtol=rtol, atol=atol))
        except (InversionError, NotImplementedError):
            pass
        try:
            assert np.all(almost_equal(inv.apply_inverse_adjoint(V, mu=mu), op.apply_adjoint(V, mu=mu),
                                       rtol=rtol, atol=atol))
        except (InversionError, LinAlgError, NotImplementedError):
            pass


def test_InverseAdjointOperator(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    if not op.linear:
        return
    inv = InverseAdjointOperator(op)
    rtol = atol = 1e-12
    try:
        assert np.all(almost_equal(inv.apply(U, mu=mu), op.apply_inverse_adjoint(U, mu=mu),
                                   rtol=rtol, atol=atol))
    except (InversionError, LinAlgError, NotImplementedError):
        pass
    try:
        assert np.all(almost_equal(inv.apply_inverse(V, mu=mu), op.apply_adjoint(V, mu=mu),
                                   rtol=rtol, atol=atol))
    except (InversionError, LinAlgError, NotImplementedError):
        pass
    try:
        assert np.all(almost_equal(inv.apply_adjoint(V, mu=mu), op.apply_inverse(V, mu=mu),
                                   rtol=rtol, atol=atol))
    except (InversionError, LinAlgError, NotImplementedError):
        pass
    try:
        assert np.all(almost_equal(inv.apply_inverse_adjoint(U, mu=mu), op.apply(U, mu=mu),
                                   rtol=rtol, atol=atol))
    except (InversionError, LinAlgError, NotImplementedError):
        pass


def test_vectorarray_op_apply_inverse():
    np.random.seed(1234)
    O = np.random.random((5, 5))
    op = VectorArrayOperator(NumpyVectorSpace.make_array(O))
    V = op.range.random()
    U = op.apply_inverse(V)
    v = V.to_numpy()
    u = np.linalg.solve(O.T, v.ravel())
    assert np.all(almost_equal(U, U.space.from_numpy(u)))


def test_vectorarray_op_apply_inverse_lstsq():
    np.random.seed(1234)
    O = np.random.random((3, 5))
    op = VectorArrayOperator(NumpyVectorSpace.make_array(O))
    V = op.range.random()
    U = op.apply_inverse(V, least_squares=True)
    v = V.to_numpy()
    u = np.linalg.lstsq(O.T, v.ravel())[0]
    assert np.all(almost_equal(U, U.space.from_numpy(u)))


def test_adjoint_vectorarray_op_apply_inverse_lstsq():
    np.random.seed(1234)
    O = np.random.random((3, 5))
    op = VectorArrayOperator(NumpyVectorSpace.make_array(O), adjoint=True)
    V = op.range.random()
    U = op.apply_inverse(V, least_squares=True)
    v = V.to_numpy()
    u = np.linalg.lstsq(O, v.ravel())[0]
    assert np.all(almost_equal(U, U.space.from_numpy(u)))
