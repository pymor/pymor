# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.exceptions import InversionError, LinAlgError
from pymor.core.config import config
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import (SelectionOperator, InverseOperator, InverseAdjointOperator, IdentityOperator,
                                           LincombOperator, VectorArrayOperator)
from pymor.operators.numpy import NumpyHankelOperator, NumpyMatrixOperator
from pymor.operators.interface import as_array_max_length
from pymor.parameters.functionals import GenericParameterFunctional, ExpressionParameterFunctional
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import assert_all_almost_equal
from pymortests.fixtures.operator import MonomOperator
from pymortests.pickling import assert_picklable, assert_picklable_without_dumps_function
from pymortests.strategies import valid_inds, valid_inds_of_same_length


def test_selection_op():
    p1 = MonomOperator(1)
    select_rhs_functional = GenericParameterFunctional(
        lambda x: round(x["nrrhs"].item()),
        {"nrrhs": 1}
    )
    s1 = SelectionOperator(
        operators=[p1],
        boundaries=[],
        parameter_functional=select_rhs_functional,
        name="foo"
    )
    x = np.linspace(-1., 1., num=3)
    vx = p1.source.make_array(x[:, np.newaxis])
    assert np.allclose(p1.apply(vx).to_numpy(),
                       s1.apply(vx, mu=s1.parameters.parse(0)).to_numpy())

    s2 = SelectionOperator(
        operators=[p1, p1, p1, p1],
        boundaries=[-3, 3, 7],
        parameter_functional=select_rhs_functional,
        name="Bar"
    )

    assert s2._get_operator_number(s2.parameters.parse({"nrrhs": -4})) == 0
    assert s2._get_operator_number(s2.parameters.parse({"nrrhs": -3})) == 0
    assert s2._get_operator_number(s2.parameters.parse({"nrrhs": -2})) == 1
    assert s2._get_operator_number(s2.parameters.parse({"nrrhs": 3})) == 1
    assert s2._get_operator_number(s2.parameters.parse({"nrrhs": 4})) == 2
    assert s2._get_operator_number(s2.parameters.parse({"nrrhs": 7})) == 2
    assert s2._get_operator_number(s2.parameters.parse({"nrrhs": 9})) == 3


def test_lincomb_op():
    p1 = MonomOperator(1)
    p2 = MonomOperator(2)
    p12 = p1 + p2
    p0 = p1 - p1
    x = np.linspace(-1., 1., num=3)
    vx = p1.source.make_array((x[:, np.newaxis]))
    one = p1.source.make_array([1])
    assert np.allclose(p0.apply(vx).to_numpy(), [0.])
    assert np.allclose(p12.apply(vx).to_numpy(), (x * x + x)[:, np.newaxis])
    assert np.allclose((p1 * 2.).apply(vx).to_numpy(), (x * 2.)[:, np.newaxis])
    with pytest.raises(AssertionError):
        p2.jacobian(vx)
    for i in range(len(vx)):
        assert almost_equal(p2.jacobian(vx[i]).apply(one), p1.apply(vx[i]) * 2.)
        assert almost_equal(p0.jacobian(vx[i]).apply(one), vx[i] * 0.)
    with pytest.raises(TypeError):
        p2.as_vector()
    p1.as_vector()
    assert almost_equal(p1.as_vector(), p1.apply(p1.source.make_array([1.])))

    basis = p1.source.make_array([1.])
    for p in (p1, p2, p12):
        projected = project(p, basis, basis)
        pa = projected.apply(vx)
        assert almost_equal(pa, p.apply(vx)).all()


def test_lincomb_op_with_zero_coefficients():
    p1 = MonomOperator(1)
    p2 = MonomOperator(2)
    p10 = p1 + 0 * p2
    p0 = 0 * p1 + 0 * p1
    x = np.linspace(-1., 1., num=3)
    vx = p1.source.make_array((x[:, np.newaxis]))

    pc1 = NumpyMatrixOperator(np.eye(p1.source.dim))
    pc2 = NumpyMatrixOperator(2*np.eye(p1.source.dim))
    pc10 = pc1 + 0 * pc2
    pc0 = 0 * pc1 + 0 * pc2

    assert np.allclose(p0.apply(vx).to_numpy(), [0.])
    assert len(p0.apply(vx)) == len(vx)
    assert almost_equal(p10.apply(vx), p1.apply(vx)).all()

    assert np.allclose(p0.apply2(vx, vx), [0.])
    assert len(p0.apply2(vx, vx)) == len(vx)
    assert np.allclose(p10.apply2(vx, vx), p1.apply2(vx, vx))

    assert np.allclose(p0.pairwise_apply2(vx, vx), [0.])
    assert len(p0.pairwise_apply2(vx, vx)) == len(vx)
    assert np.allclose(p10.pairwise_apply2(vx, vx), p1.pairwise_apply2(vx, vx))

    assert np.allclose(pc0.apply_adjoint(vx).to_numpy(), [0.])
    assert len(pc0.apply_adjoint(vx)) == len(vx)
    assert almost_equal(pc10.apply_adjoint(vx), pc1.apply_adjoint(vx)).all()


def test_lincomb_adjoint():
    op = LincombOperator([NumpyMatrixOperator(np.eye(10)), NumpyMatrixOperator(np.eye(10))],
                         [1+3j, ExpressionParameterFunctional('c[0] + 3', {'c': 1})])
    mu = op.parameters.parse(1j)
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
    idid_ = ExpressionParameterFunctional('2', {}) * identity
    assert almost_equal(ones * 2, idid.apply(ones))
    assert almost_equal(ones * 2, idid.apply_adjoint(ones))
    assert almost_equal(ones * 0.5, idid.apply_inverse(ones))
    assert almost_equal(ones * 0.5, idid.apply_inverse_adjoint(ones))
    assert almost_equal(ones * 0.5, idid_.apply_inverse(ones))
    assert almost_equal(ones * 0.5, idid_.apply_inverse_adjoint(ones))


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
            M2 = V[V_ind].inner(op.apply(U[U_ind], mu=mu))
            assert np.allclose(M, M2)


def test_pairwise_apply2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    for U_ind, V_ind in valid_inds_of_same_length(U, V):
        M = op.pairwise_apply2(V[V_ind], U[U_ind], mu=mu)
        assert M.shape == (V.len_ind(V_ind),)
        M2 = V[V_ind].pairwise_inner(op.apply(U[U_ind], mu=mu))
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
    assert np.allclose(V.inner(op.apply(U, mu=mu)), ATV.inner(U))


def test_H(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    if not op.linear:
        return
    try:
        op.H.apply(V, mu=mu)
    except NotImplementedError:
        return
    assert np.allclose(V.inner(op.apply(U, mu=mu)), op.H.apply(V, mu=mu).inner(U))


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
        assert_all_almost_equal(op_U, rop_U, rtol=1e-13)


def test_restricted_jacobian(operator_with_arrays):
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
        jac_U = U[0]
        apply_to = U[0]
        op_U = rop.range.make_array(op.jacobian(jac_U, mu=mu).apply(apply_to).dofs(dofs))
        r_apply_to = rop.source.make_array(apply_to.dofs(source_dofs))
        rop_U = rop.jacobian(r_apply_to, mu=mu).apply(r_apply_to)
        assert len(rop_U) == len(op_U)
        assert len(r_apply_to) == len(apply_to)
        assert_all_almost_equal(op_U, rop_U, rtol=1e-13)


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
    u = np.linalg.lstsq(O.T, v.ravel(), rcond=None)[0]
    assert np.all(almost_equal(U, U.space.from_numpy(u)))


def test_adjoint_vectorarray_op_apply_inverse_lstsq():
    np.random.seed(1234)
    O = np.random.random((3, 5))
    op = VectorArrayOperator(NumpyVectorSpace.make_array(O), adjoint=True)
    V = op.range.random()
    U = op.apply_inverse(V, least_squares=True)
    v = V.to_numpy()
    u = np.linalg.lstsq(O, v.ravel(), rcond=None)[0]
    assert np.all(almost_equal(U, U.space.from_numpy(u)))


def test_as_range_array(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    if (not op.linear
            or not isinstance(op.source, NumpyVectorSpace)
            or op.source.dim > as_array_max_length()):
        return
    array = op.as_range_array(mu)
    assert np.all(almost_equal(array.lincomb(U.to_numpy()), op.apply(U, mu=mu)))


def test_issue_1276():
    from pymor.operators.block import BlockOperator
    from pymor.operators.constructions import IdentityOperator
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    I = IdentityOperator(NumpyVectorSpace(1))
    B = BlockOperator([[I * (-1)]])
    v = B.source.ones()

    B.apply_inverse(v)


@pytest.mark.parametrize('iscomplex', [False, True])
def test_hankel_operator(iscomplex):
    s, p, m = 4, 2, 3
    if iscomplex:
        mp = np.random.rand(s, p, m) + 1j * np.random.rand(s, p, m)
    else:
        mp = np.random.rand(s, p, m)
    op = NumpyHankelOperator(mp)

    U = op.source.random(1)
    V = op.range.random(1)
    np.testing.assert_array_almost_equal(op.apply(U).to_numpy().T, to_matrix(op) @ U.to_numpy().T)
    np.testing.assert_array_almost_equal(op.apply_adjoint(V).to_numpy().T, to_matrix(op).conj().T @ V.to_numpy().T)

    U += 1j * op.source.random(1)
    V += 1j * op.range.random(1)
    np.testing.assert_array_almost_equal(op.apply(U).to_numpy().T, to_matrix(op) @ U.to_numpy().T)
    np.testing.assert_array_almost_equal(op.apply_adjoint(V).to_numpy().T, to_matrix(op).conj().T @ V.to_numpy().T)


if config.HAVE_DUNEGDT:
    from dune.xt.la import IstlSparseMatrix, SparsityPatternDefault
    from pymor.bindings.dunegdt import DuneXTMatrixOperator

    def make_dunegdt_identity(N):
        pattern = SparsityPatternDefault(N)
        for n in range(N):
            pattern.insert(n, n)
        pattern.sort()
        mat = IstlSparseMatrix(N, N, pattern)
        for n in range(N):
            mat.set_entry(n, n, 1.)
        return DuneXTMatrixOperator(mat)

    def test_dunegdt_identiy_apply():
        op = make_dunegdt_identity(4)
        U = op.source.ones(1)
        V = op.apply(U)
        assert (U - V).sup_norm() < 1e-14

    def test_dunegdt_identiy_apply_inverse():
        op = make_dunegdt_identity(4)
        V = op.source.ones(1)
        U = op.apply_inverse(V)
        assert (U - V).sup_norm() < 1e-14
