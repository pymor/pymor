# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
import pytest
import scipy.linalg as spla

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.exceptions import InversionError, LinAlgError
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import (
    IdentityOperator,
    InverseAdjointOperator,
    InverseOperator,
    LincombOperator,
    QuadraticFunctional,
    QuadraticProductFunctional,
    SelectionOperator,
    VectorArrayOperator,
)
from pymor.operators.interface import as_array_max_length
from pymor.operators.numpy import (
    NumpyMatrixOperator,
)
from pymor.parameters.functionals import ExpressionParameterFunctional, GenericParameterFunctional
from pymor.solvers.least_squares import QRLeastSquaresSolver
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import assert_all_almost_equal
from pymortests.core.pickling import assert_picklable, assert_picklable_without_dumps_function
from pymortests.fixtures.operator import MonomOperator
from pymortests.strategies import valid_inds, valid_inds_of_same_length


@pytest.mark.builtin
def test_selection_op():
    p1 = MonomOperator(1)
    select_rhs_functional = GenericParameterFunctional(
        lambda x: round(x['nrrhs'].item()),
        {'nrrhs': 1}
    )
    s1 = SelectionOperator(
        operators=[p1],
        boundaries=[],
        parameter_functional=select_rhs_functional,
        name='foo'
    )
    x = np.linspace(-1., 1., num=3)
    vx = p1.source.make_array(x[np.newaxis, :])
    assert np.allclose(p1.apply(vx).to_numpy(),
                       s1.apply(vx, mu=s1.parameters.parse(0)).to_numpy())

    s2 = SelectionOperator(
        operators=[p1, p1, p1, p1],
        boundaries=[-3, 3, 7],
        parameter_functional=select_rhs_functional,
        name='Bar'
    )

    assert s2._get_operator_number(s2.parameters.parse({'nrrhs': -4})) == 0
    assert s2._get_operator_number(s2.parameters.parse({'nrrhs': -3})) == 0
    assert s2._get_operator_number(s2.parameters.parse({'nrrhs': -2})) == 1
    assert s2._get_operator_number(s2.parameters.parse({'nrrhs': 3})) == 1
    assert s2._get_operator_number(s2.parameters.parse({'nrrhs': 4})) == 2
    assert s2._get_operator_number(s2.parameters.parse({'nrrhs': 7})) == 2
    assert s2._get_operator_number(s2.parameters.parse({'nrrhs': 9})) == 3


@pytest.mark.builtin
def test_lincomb_op():
    p1 = MonomOperator(1)
    p2 = MonomOperator(2)
    p12 = p1 + p2
    p0 = p1 - p1
    x = np.linspace(-1., 1., num=3)
    vx = p1.source.make_array(x[np.newaxis, :])
    one = p1.source.make_array([1])
    assert np.allclose(p0.apply(vx).to_numpy(), [0.])
    assert np.allclose(p12.apply(vx).to_numpy(), (x * x + x)[np.newaxis, :])
    assert np.allclose((p1 * 2.).apply(vx).to_numpy(), (x * 2.)[np.newaxis, :])
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


@pytest.mark.builtin
def test_lincomb_op_with_zero_coefficients():
    p1 = MonomOperator(1)
    p2 = MonomOperator(2)
    p10 = p1 + 0 * p2
    p0 = 0 * p1 + 0 * p1
    x = np.linspace(-1., 1., num=3)
    vx = p1.source.make_array(x[np.newaxis, :])

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


@pytest.mark.builtin
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


@pytest.mark.builtin
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


@pytest.mark.builtin
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


@pytest.mark.builtin
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


@pytest.mark.builtin
def test_bilin_functional():
    space = NumpyVectorSpace(10)
    scalar = NumpyVectorSpace(1)
    bilin_matrix = np.eye(space.dim)
    bilin_op = NumpyMatrixOperator(bilin_matrix)
    bilin_op = QuadraticFunctional(bilin_op)

    one_vec = np.zeros(space.dim)
    two_vec = np.zeros(space.dim)
    one_vec[0] = 1.
    two_vec[0] = 2.
    one_v = space.from_numpy(one_vec)
    two_v = space.from_numpy(two_vec)
    one_s = scalar.from_numpy([1.])
    four_s = scalar.from_numpy([4.])

    assert bilin_op.source == space
    assert bilin_op.range == scalar
    assert almost_equal(one_s, bilin_op.apply(one_v))
    assert almost_equal(four_s, bilin_op.apply(two_v))


@pytest.mark.builtin
def test_bilin_prod_functional():
    from pymor.operators.constructions import VectorFunctional
    space = NumpyVectorSpace(10)
    scalar = NumpyVectorSpace(1)
    mat = 6. * np.identity(scalar.dim)
    prod = NumpyMatrixOperator(mat)
    lin_vec = space.ones()
    lin_op = VectorFunctional(lin_vec)
    bilin_op = QuadraticProductFunctional(lin_op, lin_op)
    bilin_op_with_prod = QuadraticProductFunctional(lin_op, lin_op, product=prod)

    one_vec = np.zeros(space.dim)
    two_vec = np.zeros(space.dim)
    one_vec[0] = 1.
    two_vec[0] = 2.
    one_v = space.from_numpy(one_vec)
    two_v = space.from_numpy(two_vec)
    one_s = scalar.from_numpy([1.])
    four_s = scalar.from_numpy([4.])
    six_s = scalar.from_numpy([6.])
    twn_four_s = scalar.from_numpy([24.])

    assert bilin_op.source == space
    assert bilin_op_with_prod.source == space
    assert bilin_op.range == scalar
    assert bilin_op_with_prod.range == scalar
    assert almost_equal(one_s, bilin_op.apply(one_v))
    assert almost_equal(four_s, bilin_op.apply(two_v))
    assert almost_equal(six_s, bilin_op_with_prod.apply(one_v))
    assert almost_equal(twn_four_s, bilin_op_with_prod.apply(two_v))


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
        if op.source.dim != op.range.dim and (op.solver is None or not op.solver.least_squares):
            with pytest.raises(AssertionError):
                U = op.apply_inverse(V[ind], mu=mu)
            continue
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
        if op.source.dim != op.range.dim and (op.solver is None or not op.solver.least_squares):
            with pytest.raises(AssertionError):
                V = op.apply_inverse_adjoint(U[ind], mu=mu)
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

    if op.source.dim != op.range.dim:
        return

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


def test_restricted(operator_with_arrays, rng):
    op, mu, U, _, = operator_with_arrays
    if op.range.dim == 0:
        return
    for num in [0, 1, 3, 7]:
        dofs = rng.integers(0, op.range.dim, num)
        try:
            rop, source_dofs = op.restricted(dofs)
        except NotImplementedError:
            return
        op_U = rop.range.make_array(op.apply(U, mu=mu).dofs(dofs))
        rop_U = rop.apply(rop.source.make_array(U.dofs(source_dofs)), mu=mu)
        assert_all_almost_equal(op_U, rop_U, rtol=1e-13)


def test_restricted_jacobian(operator_with_arrays, rng):
    op, mu, U, _, = operator_with_arrays
    if op.range.dim == 0:
        return
    for num in [0, 1, 3, 7]:
        dofs = rng.integers(0, op.range.dim, num)
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
    if op.source.dim != op.range.dim:
        return
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
    if op.source.dim != op.range.dim:
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


@pytest.mark.builtin
def test_vectorarray_op_apply_inverse(rng):
    O = rng.random((5, 5))
    op = VectorArrayOperator(NumpyVectorSpace.make_array(O))
    V = op.range.random()
    U = op.apply_inverse(V)
    v = V.to_numpy()
    u = spla.solve(O, v.ravel())
    assert np.all(almost_equal(U, U.space.from_numpy(u), rtol=1e-10))


@pytest.mark.builtin
def test_vectorarray_op_apply_inverse_lstsq(rng):
    O = rng.random((5, 3))
    op = VectorArrayOperator(NumpyVectorSpace.make_array(O))
    V = op.range.random()
    U = op.apply_inverse(V, solver=QRLeastSquaresSolver())
    v = V.to_numpy()
    u = spla.lstsq(O, v.ravel())[0]
    assert np.all(almost_equal(U, U.space.from_numpy(u)))


@pytest.mark.builtin
def test_adjoint_vectorarray_op_apply_inverse_lstsq(rng):
    O = rng.random((5, 3))
    op = VectorArrayOperator(NumpyVectorSpace.make_array(O), adjoint=True)
    V = op.range.random()
    U = op.apply_inverse(V, solver=QRLeastSquaresSolver('source'))
    v = V.to_numpy()
    u = spla.lstsq(O.T, v.ravel())[0]
    assert np.all(almost_equal(U, U.space.from_numpy(u)))


def test_as_range_array(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    if (not op.linear
            or not isinstance(op.source, NumpyVectorSpace)
            or op.source.dim > as_array_max_length()):
        return
    array = op.as_range_array(mu)
    assert np.all(almost_equal(array.lincomb(U.to_numpy()), op.apply(U, mu=mu)))


@pytest.mark.builtin
def test_issue_1276():
    from pymor.operators.block import BlockOperator
    from pymor.operators.constructions import IdentityOperator
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    I = IdentityOperator(NumpyVectorSpace(1))
    B = BlockOperator([[I * (-1)]])
    v = B.source.ones()

    B.apply_inverse(v)


def test_vector_array_to_selection_operator():
    from pymor.operators.constructions import vector_array_to_selection_operator
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    vs = NumpyVectorSpace(8)
    v = vs.random(10)

    so = vector_array_to_selection_operator(v, initial_time=0., end_time=1.)
    assert_all_almost_equal(so.as_range_array(so.parameters.parse({'t': 0.})), v[0], rtol=1e-13)
    assert_all_almost_equal(so.as_range_array(so.parameters.parse({'t': 1.})), v[-1], rtol=1e-13)
    assert_all_almost_equal(so.as_range_array(so.parameters.parse({'t': 0.5})), v[4], rtol=1e-13)

    time_instances = np.linspace(0., 1., 9)
    so = vector_array_to_selection_operator(v, time_instances=time_instances)
    assert_all_almost_equal(so.as_range_array(so.parameters.parse({'t': -0.1})), v[0], rtol=1e-13)
    assert_all_almost_equal(so.as_range_array(so.parameters.parse({'t': 0.5})), v[4], rtol=1e-13)
    assert_all_almost_equal(so.as_range_array(so.parameters.parse({'t': 1.1})), v[-1], rtol=1e-13)
