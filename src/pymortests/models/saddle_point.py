# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.models.basic import StationaryModel
from pymor.models.saddle_point import StationarySaddelPointModel
from pymor.operators.block import BlockColumnOperator, BlockOperator
from pymor.operators.constructions import AdjointOperator, VectorOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

pytestmark = pytest.mark.builtin


@pytest.fixture
def dims():
    return dict(mu=5, pp=3)


@pytest.fixture
def spaces(dims):
    U = NumpyVectorSpace(dims['mu'])
    P = NumpyVectorSpace(dims['pp'])
    return U, P


@pytest.fixture
def operators(spaces):
    U, P = spaces
    A = NumpyMatrixOperator(np.eye(U.dim))
    B = NumpyMatrixOperator(np.ones((P.dim, U.dim)))
    C = NumpyMatrixOperator(np.eye(P.dim))
    return A, B, C


@pytest.fixture
def vectors(spaces):
    U, P = spaces
    f_u = U.ones()
    g_p = P.zeros()
    return f_u, g_p


@pytest.mark.parametrize('use_C', [True, False])
def test_rhs_BlockColumnOperator(operators, vectors, use_C):
    A, B, C = operators
    f_u, g_p = vectors

    rhs = BlockColumnOperator([VectorOperator(f_u), VectorOperator(g_p)], name='rhs')
    model = StationarySaddelPointModel(A=A, B=B, rhs=rhs, C=(C if use_C else None))

    assert isinstance(model, StationaryModel)
    assert isinstance(model.operator, BlockOperator)
    assert model.operator.blocks.shape == (2, 2)

    # check [0,1] is B.H, [1,0] is B
    block01 = model.operator.blocks[0, 1]
    assert isinstance(block01, AdjointOperator)
    assert block01.operator is B
    assert block01.source == B.range
    assert block01.range == B.source
    assert model.operator.blocks[1, 0] is B

    # if C is None, block [1,1] should behave as zero
    if not use_C:
        assert isinstance(model.operator.blocks[1, 1], ZeroOperator)
    else:
        assert model.operator.blocks[1, 1] == C

    # rhs kept as given, 2x1 column
    assert isinstance(model.rhs, BlockColumnOperator)
    assert model.rhs.blocks.shape == (2, 1)

    out = model.rhs.as_range_array()
    assert out.blocks[0].space == A.range
    assert out.blocks[1].space == B.range
    np.testing.assert_allclose(out.blocks[0].to_numpy(), f_u.to_numpy())
    np.testing.assert_allclose(out.blocks[1].to_numpy(), g_p.to_numpy())


@pytest.mark.parametrize('use_C', [True, False])
def test_rhs_BlockVectorArray(operators, vectors, spaces, use_C):
    A, B, C = operators
    f_u, g_p = vectors
    U, P = spaces

    block_vector_space = BlockVectorSpace([U, P])
    rhs_bva = block_vector_space.make_array([f_u, g_p])
    model = StationarySaddelPointModel(A=A, B=B, C=(C if use_C else None), rhs=rhs_bva)

    # should coerce to BlockColumnOperator([VectorOperator(f), VectorOperator(g)])
    assert isinstance(model.rhs, BlockColumnOperator)
    out = model.rhs.as_range_array()
    np.testing.assert_allclose(out.blocks[0].to_numpy(), f_u.to_numpy())
    np.testing.assert_allclose(out.blocks[1].to_numpy(), g_p.to_numpy())


@pytest.mark.parametrize('use_C', [True, False])
def test_rhs_VectorOperator_on_u_sets_g_zero(operators, spaces, use_C):
    A, B, C = operators
    U, P = spaces

    f_u = U.ones()
    rhs_vo = VectorOperator(f_u)
    model = StationarySaddelPointModel(A=A, B=B, rhs=rhs_vo, C=(C if use_C else None))

    assert isinstance(model.rhs, BlockColumnOperator)
    out = model.rhs.as_range_array()
    # f should match, g should be zero in P
    np.testing.assert_allclose(out.blocks[0].to_numpy(), f_u.to_numpy())
    np.testing.assert_allclose(out.blocks[1].to_numpy(), P.zeros().to_numpy())


@pytest.mark.parametrize('use_C', [True, False])
def test_rhs_VectorArray_on_u_sets_g_zero(operators, spaces, use_C):
    A, B, C = operators
    U, P = spaces

    f_u = U.ones()
    # pass VectorArray directly
    model = StationarySaddelPointModel(A=A, B=B, rhs=f_u, C=(C if use_C else None))

    assert isinstance(model.rhs, BlockColumnOperator)
    out = model.rhs.as_range_array()
    np.testing.assert_allclose(out.blocks[0].to_numpy(), f_u.to_numpy())
    np.testing.assert_allclose(out.blocks[1].to_numpy(), P.zeros().to_numpy())


def test_products_none_or_empty(operators, vectors):
    A, B, C = operators
    f_u, g_p = vectors
    rhs = BlockColumnOperator([VectorOperator(f_u), VectorOperator(g_p)])

    m1 = StationarySaddelPointModel(A=A, B=B, rhs=rhs, C=C, products=None)
    assert m1.products is None

    m2 = StationarySaddelPointModel(A=A, B=B, C=C, rhs=rhs, products={})
    assert m2.products is None


def test_products_with_both_keys_correct_mapping(operators, vectors):
    A, B, C = operators
    U = A.range
    P = B.range
    f_u, g_p = vectors
    rhs = BlockColumnOperator([VectorOperator(f_u), VectorOperator(g_p)])

    Pu = NumpyMatrixOperator(np.eye(U.dim))
    Pp = NumpyMatrixOperator(np.eye(P.dim))

    products = {'u': Pu, 'p': Pp}
    model = StationarySaddelPointModel(A=A, B=B, C=C, rhs=rhs, products=products)

    assert model.products['u'] == Pu
    assert model.products['p'] == Pp

    # Also check space-compatibility kept
    assert model.products['u'].source == U
    assert model.products['u'].range == U
    assert model.products['p'].source == P
    assert model.products['p'].range == P


def test_products_single_key_allowed(operators, vectors):
    A, B, C = operators
    U = A.range
    f_u, g_p = vectors
    rhs = BlockColumnOperator([VectorOperator(f_u), VectorOperator(g_p)])

    Pu = NumpyMatrixOperator(np.eye(U.dim))
    model = StationarySaddelPointModel(A=A, B=B, C=C, rhs=rhs, products={'u': Pu})
    assert model.products['u'] is Pu


def test_assert_mismatched_A_source_range(spaces, operators, vectors):
    U, P = spaces
    A_bad = NumpyMatrixOperator(np.ones((U.dim, U.dim + 1)))  # not square -> wrong spaces
    _, B_good, C = operators
    f_u, g_p = vectors
    rhs = BlockColumnOperator([VectorOperator(f_u), VectorOperator(g_p)])

    with pytest.raises(AssertionError):
        StationarySaddelPointModel(A=A_bad, B=B_good, C=C, rhs=rhs)


def test_assert_mismatched_B_source(spaces, operators, vectors):
    _, P = spaces
    A, _, C = operators
    # Make B with wrong source (P -> P instead of U -> P)
    B_bad = NumpyMatrixOperator(np.ones((P.dim, P.dim)))
    f_u, g_p = vectors
    rhs = BlockColumnOperator([VectorOperator(f_u), VectorOperator(g_p)])

    with pytest.raises(AssertionError):
        StationarySaddelPointModel(A=A, B=B_bad, C=C, rhs=rhs)


def test_assert_C_space(spaces, operators, vectors):
    U, P = spaces
    A, B, _ = operators
    # wrong C: acts on U instead of P
    C_bad = NumpyMatrixOperator(np.eye(U.dim))
    f_u, g_p = vectors
    rhs = BlockColumnOperator([VectorOperator(f_u), VectorOperator(g_p)])

    with pytest.raises(AssertionError):
        StationarySaddelPointModel(A=A, B=B, C=C_bad, rhs=rhs)


def test_assert_rhs_blockcolumn_shape(spaces, operators):
    A, B, C = operators
    U, P = spaces
    # bad RHS: 2x2 instead of (2,1)
    rhs_bad = BlockOperator([[VectorOperator(U.ones()), VectorOperator(U.ones())],
                             [VectorOperator(P.zeros()), VectorOperator(P.zeros())]])

    with pytest.raises(AssertionError):
        StationarySaddelPointModel(A=A, B=B, C=C, rhs=rhs_bad)  # not a BlockColumnOperator of shape (2,1)


def test_assert_rhs_vector_spaces(operators, spaces):
    A, B, C = operators
    _, P = spaces
    # RHS VectorOperator in wrong space (P instead of U for the single-vector case)
    rhs_vo_wrong = VectorOperator(P.ones())
    with pytest.raises(AssertionError):
        StationarySaddelPointModel(A=A, B=B, C=C, rhs=rhs_vo_wrong)


def test_assert_products_type_check(operators, vectors):
    A, B, C = operators
    f_u, g_p = vectors
    rhs = BlockColumnOperator([VectorOperator(f_u), VectorOperator(g_p)])

    with pytest.raises(AssertionError):
        StationarySaddelPointModel(A=A, B=B, C=C, rhs=rhs, products={'u': 'not an operator'})
