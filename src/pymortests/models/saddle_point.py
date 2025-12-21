# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.models.basic import StationaryModel
from pymor.models.saddle_point import SaddlePointModel
from pymor.operators.block import BlockColumnOperator, BlockDiagonalOperator, BlockOperator
from pymor.operators.constructions import (
    AdjointOperator,
    IdentityOperator,
    LincombOperator,
    VectorOperator,
    ZeroOperator,
)
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.tools.frozendict import FrozenDict
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
def test_f_g_VectorOperator(operators, vectors, use_C):
    A, B, C = operators
    f_u, g_p = vectors

    f = VectorOperator(f_u)
    g = VectorOperator(g_p)
    model = SaddlePointModel(A=A, B=B, f=f, g=g, C=(C if use_C else None))

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
def test_f_g_VectorArray(operators, vectors, use_C):
    A, B, C = operators
    f_u, g_p = vectors

    model = SaddlePointModel(A=A, B=B, f=f_u, g=g_p, C=(C if use_C else None))

    # should coerce to BlockColumnOperator([VectorOperator(f), VectorOperator(g)])
    assert isinstance(model.rhs, BlockColumnOperator)
    out = model.rhs.as_range_array()
    np.testing.assert_allclose(out.blocks[0].to_numpy(), f_u.to_numpy())
    np.testing.assert_allclose(out.blocks[1].to_numpy(), g_p.to_numpy())


@pytest.mark.parametrize('use_C', [True, False])
def test_f_VectorOperator_g_none_sets_g_zero(operators, spaces, use_C):
    A, B, C = operators
    U, P = spaces

    f_u = U.ones()
    f = VectorOperator(f_u)
    model = SaddlePointModel(A=A, B=B, f=f, g=None, C=(C if use_C else None))

    assert isinstance(model.rhs, BlockColumnOperator)
    out = model.rhs.as_range_array()
    # f should match, g should be zero in P
    np.testing.assert_allclose(out.blocks[0].to_numpy(), f_u.to_numpy())
    np.testing.assert_allclose(out.blocks[1].to_numpy(), P.zeros().to_numpy())


@pytest.mark.parametrize('use_C', [True, False])
def test_f_VectorArray_g_none_sets_g_zero(operators, spaces, use_C):
    A, B, C = operators
    U, P = spaces

    f_u = U.ones()
    # pass VectorArray directly
    model = SaddlePointModel(A=A, B=B, f=f_u, g=None, C=(C if use_C else None))

    assert isinstance(model.rhs, BlockColumnOperator)
    out = model.rhs.as_range_array()
    np.testing.assert_allclose(out.blocks[0].to_numpy(), f_u.to_numpy())
    np.testing.assert_allclose(out.blocks[1].to_numpy(), P.zeros().to_numpy())


def test_products_none_or_empty(operators, vectors):
    A, B, C = operators
    f_u, g_p = vectors

    m1 = SaddlePointModel(A=A, B=B, f=f_u, g=g_p, C=C, u_product=None, p_product=None)
    assert isinstance(m1.products, FrozenDict)
    assert len(m1.products) == 0


def test_products_with_both_keys_correct_mapping(operators, vectors):
    A, B, C = operators
    U = A.range
    P = B.range
    f_u, g_p = vectors

    Pu = NumpyMatrixOperator(np.eye(U.dim))
    Pp = NumpyMatrixOperator(np.eye(P.dim))

    model = SaddlePointModel(A=A, B=B, C=C, f=f_u, g=g_p, u_product=Pu, p_product=Pp)
    assert model.u_product == Pu
    assert model.p_product == Pp

    # Check that products dict contains the mixed block diagonal
    assert 'mixed' in model.products
    assert isinstance(model.products['mixed'], BlockDiagonalOperator)

    # Check that the block diagonal has the correct blocks
    mixed_product = model.products['mixed']
    assert mixed_product.blocks[0, 0] == Pu
    assert mixed_product.blocks[1, 1] == Pp
    assert model.u_product.source == U
    assert model.u_product.range == U
    assert model.p_product.source == P
    assert model.p_product.range == P


def test_products_single_key_allowed(operators, vectors):
    A, B, C = operators
    U = A.range
    P = B.range
    f_u, g_p = vectors

    Pu = NumpyMatrixOperator(np.eye(U.dim))
    model = SaddlePointModel(A=A, B=B, C=C, f=f_u, g=g_p, u_product=Pu)

    assert model.u_product is Pu
    assert model.p_product is None

    # Check that products dict contains the mixed block diagonal
    assert 'mixed' in model.products
    assert isinstance(model.products['mixed'], BlockDiagonalOperator)

    # Check that the block diagonal has Pu and an IdentityOperator for p
    mixed_product = model.products['mixed']
    assert mixed_product.blocks[0, 0] == Pu
    assert isinstance(mixed_product.blocks[1, 1], IdentityOperator)
    assert mixed_product.blocks[1, 1].source == P


def test_assert_mismatched_A_source_range(spaces, operators, vectors):
    U, P = spaces
    A_bad = NumpyMatrixOperator(np.ones((U.dim, U.dim + 1)))  # not square -> wrong spaces
    _, B_good, C = operators
    f_u, g_p = vectors

    with pytest.raises(AssertionError):
        SaddlePointModel(A=A_bad, B=B_good, C=C, f=f_u, g=g_p)


def test_assert_mismatched_B_source(spaces, operators, vectors):
    _, P = spaces
    A, _, C = operators
    # Make B with wrong source (P -> P instead of U -> P)
    B_bad = NumpyMatrixOperator(np.ones((P.dim, P.dim)))
    f_u, g_p = vectors

    with pytest.raises(AssertionError):
        SaddlePointModel(A=A, B=B_bad, C=C, f=f_u, g=g_p)


def test_assert_C_space(spaces, operators, vectors):
    U, P = spaces
    A, B, _ = operators
    # wrong C: acts on U instead of P
    C_bad = NumpyMatrixOperator(np.eye(U.dim))
    f_u, g_p = vectors

    with pytest.raises(AssertionError):
        SaddlePointModel(A=A, B=B, C=C_bad, f=f_u, g=g_p)


def test_assert_f_VectorOperator_wrong_space(operators, spaces):
    A, B, C = operators
    _, P = spaces
    # f VectorOperator in wrong space (P instead of U)
    f_wrong = VectorOperator(P.ones())
    with pytest.raises(AssertionError):
        SaddlePointModel(A=A, B=B, C=C, f=f_wrong)


def test_assert_f_VectorArray_wrong_space(operators, spaces):
    A, B, C = operators
    _, P = spaces
    # f VectorArray in wrong space (P instead of U)
    f_wrong = P.ones()
    with pytest.raises(AssertionError):
        SaddlePointModel(A=A, B=B, C=C, f=f_wrong)


def test_assert_g_VectorOperator_wrong_space(operators, spaces):
    A, B, C = operators
    U, _ = spaces
    f_u = U.ones()
    # g VectorOperator in wrong space (U instead of P)
    g_wrong = VectorOperator(U.ones())
    with pytest.raises(AssertionError):
        SaddlePointModel(A=A, B=B, C=C, f=f_u, g=g_wrong)


def test_assert_g_VectorArray_wrong_space(operators, spaces):
    A, B, C = operators
    U, _ = spaces
    f_u = U.ones()
    # g VectorArray in wrong space (U instead of P)
    g_wrong = U.ones()
    with pytest.raises(AssertionError):
        SaddlePointModel(A=A, B=B, C=C, f=f_u, g=g_wrong)


def test_assert_products_type_check(operators, vectors):
    A, B, C = operators
    f_u, g_p = vectors

    with pytest.raises(AssertionError):
        SaddlePointModel(A=A, B=B, C=C, f=f_u, g=g_p, u_product='not an operator')


def test_parametric_f_g_linear_scalar_source(operators, spaces):
    """Test that parametric operators work when linear with scalar source."""
    A, B, C = operators
    U, P = spaces

    # Parametric f: projects parameter and maps to U-space
    f_base = VectorOperator(U.ones())
    f_parametric = LincombOperator([f_base], [ProjectionParameterFunctional('mu')])

    # Parametric g: projects parameter and maps to P-space
    g_base = VectorOperator(P.ones())
    g_parametric = LincombOperator([g_base], [ProjectionParameterFunctional('mu')])

    # Should work: both are linear and have scalar source
    model = SaddlePointModel(A=A, B=B, C=C, f=f_parametric, g=g_parametric)

    assert isinstance(model.rhs, BlockColumnOperator)
    assert model.rhs.blocks[0, 0] is f_parametric
    assert model.rhs.blocks[1, 0] is g_parametric


def test_assert_f_not_linear(operators, spaces):
    """Test that non-linear f operator raises AssertionError."""
    A, B, C = operators
    U, P = spaces

    # Create a non-linear operator (e.g., mock with linear=False)
    from pymor.operators.interface import Operator

    class NonLinearOperator(Operator):
        linear = False
        source = NumpyVectorSpace(1)  # scalar source
        range = U

        def apply(self, U, mu=None):
            return self.range.ones()

    f_nonlinear = NonLinearOperator()
    g_p = P.zeros()

    with pytest.raises(AssertionError):
        SaddlePointModel(A=A, B=B, C=C, f=f_nonlinear, g=g_p)


def test_assert_g_not_scalar_source(operators, spaces):
    """Test that g operator with non-scalar source raises AssertionError."""
    A, B, C = operators
    U, P = spaces

    # Create a linear operator with non-scalar source (e.g., from U to P)
    g_nonscalar = NumpyMatrixOperator(np.ones((P.dim, U.dim)))
    assert g_nonscalar.linear
    assert not g_nonscalar.source.is_scalar
    assert g_nonscalar.range == P

    f_u = U.ones()

    with pytest.raises(AssertionError):
        SaddlePointModel(A=A, B=B, C=C, f=f_u, g=g_nonscalar)
