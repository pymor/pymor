# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.lincomb import assemble_lincomb
from pymor.operators.constructions import LowRankOperator, LowRankUpdatedOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


def test_low_rank_apply():
    m = 15
    n = 10
    space_m = NumpyVectorSpace(m)
    space_n = NumpyVectorSpace(n)
    rng = np.random.RandomState(0)

    r = 2
    L = space_m.random(r, distribution='normal', random_state=rng)
    C = rng.randn(r, r)
    R = space_n.random(r, distribution='normal', random_state=rng)

    k = 3
    U = space_n.random(k, distribution='normal', random_state=rng)

    LR = LowRankOperator(L, C, R)
    V = LR.apply(U)
    assert np.allclose(V.to_numpy().T, L.to_numpy().T @ C @ (R.to_numpy() @ U.to_numpy().T))

    LR = LowRankOperator(L, C, R, inverted=True)
    V = LR.apply(U)
    assert np.allclose(V.to_numpy().T,
                       L.to_numpy().T @ spla.solve(C, R.to_numpy() @ U.to_numpy().T))


def test_low_rank_apply_adjoint():
    m = 15
    n = 10
    space_m = NumpyVectorSpace(m)
    space_n = NumpyVectorSpace(n)
    rng = np.random.RandomState(0)

    r = 2
    L = space_m.random(r, distribution='normal', random_state=rng)
    C = rng.randn(r, r)
    R = space_n.random(r, distribution='normal', random_state=rng)

    k = 3
    V = space_m.random(k, distribution='normal', random_state=rng)

    LR = LowRankOperator(L, C, R)
    U = LR.apply_adjoint(V)
    assert np.allclose(U.to_numpy().T, R.to_numpy().T @ C.T @ (L.to_numpy() @ V.to_numpy().T))

    LR = LowRankOperator(L, C, R, inverted=True)
    U = LR.apply_adjoint(V)
    assert np.allclose(U.to_numpy().T,
                       R.to_numpy().T @ spla.solve(C.T, L.to_numpy() @ V.to_numpy().T))


def test_low_rank_updated_apply_inverse():
    n = 10
    space = NumpyVectorSpace(n)
    rng = np.random.RandomState(0)
    A = NumpyMatrixOperator(rng.randn(n, n))

    r = 2
    L = space.random(r, distribution='normal', random_state=rng)
    C = rng.randn(r, r)
    R = space.random(r, distribution='normal', random_state=rng)
    LR = LowRankOperator(L, C, R)

    k = 3
    V = space.random(k, distribution='normal', random_state=rng)

    op = LowRankUpdatedOperator(A, LR, 1, 1)
    U = op.apply_inverse(V)
    mat = A.matrix + L.to_numpy().T @ C @ R.to_numpy()
    assert np.allclose(U.to_numpy().T, spla.solve(mat, V.to_numpy().T))

    LR = LowRankOperator(L, C, R, inverted=True)
    op = LowRankUpdatedOperator(A, LR, 1, 1)
    U = op.apply_inverse(V)
    mat = A.matrix + L.to_numpy().T @ spla.solve(C, R.to_numpy())
    assert np.allclose(U.to_numpy().T, spla.solve(mat, V.to_numpy().T))


def test_low_rank_updated_apply_inverse_adjoint():
    n = 10
    space = NumpyVectorSpace(n)
    rng = np.random.RandomState(0)
    A = NumpyMatrixOperator(rng.randn(n, n))

    r = 2
    L = space.random(r, distribution='normal', random_state=rng)
    C = rng.randn(r, r)
    R = space.random(r, distribution='normal', random_state=rng)
    LR = LowRankOperator(L, C, R)

    k = 3
    U = space.random(k, distribution='normal', random_state=rng)

    op = LowRankUpdatedOperator(A, LR, 1, 1)
    V = op.apply_inverse_adjoint(U)
    mat = A.matrix + L.to_numpy().T @ C @ R.to_numpy()
    assert np.allclose(V.to_numpy().T, spla.solve(mat.T, U.to_numpy().T))

    LR = LowRankOperator(L, C, R, inverted=True)
    op = LowRankUpdatedOperator(A, LR, 1, 1)
    V = op.apply_inverse_adjoint(U)
    mat = A.matrix + L.to_numpy().T @ spla.solve(C, R.to_numpy())
    assert np.allclose(V.to_numpy().T, spla.solve(mat.T, U.to_numpy().T))


def test_low_rank_assemble():
    n = 10
    space = NumpyVectorSpace(n)
    rng = np.random.RandomState(0)

    r1 = 2
    L1 = space.random(r1, distribution='normal', random_state=rng)
    C1 = rng.randn(r1, r1)
    R1 = space.random(r1, distribution='normal', random_state=rng)

    r2 = 3
    L2 = space.random(r2, distribution='normal', random_state=rng)
    C2 = rng.randn(r2, r2)
    R2 = space.random(r2, distribution='normal', random_state=rng)

    LR1 = LowRankOperator(L1, C1, R1)
    LR2 = LowRankOperator(L2, C2, R2)
    op = assemble_lincomb([LR1, LR2], [1, 1])
    assert isinstance(op, LowRankOperator)
    assert len(op.left) == r1 + r2
    assert not op.inverted

    op = (LR1 + (LR1 + LR2) + LR2).assemble()
    assert isinstance(op, LowRankOperator)

    LR1 = LowRankOperator(L1, C1, R1, inverted=True)
    LR2 = LowRankOperator(L2, C2, R2, inverted=True)
    op = assemble_lincomb([LR1, LR2], [1, 1])
    assert isinstance(op, LowRankOperator)
    assert len(op.left) == r1 + r2
    assert op.inverted

    LR1 = LowRankOperator(L1, C1, R1, inverted=True)
    LR2 = LowRankOperator(L2, C2, R2)
    op = assemble_lincomb([LR1, LR2], [1, 1])
    assert op is None


def test_low_rank_updated_assemble():
    n = 10
    space = NumpyVectorSpace(n)
    rng = np.random.RandomState(0)
    A = NumpyMatrixOperator(rng.randn(n, n))

    r = 2
    L = space.random(r, distribution='normal', random_state=rng)
    C = rng.randn(r, r)
    R = space.random(r, distribution='normal', random_state=rng)
    LR = LowRankOperator(L, C, R)

    op = (A + LR).assemble()
    assert isinstance(op, LowRankUpdatedOperator)

    op = (A + LR + LR).assemble()
    assert isinstance(op, LowRankUpdatedOperator)

    op = (A + (A + LR).assemble() + LR).assemble()
    assert isinstance(op, LowRankUpdatedOperator)


def test_low_rank_updated_assemble_apply():
    n = 10
    space = NumpyVectorSpace(n)
    rng = np.random.RandomState(0)
    A = NumpyMatrixOperator(rng.randn(n, n))

    r = 2
    L = space.random(r, distribution='normal', random_state=rng)
    C = rng.randn(r, r)
    R = space.random(r, distribution='normal', random_state=rng)
    LR = LowRankOperator(L, C, R)

    k = 3
    U = space.random(k, distribution='normal', random_state=rng)

    op = (A + (A + LR).assemble() + LR).assemble()
    V = op.apply(U)
    assert np.allclose(V.to_numpy().T,
                       2 * A.matrix @ U.to_numpy().T + 2 * L.to_numpy().T @ C @ (R.to_numpy() @ U.to_numpy().T))
