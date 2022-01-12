# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


def test_complex():
    np.random.seed(0)
    I = np.eye(5)
    A = np.random.randn(5, 5)
    B = np.random.randn(5, 5)
    C = np.random.randn(3, 5)

    Iop = NumpyMatrixOperator(I)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cva = NumpyVectorSpace.from_numpy(C)

    # lincombs
    assert not np.iscomplexobj((Iop * 1 + Bop * 1).assemble().matrix)
    assert not np.iscomplexobj((Aop * 1 + Bop * 1).assemble().matrix)
    assert np.iscomplexobj((Aop * (1+0j) + Bop * (1+0j)).assemble().matrix)
    assert np.iscomplexobj((Aop * 1j + Bop * 1).assemble().matrix)
    assert np.iscomplexobj((Bop * 1 + Aop * 1j).assemble().matrix)

    # apply_inverse
    assert not np.iscomplexobj(Aop.apply_inverse(Cva).to_numpy())
    assert np.iscomplexobj((Aop * 1j).apply_inverse(Cva).to_numpy())
    assert np.iscomplexobj((Aop * 1 + Bop * 1j).assemble().apply_inverse(Cva).to_numpy())
    assert np.iscomplexobj(Aop.apply_inverse(Cva * 1j).to_numpy())

    # append
    for rsrv in (0, 10):
        for o_ind in (slice(None), [0]):
            va = NumpyVectorSpace(5).empty(reserve=rsrv)
            va.append(Cva)
            D = np.random.randn(1, 5) + 1j * np.random.randn(1, 5)
            Dva = NumpyVectorSpace.from_numpy(D)

            assert not np.iscomplexobj(va.to_numpy())
            assert np.iscomplexobj(Dva.to_numpy())
            va.append(Dva[o_ind])
            assert np.iscomplexobj(va.to_numpy())

    # scal
    assert not np.iscomplexobj(Cva.to_numpy())
    assert np.iscomplexobj((Cva * 1j).to_numpy())
    assert np.iscomplexobj((Cva * (1 + 0j)).to_numpy())

    # axpy
    assert not np.iscomplexobj(Cva.to_numpy())
    Cva[0].axpy(1, Dva)
    assert np.iscomplexobj(Cva.to_numpy())

    Cva = NumpyVectorSpace.from_numpy(C)
    assert not np.iscomplexobj(Cva.to_numpy())
    Cva[0].axpy(1j, Dva)
    assert np.iscomplexobj(Cva.to_numpy())


def test_real_imag():
    A = np.array([[1 + 2j, 3 + 4j],
                  [5 + 6j, 7 + 8j],
                  [9 + 10j, 11 + 12j]])
    Ava = NumpyVectorSpace.from_numpy(A)
    Bva = Ava.real
    Cva = Ava.imag

    k = 0
    for i in range(3):
        for j in range(2):
            k += 1
            assert Bva.to_numpy()[i, j] == k
            k += 1
            assert Cva.to_numpy()[i, j] == k


def test_scal():
    v = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=float)
    v = NumpyVectorSpace.from_numpy(v)
    v.scal(1j)

    k = 0
    for i in range(2):
        for j in range(3):
            k += 1
            assert v.to_numpy()[i, j] == k * 1j


def test_axpy():
    x = NumpyVectorSpace.from_numpy(np.array([1.]))
    y = NumpyVectorSpace.from_numpy(np.array([1.]))
    y.axpy(1 + 1j, x)
    assert y.to_numpy()[0, 0] == 2 + 1j

    x = NumpyVectorSpace.from_numpy(np.array([1 + 1j]))
    y = NumpyVectorSpace.from_numpy(np.array([1.]))
    y.axpy(-1, x)
    assert y.to_numpy()[0, 0] == -1j


def test_inner():
    x = NumpyVectorSpace.from_numpy(np.array([1 + 1j]))
    y = NumpyVectorSpace.from_numpy(np.array([1 - 1j]))
    z = x.inner(y)
    assert z[0, 0] == -2j


def test_pairwise_inner():
    x = NumpyVectorSpace.from_numpy(np.array([1 + 1j]))
    y = NumpyVectorSpace.from_numpy(np.array([1 - 1j]))
    z = x.pairwise_inner(y)
    assert z == -2j
