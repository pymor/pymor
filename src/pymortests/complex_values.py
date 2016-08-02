# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray


def test_complex():
    np.random.seed(0)
    I = np.eye(5)
    A = np.random.randn(5, 5)
    B = np.random.randn(5, 5)
    C = np.random.randn(3, 5)

    Iop = NumpyMatrixOperator(I)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cva = NumpyVectorArray(C)

    # assemble_lincomb
    assert not np.iscomplexobj(Aop.assemble_lincomb((Iop, Bop), (1, 1))._matrix)
    assert not np.iscomplexobj(Aop.assemble_lincomb((Aop, Bop), (1, 1))._matrix)
    assert not np.iscomplexobj(Aop.assemble_lincomb((Aop, Bop), (1 + 0j, 1 + 0j))._matrix)
    assert np.iscomplexobj(Aop.assemble_lincomb((Aop, Bop), (1j, 1))._matrix)
    assert np.iscomplexobj(Aop.assemble_lincomb((Bop, Aop), (1, 1j))._matrix)

    # apply_inverse
    assert not np.iscomplexobj(Aop.apply_inverse(Cva).data)
    assert np.iscomplexobj((Aop * 1j).apply_inverse(Cva).data)
    assert np.iscomplexobj(Aop.assemble_lincomb((Aop, Bop), (1, 1j)).apply_inverse(Cva).data)
    assert np.iscomplexobj(Aop.apply_inverse(Cva * 1j).data)

    # append
    for rsrv in (0, 10):
        for o_ind in (None, [0]):
            va = NumpyVectorArray.make_array(subtype=5, reserve=rsrv)
            va.append(Cva)
            D = np.random.randn(1, 5) + 1j * np.random.randn(1, 5)
            Dva = NumpyVectorArray(D)

            assert not np.iscomplexobj(va.data)
            assert np.iscomplexobj(Dva.data)
            va.append(Dva, o_ind)
            assert np.iscomplexobj(va.data)

    # scal
    assert not np.iscomplexobj(Cva.data)
    assert np.iscomplexobj((Cva * 1j).data)
    assert np.iscomplexobj((Cva * (1 + 0j)).data)

    # axpy
    assert not np.iscomplexobj(Cva.data)
    Cva.axpy(1, Dva, 0)
    assert np.iscomplexobj(Cva.data)

    Cva = NumpyVectorArray(C)
    assert not np.iscomplexobj(Cva.data)
    Cva.axpy(1j, Dva, 0)
    assert np.iscomplexobj(Cva.data)

def test_real_imag():
    A = np.array([[1 + 2j, 3 + 4j],
                  [5 + 6j, 7 + 8j],
                  [9 + 10j, 11 + 12j]])
    Ava = NumpyVectorArray(A)
    Bva = Ava.real
    Cva = Ava.imag

    k = 0
    for i in range(3):
        for j in range(2):
            k += 1
            assert Bva.data[i, j] == k
            k += 1
            assert Cva.data[i, j] == k

def test_scal():
    v = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=float)
    v = NumpyVectorArray(v)
    v.scal(1j)

    k = 0
    for i in range(2):
        for j in range(3):
            k += 1
            assert v.data[i, j] == k * 1j

def test_axpy():
    x = NumpyVectorArray(np.array([1.]))
    y = NumpyVectorArray(np.array([1.]))
    y.axpy(1 + 1j, x)
    assert y.data[0, 0] == 2 + 1j

    x = NumpyVectorArray(np.array([1 + 1j]))
    y = NumpyVectorArray(np.array([1.]))
    y.axpy(-1, x)
    assert y.data[0, 0] == -1j

def test_dot():
    x = NumpyVectorArray(np.array([1 + 1j]))
    y = NumpyVectorArray(np.array([1 - 1j]))
    z = x.dot(y)
    assert z[0, 0] == 2j

def test_pairwise_dot():
    x = NumpyVectorArray(np.array([1 + 1j]))
    y = NumpyVectorArray(np.array([1 - 1j]))
    z = x.pairwise_dot(y)
    assert z == 2j
