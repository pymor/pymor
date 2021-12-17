# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.models.iosys import LTIModel, SecondOrderModel, LinearDelayModel
from pymor.models.transfer_function import TransferFunction, FactorizedTransferFunction
from pymor.operators.numpy import NumpyMatrixOperator

type_list = [
    'LTIModel',
    'SecondOrderModel',
    'LinearDelayModel',
    'TransferFunction',
    'FactorizedTransferFunction',
]


def get_model(name):
    if name == 'LTIModel':
        A = np.array([[-1]])
        B = np.array([[1]])
        C = np.array([[1]])
        D = np.array([[1]])
        return LTIModel.from_matrices(A, B, C, D)
    elif name == 'SecondOrderModel':
        M = np.array([[1]])
        E = np.array([[1]])
        K = np.array([[1]])
        B = np.array([[1]])
        C = np.array([[1]])
        D = np.array([[1]])
        return SecondOrderModel.from_matrices(M, E, K, B, C, D=D)
    elif name == 'LinearDelayModel':
        A = NumpyMatrixOperator(np.array([[-1]]))
        Ad = NumpyMatrixOperator(np.array([[-0.1]]))
        B = NumpyMatrixOperator(np.array([[1]]))
        C = NumpyMatrixOperator(np.array([[1]]))
        D = NumpyMatrixOperator(np.array([[1]]))
        tau = 1
        return LinearDelayModel(A, (Ad,), (tau,), B, C, D)
    elif name == 'TransferFunction':
        H = lambda s: np.array([[1 / (s + 1)]])
        dH = lambda s: np.array([[-1 / (s + 1)**2]])
        return TransferFunction(1, 1, H, dH)
    elif name == 'FactorizedTransferFunction':
        K = lambda s: NumpyMatrixOperator(np.array([[s + 1]]))
        B = lambda s: NumpyMatrixOperator(np.array([[1]]))
        C = lambda s: NumpyMatrixOperator(np.array([[1]]))
        D = lambda s: NumpyMatrixOperator(np.array([[1]]))
        dK = lambda s: NumpyMatrixOperator(np.array([[1]]))
        dB = lambda s: NumpyMatrixOperator(np.array([[0]]))
        dC = lambda s: NumpyMatrixOperator(np.array([[0]]))
        dD = lambda s: NumpyMatrixOperator(np.array([[0]]))
        return FactorizedTransferFunction(1, 1, K, B, C, D, dK, dB, dC, dD)


def expected_return_type(m1, m2):
    model_hierarchy = [
        TransferFunction,
        FactorizedTransferFunction,
        LinearDelayModel,
        LTIModel,
        SecondOrderModel,
    ]
    m1_idx = model_hierarchy.index(type(m1))
    m2_idx = model_hierarchy.index(type(m2))
    return model_hierarchy[min(m1_idx, m2_idx)]


def get_tf(m):
    if isinstance(m, TransferFunction):
        return m
    return m.transfer_function


@pytest.mark.parametrize('m1', type_list)
@pytest.mark.parametrize('m2', type_list)
def test_add(m1, m2):
    m1 = get_model(m1)
    m2 = get_model(m2)
    m = m1 + m2
    assert type(m) is expected_return_type(m1, m2)
    m1 = get_tf(m1)
    m2 = get_tf(m2)
    m = get_tf(m)
    assert np.allclose(m.eval_tf(0), m1.eval_tf(0) + m2.eval_tf(0))
    assert np.allclose(m.eval_dtf(0), m1.eval_dtf(0) + m2.eval_dtf(0))
    assert np.allclose(m.eval_tf(1j), m1.eval_tf(1j) + m2.eval_tf(1j))
    assert np.allclose(m.eval_dtf(1j), m1.eval_dtf(1j) + m2.eval_dtf(1j))


@pytest.mark.parametrize('m1', type_list)
@pytest.mark.parametrize('m2', type_list)
def test_sub(m1, m2):
    m1 = get_model(m1)
    m2 = get_model(m2)
    m = m1 - m2
    assert type(m) is expected_return_type(m1, m2)
    m1 = get_tf(m1)
    m2 = get_tf(m2)
    m = get_tf(m)
    assert np.allclose(m.eval_tf(0), m1.eval_tf(0) - m2.eval_tf(0))
    assert np.allclose(m.eval_dtf(0), m1.eval_dtf(0) - m2.eval_dtf(0))
    assert np.allclose(m.eval_tf(1j), m1.eval_tf(1j) - m2.eval_tf(1j))
    assert np.allclose(m.eval_dtf(1j), m1.eval_dtf(1j) - m2.eval_dtf(1j))


@pytest.mark.parametrize('m1', type_list)
@pytest.mark.parametrize('m2', type_list)
def test_mul(m1, m2):
    m1 = get_model(m1)
    m2 = get_model(m2)
    m = m1 * m2
    assert type(m) is expected_return_type(m1, m2)
    m1 = get_tf(m1)
    m2 = get_tf(m2)
    m = get_tf(m)
    assert np.allclose(m.eval_tf(0), m1.eval_tf(0) @ m2.eval_tf(0))
    assert np.allclose(m.eval_dtf(0), m1.eval_dtf(0) @ m2.eval_tf(0) + m1.eval_tf(0) @ m2.eval_dtf(0))
    assert np.allclose(m.eval_tf(1j), m1.eval_tf(1j) @ m2.eval_tf(1j))
    assert np.allclose(m.eval_dtf(1j), m1.eval_dtf(1j) @ m2.eval_tf(1j) + m1.eval_tf(1j) @ m2.eval_dtf(1j))
