# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.models.iosys import LTIModel, TransferFunction, SecondOrderModel, LinearDelayModel
from pymor.operators.numpy import NumpyMatrixOperator

type_list = [
    'LTIModel',
    'TransferFunction',
    'SecondOrderModel',
    'LinearDelayModel',
]


def get_model(name):
    if name == 'LTIModel':
        A = np.array([[-1]])
        B = np.array([[1]])
        C = np.array([[1]])
        D = np.array([[1]])
        return LTIModel.from_matrices(A, B, C, D)
    elif name == 'TransferFunction':
        H = lambda s: np.array([[1 / (s + 1)]])
        dH = lambda s: np.array([[-1 / (s + 1)**2]])
        return TransferFunction(1, 1, H, dH)
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


def expected_return_type(m1, m2):
    if type(m1) is TransferFunction or type(m2) is TransferFunction:
        return TransferFunction
    if type(m1) is type(m2):
        return type(m1)
    if type(m1) is LTIModel:
        if type(m2) is SecondOrderModel:
            return LTIModel
        else:  # LinearDelayModel
            return LinearDelayModel
    elif type(m1) is SecondOrderModel:
        if type(m2) is LinearDelayModel:
            return LinearDelayModel
        else:
            return expected_return_type(m2, m1)
    else:
        return expected_return_type(m2, m1)


@pytest.mark.parametrize('m1', type_list)
@pytest.mark.parametrize('m2', type_list)
def test_add(m1, m2):
    m1 = get_model(m1)
    m2 = get_model(m2)
    m = m1 + m2
    assert type(m) is expected_return_type(m1, m2)
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
    assert np.allclose(m.eval_tf(0), m1.eval_tf(0) @ m2.eval_tf(0))
    assert np.allclose(m.eval_dtf(0), m1.eval_dtf(0) @ m2.eval_tf(0) + m1.eval_tf(0) @ m2.eval_dtf(0))
    assert np.allclose(m.eval_tf(1j), m1.eval_tf(1j) @ m2.eval_tf(1j))
    assert np.allclose(m.eval_dtf(1j), m1.eval_dtf(1j) @ m2.eval_tf(1j) + m1.eval_tf(1j) @ m2.eval_dtf(1j))
