# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain

import numpy as np
import pytest

from pymor.models.iosys import LinearDelayModel, LTIModel, PHLTIModel, SecondOrderModel
from pymor.models.transfer_function import FactorizedTransferFunction, TransferFunction
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Mu
from pymor.parameters.functionals import ProjectionParameterFunctional

pytestmark = pytest.mark.builtin


name_list = [
    'LTIModel',
    'SecondOrderModel',
    'LinearDelayModel',
    'TransferFunction',
    'FactorizedTransferFunction',
]

sampling_time_list = [0, 1]


def get_model(name, sampling_time, parametric):
    if name == 'LTIModel':
        if not parametric:
            A = NumpyMatrixOperator(np.array([[-1]]))
        else:
            A = (NumpyMatrixOperator(np.array([[-1]]))
                 + ProjectionParameterFunctional('mu') * NumpyMatrixOperator(np.eye(1)))
        B = NumpyMatrixOperator(np.array([[1]]))
        C = NumpyMatrixOperator(np.array([[1]]))
        D = NumpyMatrixOperator(np.array([[1]]))
        return LTIModel(A, B, C, D, sampling_time=sampling_time)
    elif name == 'PHLTIModel':
        J = NumpyMatrixOperator(np.zeros((1, 1)))
        if not parametric:
            R = NumpyMatrixOperator(np.eye(1))
        else:
            R = (NumpyMatrixOperator(np.array([[1]]))
                 + ProjectionParameterFunctional('mu') * NumpyMatrixOperator(np.eye(1)))
        G = NumpyMatrixOperator(np.eye(1))
        P = NumpyMatrixOperator(-np.eye(1))
        S = NumpyMatrixOperator(np.eye(1))
        N = NumpyMatrixOperator(np.zeros([1, 1]))
        E = NumpyMatrixOperator(np.eye(1))
        Q = NumpyMatrixOperator(np.eye(1))
        return PHLTIModel(J, R, G, P=P, S=S, N=N, E=E, Q=Q)
    elif name == 'SecondOrderModel':
        M = NumpyMatrixOperator(np.array([[1]]))
        E = NumpyMatrixOperator(np.array([[1]]))
        if not parametric:
            K = NumpyMatrixOperator(np.array([[1]]))
        else:
            K = (NumpyMatrixOperator(np.array([[1]]))
                 + ProjectionParameterFunctional('mu') * NumpyMatrixOperator(np.eye(1)))
        B = NumpyMatrixOperator(np.array([[1]]))
        C = NumpyMatrixOperator(np.array([[1]]))
        D = NumpyMatrixOperator(np.array([[1]]))
        return SecondOrderModel(M, E, K, B, C, D=D, sampling_time=sampling_time)
    elif name == 'LinearDelayModel':
        if not parametric:
            A = NumpyMatrixOperator(np.array([[-1]]))
        else:
            A = (NumpyMatrixOperator(np.array([[-1]]))
                 + ProjectionParameterFunctional('mu') * NumpyMatrixOperator(np.eye(1)))
        Ad = NumpyMatrixOperator(np.array([[-0.1]]))
        B = NumpyMatrixOperator(np.array([[1]]))
        C = NumpyMatrixOperator(np.array([[1]]))
        D = NumpyMatrixOperator(np.array([[1]]))
        tau = 1
        return LinearDelayModel(A, (Ad,), (tau,), B, C, D, sampling_time=sampling_time)
    elif name == 'TransferFunction':
        if not parametric:
            H = lambda s: np.array([[1 / (s + 1)]])
            dH = lambda s: np.array([[-1 / (s + 1)**2]])
        else:
            H = lambda s, mu: np.array([[1 / (s + 1 + mu['mu'][0])]])
            dH = lambda s, mu: np.array([[-1 / (s + 1 + mu['mu'][0])**2]])
        return TransferFunction(1, 1, H, dH, sampling_time=sampling_time, parameters={'mu': 1} if parametric else {})
    elif name == 'FactorizedTransferFunction':
        s = ProjectionParameterFunctional('s')
        if not parametric:
            K = s * NumpyMatrixOperator(np.array([[1]])) + NumpyMatrixOperator(np.array([[1]]))
        else:
            K = (s * NumpyMatrixOperator(np.array([[1]])) + NumpyMatrixOperator(np.array([[1]]))
                 + ProjectionParameterFunctional('mu') * NumpyMatrixOperator(np.eye(1)))
        B = NumpyMatrixOperator(np.array([[1]]))
        C = NumpyMatrixOperator(np.array([[1]]))
        D = NumpyMatrixOperator(np.array([[1]]))
        dK = NumpyMatrixOperator(np.array([[1]]))
        dB = NumpyMatrixOperator(np.array([[0]]))
        dC = NumpyMatrixOperator(np.array([[0]]))
        dD = NumpyMatrixOperator(np.array([[0]]))
        return FactorizedTransferFunction(1, 1, K, B, C, D, dK, dB, dC, dD, sampling_time=sampling_time,
                                          parameters={'mu': 1} if parametric else {})


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


def get_models_mu(m1, m2, m, parametric):
    m1 = get_tf(m1)
    m2 = get_tf(m2)
    m = get_tf(m)
    mu = Mu(mu=0) if parametric else None
    return m1, m2, m, mu


def assert_tf_add(m1, m2, m, parametric):
    m1, m2, m, mu = get_models_mu(m1, m2, m, parametric)
    for s in (0, 1j):
        assert np.allclose(m.eval_tf(s, mu=mu),
                           m1.eval_tf(s, mu=mu) + m2.eval_tf(s, mu=mu))
        assert np.allclose(m.eval_dtf(s, mu=mu),
                           m1.eval_dtf(s, mu=mu) + m2.eval_dtf(s, mu=mu))


def assert_tf_sub(m1, m2, m, parametric):
    m1, m2, m, mu = get_models_mu(m1, m2, m, parametric)
    for s in (0, 1j):
        assert np.allclose(m.eval_tf(s, mu=mu),
                           m1.eval_tf(s, mu=mu) - m2.eval_tf(s, mu=mu))
        assert np.allclose(m.eval_dtf(s, mu=mu),
                           m1.eval_dtf(s, mu=mu) - m2.eval_dtf(s, mu=mu))


def assert_tf_mul(m1, m2, m, parametric):
    m1, m2, m, mu = get_models_mu(m1, m2, m, parametric)
    for s in (0, 1j):
        assert np.allclose(m.eval_tf(s, mu=mu),
                           m1.eval_tf(s, mu=mu) @ m2.eval_tf(s, mu=mu))
        assert np.allclose(m.eval_dtf(s, mu=mu),
                           m1.eval_dtf(s, mu=mu) @ m2.eval_tf(s, mu=mu) + m1.eval_tf(s, mu=mu) @ m2.eval_dtf(s, mu=mu))


@pytest.mark.parametrize('param2', [False, True])
@pytest.mark.parametrize('param1', [False, True])
@pytest.mark.parametrize('sampling_time', sampling_time_list)
@pytest.mark.parametrize('n2', name_list)
@pytest.mark.parametrize('n1', name_list)
def test_add(n1, n2, sampling_time, param1, param2):
    m1 = get_model(n1, sampling_time, param1)
    m2 = get_model(n2, sampling_time, param2)
    m = m1 + m2
    assert type(m) is expected_return_type(m1, m2)
    assert_tf_add(m1, m2, m, param1 or param2)


@pytest.mark.parametrize('param2', [False, True])
@pytest.mark.parametrize('param1', [False, True])
@pytest.mark.parametrize('sampling_time', sampling_time_list)
@pytest.mark.parametrize('n2', name_list)
@pytest.mark.parametrize('n1', name_list)
def test_sub(n1, n2, sampling_time, param1, param2):
    m1 = get_model(n1, sampling_time, param1)
    m2 = get_model(n2, sampling_time, param2)
    m = m1 - m2
    assert type(m) is expected_return_type(m1, m2)
    assert_tf_sub(m1, m2, m, param1 or param2)


@pytest.mark.parametrize('param2', [False, True])
@pytest.mark.parametrize('param1', [False, True])
@pytest.mark.parametrize('sampling_time', sampling_time_list)
@pytest.mark.parametrize('n2', name_list)
@pytest.mark.parametrize('n1', name_list)
def test_mul(n1, n2, sampling_time, param1, param2):
    m1 = get_model(n1, sampling_time, param1)
    m2 = get_model(n2, sampling_time, param2)
    m = m1 * m2
    assert type(m) is expected_return_type(m1, m2)
    assert_tf_mul(m1, m2, m, param1 or param2)


@pytest.mark.parametrize('param2', [False, True])
@pytest.mark.parametrize('param1', [False, True])
@pytest.mark.parametrize('n,ph_first', chain(((n, True) for n in name_list + ['PHLTIModel']),
                                             ((n, False) for n in name_list)))
def test_add_ph(n, ph_first, param1, param2):
    if ph_first:
        m1 = get_model('PHLTIModel', 0, param1)
        m2 = get_model(n, 0, param2)
    else:
        m1 = get_model(n, 0, param1)
        m2 = get_model('PHLTIModel', 0, param2)
    m = m1 + m2
    if n == 'PHLTIModel':
        assert type(m) is PHLTIModel
    elif n == 'LTIModel' or n == 'SecondOrderModel':
        assert type(m) is LTIModel
    else:
        assert m.__class__.__name__ == n
    assert_tf_add(m1, m2, m, param1 or param2)


@pytest.mark.parametrize('param2', [False, True])
@pytest.mark.parametrize('param1', [False, True])
@pytest.mark.parametrize('n,ph_first', chain(((n, True) for n in name_list + ['PHLTIModel']),
                                             ((n, False) for n in name_list)))
def test_sub_ph(n, ph_first, param1, param2):
    if ph_first:
        m1 = get_model('PHLTIModel', 0, param1)
        m2 = get_model(n, 0, param2)
    else:
        m1 = get_model(n, 0, param1)
        m2 = get_model('PHLTIModel', 0, param2)
    m = m1 - m2
    if n in ('PHLTIModel', 'LTIModel', 'SecondOrderModel'):
        assert type(m) is LTIModel
    else:
        assert m.__class__.__name__ == n
    assert_tf_sub(m1, m2, m, param1 or param2)


@pytest.mark.parametrize('param2', [False, True])
@pytest.mark.parametrize('param1', [False, True])
@pytest.mark.parametrize('n,ph_first', chain(((n, True) for n in name_list + ['PHLTIModel']),
                                             ((n, False) for n in name_list)))
def test_mul_ph(n, ph_first, param1, param2):
    if ph_first:
        m1 = get_model('PHLTIModel', 0, param1)
        m2 = get_model(n, 0, param2)
    else:
        m1 = get_model(n, 0, param1)
        m2 = get_model('PHLTIModel', 0, param2)
    m = m1 * m2
    if n in ('PHLTIModel', 'LTIModel', 'SecondOrderModel'):
        assert type(m) is LTIModel
    else:
        assert m.__class__.__name__ == n
    assert_tf_mul(m1, m2, m, param1 or param2)
