# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain

import numpy as np
import pytest

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.algorithms.to_matrix import to_matrix
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


@pytest.mark.parametrize('alpha', [0, 2, -0.5, 1j])
@pytest.mark.parametrize('scalar_first', [False, True])
@pytest.mark.parametrize('param', [False, True])
@pytest.mark.parametrize('sampling_time', sampling_time_list)
@pytest.mark.parametrize('n', name_list + ['PHLTIModel'])
def test_mul_scalar(n, sampling_time, param, scalar_first, alpha):
    m1 = get_model(n, sampling_time, param)
    m = alpha * m1 if scalar_first else m1 * alpha
    assert type(m) is (LTIModel if n == 'PHLTIModel' else type(m1))
    if alpha == 0:
        if isinstance(m, FactorizedTransferFunction):
            assert m.K.source.dim == 0
        elif not isinstance(m, TransferFunction):
            assert m.order == 0
    m1, _, m, mu = get_models_mu(m1, m1, m, param)
    for s in (0, 1j):
        assert np.allclose(m.eval_tf(s, mu=mu), alpha * m1.eval_tf(s, mu=mu))
        assert np.allclose(m.eval_dtf(s, mu=mu), alpha * m1.eval_dtf(s, mu=mu))


def get_lti_model():
    A = np.array([[-1, 1], [0, -2]])
    B = np.array([[1], [2]])
    C = np.array([[3, 4]])
    D = np.array([[1]])
    return LTIModel.from_matrices(A, B, C, D)


@pytest.mark.parametrize('alpha', [4, -0.25, 2j])
def test_mul_scalar_lti_balanced(alpha):
    lti = get_lti_model()
    lti_alpha = alpha * lti
    B, C, D = to_matrix(lti.B), to_matrix(lti.C), to_matrix(lti.D)
    assert np.allclose(to_matrix(lti_alpha.B), np.sqrt(abs(alpha)) * B)
    assert np.allclose(to_matrix(lti_alpha.C), alpha / np.sqrt(abs(alpha)) * C)
    assert np.allclose(to_matrix(lti_alpha.D), alpha * D)


@pytest.mark.parametrize('alpha', [4, -0.25])
def test_mul_scalar_lti_presets(alpha):
    lti = get_lti_model()
    presets = {'poles': lti.poles(), 'c_lr': lti.gramian('c_lr'), 'o_dense': lti.gramian('o_dense'),
               'hsv': lti.hsv(), 'h2_norm': lti.h2_norm(), 'linf_norm': lti.linf_norm()}
    lti_alpha = alpha * lti.with_(presets=presets)
    lti_alpha_ref = alpha * lti
    assert np.allclose(lti_alpha.poles(), lti_alpha_ref.poles())
    for typ in ('c_dense', 'o_dense'):
        assert np.allclose(lti_alpha.gramian(typ), lti_alpha_ref.gramian(typ))
    assert np.allclose(lti_alpha.hsv(), lti_alpha_ref.hsv())
    assert np.isclose(lti_alpha.h2_norm(), lti_alpha_ref.h2_norm())
    assert np.isclose(lti_alpha.linf_norm(), lti_alpha_ref.linf_norm())


@pytest.mark.parametrize('sampling_time', sampling_time_list)
def test_mul_scalar_lti_zero(sampling_time):
    lti = get_lti_model().with_(sampling_time=sampling_time)
    lti_zero = 0 * lti
    assert lti_zero.order == 0
    assert lti_zero.dim_input == lti.dim_input
    assert lti_zero.dim_output == lti.dim_output
    assert lti_zero.sampling_time == sampling_time
    assert lti_zero.gramian('c_lr').dim == 0
    assert lti_zero.gramian('o_dense').shape == (0, 0)
    assert lti_zero.hsv().shape == (0,)
    if sampling_time == 0:
        assert lti_zero.h2_norm() == 0
        assert lti_zero.hinf_norm() == 0
        assert lti_zero.hinf_norm(return_fpeak=True) == (0, 0)
    else:
        assert lti_zero.l2_norm() == 0
    assert lti_zero.linf_norm() == 0
    assert np.allclose((lti + lti_zero).transfer_function.eval_tf(1j), lti.transfer_function.eval_tf(1j))


@pytest.mark.parametrize('alpha', [0, 3, -0.5])
@pytest.mark.parametrize('sampling_time', sampling_time_list)
def test_mul_scalar_lti_output(sampling_time, alpha):
    time_stepper = ImplicitEulerTimeStepper(10) if sampling_time == 0 else None
    lti = get_lti_model()
    lti = lti.with_(sampling_time=sampling_time, T=5, initial_data=lti.solution_space.from_numpy(np.array([1, -2])),
                    time_stepper=time_stepper)
    y = lti.output(input='[sin(t[0])]')
    assert np.allclose((alpha * lti).output(input='[sin(t[0])]'), alpha * y)


@pytest.mark.parametrize('param2', [False, True])
@pytest.mark.parametrize('param1', [False, True])
@pytest.mark.parametrize(('n', 'ph_first'), chain(((n, True) for n in name_list + ['PHLTIModel']),
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
@pytest.mark.parametrize(('n', 'ph_first'), chain(((n, True) for n in name_list + ['PHLTIModel']),
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
@pytest.mark.parametrize(('n', 'ph_first'), chain(((n, True) for n in name_list + ['PHLTIModel']),
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
