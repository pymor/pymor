# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.sparse as sps

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.timestepping import (
    ExplicitEulerTimeStepper,
    ImplicitEulerTimeStepper,
    ImplicitMidpointTimeStepper,
)
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, GenericFunction
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.core.pickle import dumps, loads
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.models.iosys import LTIModel
from pymor.models.symplectic import QuadraticHamiltonianModel
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import IdentityOperator, VectorOperator
from pymor.operators.numpy import NumpyGenericOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import runmodule
from pymortests.core.pickling import assert_picklable, assert_picklable_without_dumps_function

pytestmark = pytest.mark.builtin


def test_pickle(model):
    assert_picklable(model)


def test_pickle_without_dumps_function(picklable_model):
    assert_picklable_without_dumps_function(picklable_model)


def test_pickle_by_solving(model):
    m = model
    m2 = loads(dumps(m))
    m.disable_caching()
    m2.disable_caching()
    mus = m.parameters.space(1, 2).sample_randomly(3)
    for mu in mus:
        assert np.all(almost_equal(m.solve(mu), m2.solve(mu)))


def test_StationaryModel_deaffinize():

    p = thermal_block_problem((2, 2)).with_(
        dirichlet_data=ExpressionFunction('x[0]', 2),
        outputs=[('l2', ConstantFunction(1., 2))]
    )
    m, _ = discretize_stationary_cg(p, diameter=1/10)

    U_aff = m.solve([1, 1, 1, 1])
    m_deaff = m.deaffinize(U_aff)

    mu = m.parameters.parse([0.1, 10, 7, 1])

    U = m.solve(mu)
    U_deaff = m_deaff.solve(mu)
    assert np.all(almost_equal(U, U_deaff + U_aff))
    assert np.allclose(m.output(mu), m_deaff.output(mu))


@pytest.mark.parametrize('block_phase_space', [False, True])
def test_quadratic_hamiltonian_model(block_phase_space):
    """Check QuadraticHamiltonianModel with implicit midpoint rule."""
    if block_phase_space:
        H_op = BlockDiagonalOperator([IdentityOperator(NumpyVectorSpace(1))] * 2)
    else:
        phase_space = NumpyVectorSpace(2)
        H_op = IdentityOperator(phase_space)

    model = QuadraticHamiltonianModel(
        3,
        H_op.source.ones(),
        H_op,
        h=H_op.range.from_numpy(np.array([1, 0])),
        time_stepper=ImplicitMidpointTimeStepper(3),
        name='test_mass_spring_model'
    )

    res = model.solve()

    ham = model.eval_hamiltonian(res)

    # check preservation of the Hamiltonian
    assert np.allclose(ham, ham[0])


@pytest.mark.parametrize('nt', [4, 5])
@pytest.mark.parametrize('T', [4, 5])
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('m', [1, 2])
@pytest.mark.parametrize('sampling_time', [0, 1])
def test_lti_solve(sampling_time, m, p, T, nt):
    if sampling_time == 0:
        time_stepper = ExplicitEulerTimeStepper(nt)
    else:
        time_stepper = None
    lti = LTIModel.from_matrices(np.diag([-0.1, -0.2]), np.ones((2, m)), np.ones((p, 2)),
                                 sampling_time=sampling_time,
                                 T=T, initial_data=np.array([1, 2]), time_stepper=time_stepper)
    if m == 1:
        f = '[sin(t[0])]'
    else:
        f = '[sin(t[0]),sin(2*t[0])]'

    X = lti.solve(input=f)
    assert X.dim == 2
    if sampling_time == 0:
        assert len(X) == nt + 1
    else:
        assert len(X) == T + 1

    y = lti.output(input=f)
    assert y.shape[0] == p
    assert y.shape[1] == len(X)

    y_impulse = lti.impulse_resp()
    assert y_impulse.shape[:2] == y.shape
    assert y_impulse.shape[2] == m

    y_impulse2, X_impulse = lti.impulse_resp(return_solution=True)
    assert np.all(y_impulse == y_impulse2)
    assert isinstance(X_impulse, tuple)
    assert all(Xi.dim == 2 for Xi in X_impulse)
    assert all(len(Xi) == len(X) for Xi in X_impulse)

    y_step = lti.step_resp()
    assert y_step.shape[:2] == y.shape
    assert y_step.shape[2] == m

    y_step2, X_step = lti.step_resp(return_solution=True)
    assert np.all(y_step == y_step2)
    assert isinstance(X_step, tuple)
    assert all(Xi.dim == 2 for Xi in X_step)
    assert all(len(Xi) == len(X) for Xi in X_step)


@pytest.mark.parametrize('sampling_time', [0, 1])
@pytest.mark.parametrize('num_values', [None, 2, 5])
@pytest.mark.parametrize('input_value', [1., 1. + 1j], ids=['real_input', 'complex_input'])
@pytest.mark.parametrize('complex_parts', [
    (), ('A',), ('B',), ('C',), ('D',), ('E',), ('initial_data',), ('A', 'B', 'C', 'D', 'E'),
], ids=['real', 'A', 'B', 'C', 'D', 'E', 'initial_data', 'all_matrices'])
def test_lti_complex_output_not_supported(complex_parts, input_value, num_values, sampling_time):
    matrices = {
        'A': -np.eye(2),
        'B': np.ones((2, 1)),
        'C': np.array([[1., 2.], [3., 4.]]),
        'D': np.ones((2, 1)),
        'E': np.eye(2),
        'initial_data': np.ones(2),
    }
    for part in complex_parts:
        matrices[part] = matrices[part] * (1 + 1j)
    lti = LTIModel.from_matrices(
        **matrices, sampling_time=sampling_time, T=2, num_values=num_values,
        time_stepper=ImplicitEulerTimeStepper(2) if sampling_time == 0 else None,
    )

    X = lti.solve(input=[input_value]).to_numpy()
    assert X.shape[0] == 2
    if complex_parts or np.iscomplexobj(input_value):
        with pytest.raises(NotImplementedError, match='Complex-valued time-domain outputs'):
            lti.output(input=[input_value])
    else:
        np.testing.assert_allclose(
            lti.output(input=[input_value]), matrices['C'] @ X + matrices['D'] * input_value,
        )


@pytest.mark.parametrize('quantity', ['solve', 'output', 'both'])
@pytest.mark.parametrize('sampling_time', [0, 1])
@pytest.mark.parametrize('functional_data', [False, True])
@pytest.mark.parametrize('complex_part', ['input', 'initial_data'])
def test_lti_complex_simulation_data(complex_part, functional_data, sampling_time, quantity, caplog, monkeypatch):
    lti = LTIModel.from_matrices(
        -np.eye(1), np.eye(1), np.eye(1), sampling_time=sampling_time, T=2,
        time_stepper=ImplicitEulerTimeStepper(2) if sampling_time == 0 else None,
    )
    if complex_part == 'initial_data':
        initial_data = lti.solution_space.from_numpy(np.array([1. + 1j]))
        if functional_data:
            lti = lti.with_(initial_data=VectorOperator(initial_data) * ProjectionParameterFunctional('initial'))
            mu = {'initial': 1.}
        else:
            lti = lti.with_(initial_data=initial_data)
            mu = None
        input = [1.]
    else:
        input = (GenericFunction(lambda t: np.array([1j if t[0] > 0 else 1.]), 1, (1,))
                 if functional_data else [1. + 1j])
        mu = None
    monkeypatch.setattr(lti.logger, 'handlers', [caplog.handler])

    if quantity == 'solve':
        X = lti.solve(input=input, mu=mu).to_numpy()
        assert np.iscomplexobj(X)
    else:
        with pytest.raises(NotImplementedError, match='Complex-valued time-domain outputs'):
            lti.compute(solution=quantity == 'both', output=True, input=input, mu=mu)
    assert not any('Complex-valued' in record.message for record in caplog.records)


@pytest.mark.parametrize('matrix_type', [np.asarray, sps.csr_matrix])
@pytest.mark.parametrize('imaginary_part', [0., 1.])
def test_lti_complex_output_dtype_not_supported(matrix_type, imaginary_part):
    lti = LTIModel.from_matrices(
        -np.eye(1), np.empty((1, 0)), matrix_type((1. + imaginary_part * 1j) * np.eye(1)),
        T=1, initial_data=np.ones(1), time_stepper=ImplicitEulerTimeStepper(2),
    )
    np.testing.assert_allclose(lti.solve().to_numpy(), [[1., 2/3, 4/9]])
    with pytest.raises(NotImplementedError, match='Complex-valued time-domain outputs'):
        lti.output()


@pytest.mark.parametrize('response', ['output', 'impulse_resp', 'step_resp'])
def test_lti_complex_response_not_supported(response):
    lti = LTIModel.from_matrices(
        -np.eye(1), np.eye(1), 1j * np.eye(1), T=1, time_stepper=ImplicitEulerTimeStepper(2),
    )
    kwargs = {'input': [1.]} if response == 'output' else {}
    with pytest.raises(NotImplementedError, match='Complex-valued time-domain outputs'):
        getattr(lti, response)(**kwargs)


@pytest.mark.parametrize('estimated_count', [0, 2])
def test_lti_output_becomes_complex(estimated_count, monkeypatch):
    monkeypatch.setattr(ImplicitEulerTimeStepper, 'estimate_time_step_count', lambda self, t0, t1: estimated_count)
    lti = LTIModel.from_matrices(
        -np.eye(1), np.eye(1), np.eye(1), D=np.eye(1), T=2, initial_data=np.ones(1),
        time_stepper=ImplicitEulerTimeStepper(2),
    )
    input = GenericFunction(lambda t: np.array([1j if t[0] > 0 else 1.]), 1, (1,))

    with pytest.raises(NotImplementedError, match='Complex-valued time-domain outputs'):
        lti.output(input=input)


@pytest.mark.parametrize('operator', ['A', 'B', 'C', 'D', 'E'])
@pytest.mark.parametrize('T', [None, 1])
@pytest.mark.parametrize('sampling_time', [0, 1])
def test_lti_rejects_time_dependent_operators(operator, T, sampling_time):
    lti = LTIModel.from_matrices(
        -np.eye(1), np.eye(1), np.eye(1), D=np.eye(1), E=np.eye(1), T=T,
        sampling_time=sampling_time,
        time_stepper=ImplicitEulerTimeStepper(2) if T is not None and sampling_time == 0 else None,
    )
    varying_operator = getattr(lti, operator) * ProjectionParameterFunctional('t')

    with pytest.raises(ValueError, match=f'time-dependent {operator}'):
        lti.with_(**{operator: varying_operator})


@pytest.mark.parametrize('sampling_time', [0, 1])
def test_lti_rejects_time_dependent_initial_data(sampling_time):
    lti = LTIModel.from_matrices(
        -np.eye(1), np.eye(1), np.eye(1), T=1, initial_data=np.ones(1), sampling_time=sampling_time,
        time_stepper=ImplicitEulerTimeStepper(2) if sampling_time == 0 else None,
    )

    with pytest.raises(ValueError, match='time-dependent initial_data'):
        lti.with_(initial_data=lti.initial_data * ProjectionParameterFunctional('t'))


@pytest.mark.parametrize('operator', ['A', 'B', 'C', 'D', 'E', 'initial_data'])
@pytest.mark.parametrize('sampling_time', [0, 1])
@pytest.mark.parametrize('quantity', ['solve', 'output', 'both'])
@pytest.mark.parametrize('time_dependent', [False, True])
def test_lti_operator_parameter_values(operator, sampling_time, quantity, time_dependent):
    lti = LTIModel.from_matrices(
        -np.eye(1), np.eye(1), np.eye(1), D=np.eye(1), E=np.eye(1), T=2,
        initial_data=np.ones(1), sampling_time=sampling_time,
        time_stepper=ImplicitEulerTimeStepper(2) if sampling_time == 0 else None,
    )
    parametric = lti.with_(**{operator: getattr(lti, operator) * ProjectionParameterFunctional('coefficient')})
    mu = {'coefficient': '1 + t[0]' if time_dependent else 1.}
    kwargs = {'solution': quantity != 'output', 'output': quantity != 'solve', 'input': '1 + t[0]'}

    if time_dependent:
        with pytest.raises(ValueError, match=f'time-dependent {operator}'):
            parametric.compute(mu=mu, **kwargs)
    else:
        data = parametric.compute(mu=mu, **kwargs)
        expected = lti.compute(**kwargs)
        for key, value in data.items():
            np.testing.assert_allclose(value.to_numpy() if key == 'solution' else value,
                                       expected[key].to_numpy() if key == 'solution' else expected[key])


@pytest.mark.parametrize('sampling_time', [0, 1])
def test_lti_without_input(sampling_time):
    A = np.diag([-0.25, -0.5])
    C = np.array([[1., 2.], [3., 4.]])
    initial_data = np.array([1., 2.])
    lti = LTIModel.from_matrices(
        A, np.empty((2, 0)), C, sampling_time=sampling_time, T=2, initial_data=initial_data,
        time_stepper=ExplicitEulerTimeStepper(2) if sampling_time == 0 else None,
    )
    factors = 1 + A.diagonal() if sampling_time == 0 else A.diagonal()
    expected = initial_data[:, np.newaxis] * factors[:, np.newaxis] ** np.arange(3)

    np.testing.assert_allclose(lti.solve().to_numpy(), expected)
    np.testing.assert_allclose(lti.output(), C @ expected)


@pytest.mark.parametrize('input_type', ['constant', 'expression', 'function'])
@pytest.mark.parametrize('parametric', [False, True])
def test_lti_feedthrough_uses_fixed_operators(input_type, parametric):
    D = np.array([[2., -1.], [3., 4.]])
    lti = LTIModel.from_matrices(
        -np.eye(1), np.zeros((1, 2)), np.ones((2, 1)), D=D, T=2,
        time_stepper=ExplicitEulerTimeStepper(2),
    )
    mu = None
    if parametric:
        def mapping(U, mu):
            assert 't' not in mu
            return mu['gain'][0] * (D @ U)

        lti = lti.with_(D=NumpyGenericOperator(mapping, dim_source=2, dim_range=2,
                                              linear=True, parameters={'gain': 1}))
        mu = {'gain': 2.}
    times = np.arange(3)
    values = np.vstack((1 + times, 2 - times))
    if input_type == 'constant':
        input = [1., 2.]
        values = np.repeat(np.array(input)[:, np.newaxis], 3, axis=1)
    elif input_type == 'expression':
        input = '[1 + t[0], 2 - t[0]]'
    else:
        input = GenericFunction(lambda t: np.array([1 + t[0], 2 - t[0]]), 1, (2,))

    np.testing.assert_allclose(lti.output(input=input, mu=mu), (2 if parametric else 1) * D @ values)


if __name__ == '__main__':
    runmodule(filename=__file__)
