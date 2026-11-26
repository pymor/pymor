import numpy as np
import pytest

from pymor.algorithms.timestepping import DiscreteTimeStepper, ImplicitEulerTimeStepper, ImplicitMidpointTimeStepper
from pymor.models.examples import heat_equation_example, heat_equation_non_parametric_example
from pymor.operators.constructions import vector_array_to_selection_operator
from pymor.operators.numpy import NumpyMatrixOperator

pytestmark = pytest.mark.builtin


tol = 1e-10


def setup_non_parametric_example():
    initial_time = 0.
    end_time = 1.
    nt = 12
    fom = heat_equation_non_parametric_example(diameter=1., nt=nt)
    num_values = 19
    initial_data = fom.initial_data.as_range_array()
    operator = fom.operator
    rhs = fom.rhs
    mass = fom.mass
    return fom, initial_time, end_time, nt, num_values, initial_data, operator, rhs, mass


def setup_parametric_example():
    nt = 12
    num_values = 19
    initial_time = 0.
    end_time = 1.
    grid_intervals = 20
    fom = heat_equation_example(grid_intervals=grid_intervals, nt=nt)
    initial_data = fom.initial_data.as_range_array()
    operator = fom.operator
    rhs = fom.rhs
    mass = fom.mass
    mu = fom.parameters.parse({'top': 1.})
    return fom, initial_time, end_time, nt, num_values, initial_data, operator, rhs, mass, mu


def test_va_as_rhs_implicit_euler():
    fom, initial_time, end_time, nt, num_values, initial_data, operator, rhs, mass, mu = setup_parametric_example()
    dt = (end_time - initial_time) / (nt - 1)

    rhs_time_dep = fom.operator.range.empty(reserve=nt)
    for n in range(nt):
        mu_t = mu.at_time(initial_time + n * dt)
        rhs_time_dep.append(2. * rhs.as_vector(mu=mu_t))

    rhs_time_dep_operator = vector_array_to_selection_operator(rhs_time_dep,
                                                               initial_time=initial_time, end_time=end_time)

    for n in range(nt):
        mu_t = mu.at_time(initial_time + n * dt)
        assert (rhs_time_dep_operator.as_vector(mu=mu_t) - 2. * rhs.as_vector(mu=mu_t)).norm() <= tol

    time_stepper = ImplicitEulerTimeStepper(nt - 1)
    U_va_rhs = time_stepper.solve(initial_time, end_time, initial_data, operator, rhs=rhs_time_dep,
                                  mass=mass, mu=mu)
    U = time_stepper.solve(initial_time, end_time, initial_data, operator, rhs=2. * rhs, mass=mass, mu=mu)
    assert np.all((U - U_va_rhs).norm() <= tol)


def test_backward_stepping_implicit_euler():
    fom, initial_time, end_time, nt, num_values, initial_data, operator, rhs, mass = setup_non_parametric_example()
    time_stepper = ImplicitEulerTimeStepper(nt)

    U = time_stepper.solve(initial_time, end_time, initial_data, operator, rhs=rhs, mass=mass)
    U_backwards = time_stepper.solve(end_time, initial_time, initial_data, operator, rhs=rhs, mass=-mass)
    assert np.all((U - U_backwards).norm() <= tol)

    U = time_stepper.solve(initial_time, end_time, initial_data, operator, rhs=rhs, mass=mass, num_values=num_values)
    U_backwards = time_stepper.solve(end_time, initial_time, initial_data, operator, rhs=rhs, mass=-mass,
                                     num_values=num_values)
    assert np.all((U - U_backwards).norm() <= tol)


def test_backward_stepping_implicit_midpoint():
    fom, initial_time, end_time, nt, num_values, initial_data, operator, rhs, mass = setup_non_parametric_example()
    time_stepper = ImplicitMidpointTimeStepper(nt)

    U = time_stepper.solve(initial_time, end_time, initial_data, operator, rhs=rhs, mass=mass)
    U_backwards = time_stepper.solve(end_time, initial_time, initial_data, operator, rhs=rhs, mass=-mass)
    assert np.all((U - U_backwards).norm() <= tol)

    U = time_stepper.solve(initial_time, end_time, initial_data, operator, rhs=rhs, mass=mass, num_values=num_values)
    U_backwards = time_stepper.solve(end_time, initial_time, initial_data, operator, rhs=rhs, mass=-mass,
                                     num_values=num_values)
    assert np.all((U - U_backwards).norm() <= tol)


def test_backward_stepping_discrete():
    fom, initial_time, end_time, nt, num_values, initial_data, operator, rhs, mass = setup_non_parametric_example()
    time_stepper = DiscreteTimeStepper()

    initial_time = 0
    end_time = 10
    dt = 1. / (end_time - initial_time)

    U = time_stepper.solve(initial_time, end_time, initial_data, dt * operator, rhs=dt * rhs, mass=mass)
    U_backwards = time_stepper.solve(end_time, initial_time, initial_data, dt * operator, rhs=dt * rhs, mass=-mass)
    assert np.all((U - U_backwards).norm() <= tol)

    U = time_stepper.solve(initial_time, end_time, initial_data, dt * operator, rhs=dt * rhs, mass=mass,
                           num_values=num_values)
    U_backwards = time_stepper.solve(end_time, initial_time, initial_data, dt * operator, rhs=dt * rhs, mass=-mass,
                                     num_values=num_values)
    assert np.all((U - U_backwards).norm() <= tol)


@pytest.mark.parametrize(('initial_time', 'end_time', 'num_values', 'expected_times'), [
    (0, 4, None, [0, 1, 2, 3, 4]),
    (2, 6, None, [2, 3, 4, 5, 6]),
    (4, 0, None, [4, 3, 2, 1, 0]),
    (0, 4, 3, [0, 2, 4]),
])
def test_discrete_time_stepper_timestamps(initial_time, end_time, num_values, expected_times):
    # x_{k+1} = 0.5 * x_k, so the value returned for time t must be 0.5**|t - initial_time| * x_0
    operator = NumpyMatrixOperator(np.array([[-0.5]]))
    initial_data = operator.source.ones()
    sign = np.sign(end_time - initial_time)
    mass = NumpyMatrixOperator(np.array([[float(sign)]]))  # as in test_backward_stepping_discrete
    time_stepper = DiscreteTimeStepper()

    results = list(time_stepper.iterate(initial_time, end_time, initial_data, operator, mass=mass,
                                        num_values=num_values))
    times = [t for _, t in results]
    assert times == expected_times
    for U, t in results:
        assert np.allclose(U.to_numpy(), (0.5 ** (sign * (t - initial_time))) * initial_data.to_numpy())
