import numpy as np

from pymor.algorithms.timestepping import DiscreteTimeStepper, ImplicitEulerTimeStepper, ImplicitMidpointTimeStepper
from pymor.models.examples import heat_equation_non_parametric_example

nt = 10
initial_time = 0.
end_time = 1.
fom = heat_equation_non_parametric_example(diameter=1., nt=nt)
initial_data = fom.initial_data.as_range_array()
operator = fom.operator
rhs = fom.rhs
mass = fom.mass


def test_implicit_euler():
    time_stepper = ImplicitEulerTimeStepper(nt)

    U = time_stepper.solve(initial_time, end_time, initial_data, operator, rhs=rhs, mass=mass)

    U_backwards = time_stepper.solve(end_time, initial_time, initial_data, operator, rhs=rhs, mass=-mass)

    assert np.all((U - U_backwards).norm() <= 1e-12)


def test_implicit_midpoint():
    time_stepper = ImplicitMidpointTimeStepper(nt)

    U = time_stepper.solve(initial_time, end_time, initial_data, operator, rhs=rhs, mass=mass)

    U_backwards = time_stepper.solve(end_time, initial_time, initial_data, operator, rhs=rhs, mass=-mass)

    assert np.all((U - U_backwards).norm() <= 1e-12)


def test_discrete():
    time_stepper = DiscreteTimeStepper()

    initial_time = 0
    end_time = 10
    dt = 1. / (end_time - initial_time)

    U = time_stepper.solve(initial_time, end_time, initial_data, dt * operator, rhs=dt * rhs, mass=mass)

    U_backwards = time_stepper.solve(end_time, initial_time, initial_data, dt * operator, rhs=dt * rhs, mass=-mass)

    assert np.all((U - U_backwards).norm() <= 1e-12)
