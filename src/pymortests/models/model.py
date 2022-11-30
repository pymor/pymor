# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper, ImplicitMidpointTimeStepper
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.core.pickle import dumps, loads
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.models.iosys import LTIModel
from pymor.models.symplectic import QuadraticHamiltonianModel
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import IdentityOperator
from pymor.tools.random import new_rng
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import runmodule
from pymortests.core.pickling import assert_picklable, assert_picklable_without_dumps_function


def test_pickle(model):
    assert_picklable(model)


def test_pickle_without_dumps_function(picklable_model):
    assert_picklable_without_dumps_function(picklable_model)


def test_pickle_by_solving(model):
    m = model
    m2 = loads(dumps(m))
    m.disable_caching()
    m2.disable_caching()
    with new_rng(234):
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


@pytest.mark.parametrize('block_phase_space', (False, True))
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


@pytest.mark.parametrize('sampling_time', (0, 1))
def test_lti_solve(sampling_time):
    if sampling_time == 0:
        time_stepper = ExplicitEulerTimeStepper(4)
    else:
        time_stepper = None
    lti = LTIModel.from_matrices(np.diag([-0.1, -0.2]), np.ones((2, 1)), np.ones((1, 2)),
                                 sampling_time=sampling_time,
                                 T=4, initial_data=np.array([1, 2]), time_stepper=time_stepper)
    f = '[sin(t[0])]'
    X = lti.solve(input=f)
    assert X.dim == 2
    assert len(X) == 5
    y = lti.output(input=f)
    assert y.shape[1] == 1
    assert y.shape[0] == 5


if __name__ == "__main__":
    runmodule(filename=__file__)
