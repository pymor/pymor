import numpy as np
import pytest
from pymor.algorithms.timestepping import ImplicitMidpointTimeStepper
from pymor.models.symplectic import QuadraticHamiltonianModel
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import IdentityOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


@pytest.mark.parametrize('block_phase_space', (False, True))
def test_quadratic_hamiltonian_model(block_phase_space):
    """Check QuadraticHamiltonianModel with implicit midpoint rule."""
    if block_phase_space:
        H_op = BlockDiagonalOperator([IdentityOperator(NumpyVectorSpace(1))] * 2)
    else:
        phase_space = NumpyVectorSpace(2)
        H_op = IdentityOperator(phase_space)

    model = QuadraticHamiltonianModel(
        100,
        H_op.source.ones(),
        H_op,
        time_stepper=ImplicitMidpointTimeStepper(100),
        name='test_mass_spring_model'
    )

    res = model.solve()

    ham = model.eval_hamiltonian(res).to_numpy()

    # check preservation of the Hamiltonian
    assert np.allclose(ham, ham[0])
