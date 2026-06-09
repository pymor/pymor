# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from typer import run

from pymor.algorithms.pod import pod
from pymor.models.nonlinear_ph import NonlinearPHModel
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.ph.ph_deim import PGNonlinearPHReductor, PHdeimReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace


def main(
    n: int = 500,
    reduced_order: int = 100,
    deim_order: int = 100,
    final_time: float = 100.0,
    time_steps: int = 1000,
    damping: float = 0.1,
    input_amplitude: float = 0.1,
    plot: bool = True,
):
    """Compare projection-based MOR methods for a nonlinear port-Hamiltonian Toda lattice.

    The Toda lattice is introduced as a benchmark example in Section 3.4 of :cite:`CBG16`.

    The demo first simulates the full-order model and computes POD bases for the state trajectory
    and the nonlinear part of the Hamiltonian gradient. It then compares a structure-preserving
    Petrov--Galerkin ROM with a pH-DEIM ROM.

    Parameters
    ----------
    n
        Number of particles in the Toda lattice. The full-order state dimension is ``2 * n``.
    reduced_order
        Number of POD modes used for the state reduction basis.
    deim_order
        Number of POD modes used for the collateral basis of the pH-DEIM approximation.
    final_time
        Final simulation time.
    time_steps
        Number of implicit Euler time steps.
    damping
        Uniform damping coefficient in the momentum variables.
    input_amplitude
        Amplitude of the constant scalar input.
    plot
        Plot the outputs of the full-order and reduced-order models.
    """
    fom = create_toda_fom(n=n, damping=damping, final_time=final_time, time_steps=time_steps)
    input_value = np.array([input_amplitude])

    print('Solving the full-order model ...')
    fom_result = solve_and_time(fom, input_value)
    solution_fom = fom_result['solution']
    output_fom = fom_result['output']

    print('Computing the state POD basis ...')
    state_basis, _ = pod(solution_fom, modes=reduced_order, product=fom.Q, rtol=0)

    print('Evaluating the nonlinear force snapshots ...')
    nonlinear_snapshots = fom.dh.apply(solution_fom)

    print('Computing the collateral POD basis ...')
    collateral_basis, _ = pod(nonlinear_snapshots, modes=deim_order, product=fom.Q, rtol=0)

    reductors = {
        'Petrov--Galerkin': PGNonlinearPHReductor(fom, V=state_basis, QTE_orthonormal=True),
        'pH-DEIM': PHdeimReductor(fom, V=state_basis, G=collateral_basis, QTE_orthonormal=True),
    }

    results = {}

    for name, reductor in reductors.items():
        print(f'Reducing with {name} ...')
        tic = perf_counter()
        rom = reductor.reduce()
        reduction_time = perf_counter() - tic

        print(f'Solving the {name} ROM ...')
        rom_result = solve_and_time(rom, input_value)
        reconstructed_solution = reductor.reconstruct(rom_result['solution'])

        results[name] = {
            'rom': rom,
            'reduction_time': reduction_time,
            'simulation_time': rom_result['time'],
            'solution': rom_result['solution'],
            'reconstructed_solution': reconstructed_solution,
            'output': rom_result['output'],
            'state_error': relative_trajectory_error(solution_fom, reconstructed_solution, product=fom.Q),
            'output_error': relative_output_error(output_fom, rom_result['output']),
            'speedup': fom_result['time'] / rom_result['time'],
        }

    print_results(
        fom=fom,
        fom_time=fom_result['time'],
        state_basis=state_basis,
        collateral_basis=collateral_basis,
        results=results,
    )

    if plot:
        plot_outputs(final_time=fom.T, output_fom=output_fom, results=results)


def create_toda_fom(n, damping, final_time, time_steps):
    r"""Construct a tethered Toda lattice as a |NonlinearPHModel|.

    The state is ordered as :math:`x=[q^T,p^T]^T`, and the Hamiltonian is

    .. math::
        H(q,p)
        = \frac{1}{2}\sum_{i=1}^n p_i^2 + \sum_{i=1}^{n-1}\exp(q_i-q_{i+1}) + \exp(q_n)-q_1-n.

    The Hamiltonian gradient is decomposed as

    .. math::
        \nabla H(x) = Qx + dh(x),

    where :math:`Q` is the Hessian of :math:`H` at the equilibrium and ``dh`` contains
    the nonlinear remainder.
    """
    identity = np.eye(n)
    zero = np.zeros((n, n))

    interconnection = NumpyMatrixOperator(np.block([[zero, identity], [-identity, zero]]))
    dissipation = NumpyMatrixOperator(np.block([[zero, zero], [zero, damping * identity]]))

    input_matrix = np.zeros((2*n, 1))
    input_matrix[n, 0] = 1
    port = NumpyMatrixOperator(input_matrix)

    nonlinear_force = TodaNonlinearForceOperator(n)
    energy_product = NumpyMatrixOperator(np.block([[nonlinear_force.position_hessian, zero], [zero, identity]]))

    return NonlinearPHModel(
        J=interconnection,
        R=dissipation,
        G=port,
        dh=nonlinear_force,
        Q=energy_product,
        T=final_time,
        nt=time_steps
    )


class TodaNonlinearForceOperator(Operator):
    r"""Nonlinear remainder of the Hamiltonian gradient for the Toda lattice.

    For :math:`x=[q^T,p^T]^T`, this operator evaluates

    .. math::
        dh(x) = \nabla H(x) - Qx.

    Only the position block is nonlinear. The momentum block of the result is identically zero.
    """

    linear = False

    def __init__(self, n):
        self.n = n
        self.source = NumpyVectorSpace(2*n)
        self.range = NumpyVectorSpace(2*n)

        diagonal = 2 * np.ones(n)
        diagonal[0] = 1
        self.position_hessian = np.diag(diagonal) - np.diag(np.ones(n - 1), k=1) - np.diag(np.ones(n - 1), k=-1)

    def apply(self, U, mu=None):
        states = U.to_numpy()
        result = np.zeros_like(states)
        result[:self.n] = self._position_force(states[:self.n])
        return self.range.from_numpy(result)

    def jacobian(self, U, mu=None):
        assert len(U) == 1

        positions = U.to_numpy()[:self.n, 0]
        interaction_exponentials = np.exp(positions[:-1] - positions[1:])

        diagonal = np.empty(self.n)
        diagonal[0] = interaction_exponentials[0]
        diagonal[1:-1] = interaction_exponentials[:-1] + interaction_exponentials[1:]
        diagonal[-1] = interaction_exponentials[-1] + np.exp(positions[-1])

        position_jacobian = np.diag(diagonal) - np.diag(interaction_exponentials, k=1) \
                          - np.diag(interaction_exponentials, k=-1) - self.position_hessian

        jacobian = np.zeros((2*self.n, 2*self.n))
        jacobian[:self.n, :self.n] = position_jacobian

        return NumpyMatrixOperator(jacobian)

    def restricted(self, dofs):
        output_dofs = np.asarray(dofs, dtype=int)
        source_dofs = self._required_source_dofs(output_dofs)
        restricted_operator = RestrictedTodaNonlinearForceOperator(self.n, output_dofs, source_dofs)
        return restricted_operator, source_dofs

    def _position_force(self, positions):
        interaction_exponentials = np.exp(positions[:-1] - positions[1:])

        hamiltonian_gradient = np.empty_like(positions)
        hamiltonian_gradient[0] = interaction_exponentials[0] - 1
        hamiltonian_gradient[1:-1] = interaction_exponentials[1:] - interaction_exponentials[:-1]
        hamiltonian_gradient[-1] = np.exp(positions[-1]) - interaction_exponentials[-1]

        return hamiltonian_gradient - self.position_hessian @ positions

    def _required_source_dofs(self, output_dofs):
        position_dofs = output_dofs[output_dofs < self.n]

        if len(position_dofs) == 0:
            return np.empty(0, dtype=int)

        neighboring_dofs = np.concatenate((position_dofs - 1, position_dofs, position_dofs + 1))
        valid_neighbors = neighboring_dofs[(0 <= neighboring_dofs) & (neighboring_dofs < self.n)]
        return np.unique(valid_neighbors)


class RestrictedTodaNonlinearForceOperator(Operator):
    """Evaluate selected entries of the Toda nonlinear force from their local stencils."""

    linear = False

    def __init__(self, n, output_dofs, source_dofs):
        self.n = n
        self.output_dofs = np.asarray(output_dofs, dtype=int)
        self.source_dofs = np.asarray(source_dofs, dtype=int)
        self.source = NumpyVectorSpace(len(self.source_dofs))
        self.range = NumpyVectorSpace(len(self.output_dofs))

        input_index = {dof: local_index for local_index, dof in enumerate(self.source_dofs)}
        self._position_stencils = []

        for output_index, output_dof in enumerate(self.output_dofs):
            if output_dof >= n:
                continue

            self._position_stencils.append((output_index, output_dof, input_index.get(output_dof - 1),
                                            input_index[output_dof], input_index.get(output_dof + 1))
            )

    def apply(self, U, mu=None):
        restricted_states = U.to_numpy()
        result = np.zeros((len(self.output_dofs), restricted_states.shape[1]))

        for output_index, position_dof, left_index, center_index, right_index in self._position_stencils:
            center = restricted_states[center_index]

            if position_dof == 0:
                right = restricted_states[right_index]
                result[output_index] = np.exp(center - right) - 1 - center + right
            elif position_dof == self.n - 1:
                left = restricted_states[left_index]
                result[output_index] = np.exp(center) - np.exp(left - center) + left - 2 * center
            else:
                left = restricted_states[left_index]
                right = restricted_states[right_index]
                result[output_index] = np.exp(center - right) - np.exp(left - center) + left - 2 * center + right

        return self.range.from_numpy(result)

    def jacobian(self, U, mu=None):
        assert len(U) == 1

        restricted_state = U.to_numpy()[:, 0]
        jacobian = np.zeros((len(self.output_dofs), len(self.source_dofs)))

        for output_index, position_dof, left_index, center_index, right_index in self._position_stencils:
            center = restricted_state[center_index]

            if position_dof == 0:
                right = restricted_state[right_index]
                interaction_exponential = np.exp(center - right)
                jacobian[output_index, center_index] = interaction_exponential - 1
                jacobian[output_index, right_index] = 1 - interaction_exponential
            elif position_dof == self.n - 1:
                left = restricted_state[left_index]
                terminal_exponential = np.exp(center)
                left_exponential = np.exp(left - center)
                jacobian[output_index, left_index] = 1 - left_exponential
                jacobian[output_index, center_index] = terminal_exponential + left_exponential - 2
            else:
                left = restricted_state[left_index]
                right = restricted_state[right_index]
                right_exponential = np.exp(center - right)
                left_exponential = np.exp(left - center)
                jacobian[output_index, left_index] = 1 - left_exponential
                jacobian[output_index, center_index] = right_exponential + left_exponential - 2
                jacobian[output_index, right_index] = 1 - right_exponential

        return NumpyMatrixOperator(jacobian)


def solve_and_time(model, input_value):
    tic = perf_counter()
    data = model.compute(solution=True, output=True, input=input_value)
    elapsed = perf_counter() - tic
    return {'solution': data['solution'], 'output': data['output'], 'time': elapsed}


def relative_trajectory_error(reference, approximation, product):
    error_norms = (reference - approximation).norm(product=product)
    reference_norms = reference.norm(product=product)
    return np.linalg.norm(error_norms) / np.linalg.norm(reference_norms)


def relative_output_error(reference, approximation):
    return np.linalg.norm(reference - approximation) / np.linalg.norm(reference)


def print_results(fom, fom_time, state_basis, collateral_basis, results):
    print('\n======== Nonlinear pH ROM Evaluation ========\n')
    print(f'FOM order:             {fom.order}')
    print(f'State basis size:      {len(state_basis)}')
    print(f'Collateral basis size: {len(collateral_basis)}')
    print(f'FOM simulation time:   {fom_time:.3f}s\n')

    print(
        f"{'Method':<20} | {'ROM order':>9} | {'Offline [s]':>11} | {'Online [s]':>10} | "
        f"{'Speedup':>9} | {'State error':>12} | {'Output error':>12}"
    )
    print('-' * 108)

    for name, result in results.items():
        print(
            f"{name:<20} | {result['rom'].order:9d} | {result['reduction_time']:11.3f} | "
            f"{result['simulation_time']:10.3f} | {result['speedup']:9.2f} | "
            f"{result['state_error']:12.4e} | {result['output_error']:12.4e}"
        )

    print()


def plot_outputs(final_time, output_fom, results):
    times = np.linspace(0, final_time, output_fom.shape[1])

    for output_index in range(output_fom.shape[0]):
        fig, ax = plt.subplots()
        ax.plot(times, output_fom[output_index], label='FOM')

        for name, result in results.items():
            output = result['output']
            rom_times = np.linspace(0, final_time, output.shape[1])
            ax.plot(rom_times, output[output_index], label=name)

        ax.set_xlabel('$t$')
        ax.set_ylabel(f'$y_{output_index + 1}(t)$')
        ax.set_title(f'Output component {output_index + 1}')
        ax.legend()
        ax.grid()
        fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    run(main)
