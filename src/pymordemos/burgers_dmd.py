"""Burgers demo with different Applications of Dynamic Mode Decomposition.

"""
import sys
import math
import time

import matplotlib.pyplot as plt
from typer import Argument, Option, run
from pymor.algorithms.dmd import dmd, rand_dmd

from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import discretize_instationary_fv, RectGrid, TriaGrid
from pymor.tools.typer import Choices


def main(
    exp: float = Argument(..., help='Exponent'),

    grid: int = Option(60, help='Use grid with (2*NI)*NI elements.'),
    grid_type: Choices('rect tria') = Option('rect', help='Type of grid to use.'),
    initial_data: Choices('sin bump') = Option('sin', help='Select the initial data.'),
    lxf_lambda: float = Option(1., help='Parameter lambda in Lax-Friedrichs flux.'),
    periodic: bool = Option(True, help='If not, solve with dirichlet boundary conditions on left and bottom boundary.'),
    nt: int = Option(100, help='Number of time steps.'),
    num_flux: Choices('lax_friedrichs engquist_osher simplified_engquist_osher') = Option(
        'engquist_osher',
        help='Numerical flux to use.'
    ),
    vx: float = Option(1., help='Speed in x-direction.'),
    vy: float = Option(1., help='Speed in y-direction.'),
):
    """Solves a two-dimensional Burgers-type equation.

    See pymor.analyticalproblems.burgers for more details.
    """
    print('Setup Problem ...')
    problem = burgers_problem_2d(vx=vx, vy=vy, initial_data_type=initial_data.value,
                                 parameter_range=(0, 1e42), torus=periodic)

    print('Discretize ...')
    if grid_type == 'rect':
        grid *= 1. / math.sqrt(2)
    m, data = discretize_instationary_fv(
        problem,
        diameter=1. / grid,
        grid_type=RectGrid if grid_type == 'rect' else TriaGrid,
        num_flux=num_flux.value,
        lxf_lambda=lxf_lambda,
        nt=nt
    )
    print(m.operator.grid)

    print(f'The parameters are {m.parameters}')

    mu = exp
    print(f'Solving for exponent = {mu} ... ')
    sys.stdout.flush()
    tic = time.perf_counter()
    U = m.solve(mu)
    print(f'Solving took {time.perf_counter()-tic}s')
    m.visualize(U)

    # ----- Testing DMD -----

    W, E = dmd(A=U, modes='standard')
    W2, E2 = dmd(A=U, target_rank=5, modes='exact')
    W3, E3 = dmd(A=U, modes='exact_scaled')
    W4, E4 = dmd(A=U, modes='standard', order=False)

    print('Visualize ...')
    m.visualize(W, title='DMD Modes - standard')
    plt.plot(E.real, E.imag, 'b.')
    plt.title('DMD Eigenvalues - standard')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

    m.visualize(W4, title='DMD Modes - standard - not ordered')
    plt.plot(E4.real, E4.imag, 'b.')
    plt.title('DMD Eigenvalues - standard - not ordered')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

    m.visualize(W2, title='5 DMD Modes - exact')
    plt.plot(E2.real, E2.imag, 'b.')
    plt.title('DMD Eigenvalues - exact')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

    m.visualize(W3, title='DMD Modes - exact_scaled')
    plt.plot(E3.real, E3.imag, 'b.')
    plt.title('DMD Eigenvalues - exact_scaled')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

    # ----- Testing rDMD -----
    rW, rE = rand_dmd(U, None, 1, 'standard', None, 'normal', oversampling=5, power_iterations=1)
    rW2, rE2 = rand_dmd(U, None, 1, 'standard', None, 'uniform', oversampling=10, power_iterations=2)
    rW3, rE3 = rand_dmd(U, 5, 1, 'exact', None, 'normal', oversampling=2, power_iterations=1)
    rW4, rE4 = rand_dmd(U, None, 1, 'exact', None, 'uniform', oversampling=2, power_iterations=1)

    print('Visualize ...')
    m.visualize(rW, title='randomized DMD Modes - standard - normal distribution')
    plt.plot(rE.real, rE.imag, 'b.')
    plt.plot(E.real, E.imag, 'rx')
    plt.title('rDMD Eigenvalues - standard - normal distribution')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

    m.visualize(rW2, title='randomized DMD Modes - standard - uniform distribution')
    plt.plot(rE2.real, rE2.imag, 'b.')
    plt.plot(E.real, E.imag, 'rx')
    plt.title('rDMD Eigenvalues - standard - uniform distribution')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

    m.visualize(rW3, title='5 randomized DMD Modes - exact - normal distribution')
    plt.plot(rE3.real, rE3.imag, 'b.')
    plt.plot(E2.real, E2.imag, 'rx')
    plt.title('rDMD Eigenvalues - exact - normal distribution')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

    m.visualize(rW4, title='randomized DMD Modes - exact - uniform distribution')
    plt.plot(rE4.real, rE4.imag, 'b.')
    plt.plot(E3.real, E3.imag, 'rx')
    plt.title('rDMD Eigenvalues - standard - uniform distribution')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()


if __name__ == '__main__':
    run(main)
