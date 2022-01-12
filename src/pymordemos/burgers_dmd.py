#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Burgers demo with different applications of Dynamic Mode Decomposition."""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from pymor.algorithms.dmd import dmd
from pymor.analyticalproblems.burgers import burgers_problem
from pymor.discretizers.builtin import discretize_instationary_fv
from pymor.tools.typer import Choices
from typer import Argument, Option, run


def main(
        exp: float = Argument(..., help='Exponent'),

        atol: float = Option(None, help='Absolute tolerance'),
        continuous_time: bool = Option(False, help='Show continous time system eigenvalues.'),
        grid: int = Option(100, help='Use grid with this number of elements.'),
        initial_data: Choices('sin bump') = Option('sin', help='Select the initial data.'),
        modes: int = Option(None, help='Number of DMD modes'),
        nt: int = Option(1000, help='Number of time steps.'),
        periodic: bool = Option(True,
                                help='If not, solve with dirichlet boundary conditions on left and bottom boundary.'),
        rtol: float = Option(None, help='Relative tolerance'),
        v: float = Option(10., help='Speed in x-direction.'),
):
    """Solves a one-dimensional Burgers-type equation.

    See pymor.analyticalproblems.burgers for more details.
    """
    print('Setup Problem ...')
    problem = burgers_problem(v=v, initial_data_type=initial_data.value, circle=periodic)

    print('Discretize ...')
    m, data = discretize_instationary_fv(
        problem,
        diameter=1 / grid,
        num_flux='engquist_osher',
        nt=nt
    )
    print(m.operator.grid)

    mu = exp
    print(f'Solving for exponent = {mu} ... ')
    sys.stdout.flush()
    tic = time.perf_counter()
    U = m.solve(mu)
    print(f'Solving took {time.perf_counter() - tic}s')
    m.visualize(U, title='Solution Trajectory')

    print('Computing DMD ...')
    dt = problem.T / nt if continuous_time else None
    W1, E1 = dmd(X=U, modes=modes, atol=atol, rtol=rtol, type='standard', cont_time_dt=dt, order='phase')
    W2, E2 = dmd(X=U, modes=modes, atol=atol, rtol=rtol, type='exact', cont_time_dt=dt, order='phase')

    print(E1)

    print('Visualize ...')
    # plot DMD modes
    m.visualize((W1.real, W1.imag, W2.real, W2.imag),
                legend=('standard - real part', 'standard - imaginary part',
                        'exact - real part', 'exact - imaginary part'),
                title='DMD Modes')

    # plot DMD eigenvalues (same for standard and exact)
    plt.figure()
    if not continuous_time:
        c = plt.Circle((0., 0.), 1., linestyle='--', color='k', fill=False)
        xl = max(np.max(np.abs(E1.real)), 1.) * 1.1
        yl = max(np.max(np.abs(E1.imag)), 1.) * 1.1
        plt.gca().add_artist(c)
        plt.axis('square')
        plt.xlim([-xl, xl])
        plt.ylim([-yl, yl])
    plt.plot(E1.real, E1.imag, '.')
    plt.title(f'{"Continuous" if continuous_time else "Discrete"}-time DMD Eigenvalues')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()


if __name__ == '__main__':
    run(main)
