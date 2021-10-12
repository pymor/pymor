#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Burgers demo with different applications of Dynamic Mode Decomposition."""

import math
import sys
import time

import matplotlib.pyplot as plt
from pymor.algorithms.dmd import dmd
from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import discretize_instationary_fv, RectGrid, TriaGrid
from pymor.tools.typer import Choices
from typer import Argument, Option, run


def main(
        exp: float = Argument(..., help='Exponent'),

        grid: int = Option(60, help='Use grid with (2*NI)*NI elements.'),
        grid_type: Choices('rect tria') = Option('rect', help='Type of grid to use.'),
        initial_data: Choices('sin bump') = Option('sin', help='Select the initial data.'),
        lxf_lambda: float = Option(1., help='Parameter lambda in Lax-Friedrichs flux.'),
        periodic: bool = Option(True,
                                help='If not, solve with dirichlet boundary conditions on left and bottom boundary.'),
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
    print(f'Solving took {time.perf_counter() - tic}s')
    m.visualize(U)

    # ----- Testing DMD -----
    W1, E1 = dmd(X=U, modes='standard')
    W2, E2 = dmd(X=U, svd_rank=5, modes='exact')

    print('Visualize ...')
    m.visualize(W1, title='DMD Modes - standard')
    plt.plot(E1.real, E1.imag, 'b.')
    plt.title('DMD Eigenvalues - standard')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

    m.visualize(W2, title='5 DMD Modes - exact')
    plt.plot(E2.real, E2.imag, 'b.')
    plt.title('DMD Eigenvalues - exact')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()


if __name__ == '__main__':
    run(main)
