# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App

from pymor.algorithms.dmd import dmd
from pymor.analyticalproblems.burgers import burgers_problem
from pymor.discretizers.builtin import discretize_instationary_fv

TAGS = ('instationary', 'nonlinear', 'data-driven', 'builtin-discretizer')

app = App(help_on_error=True)

@app.default
def main(
    exp: float,
    /, *,
    atol: float | None = None,
    continuous_time: bool = False,
    grid: int = 100,
    initial_data: Literal['sin', 'bump'] = 'sin',
    modes: int | None = None,
    nt: int = 1000,
    periodic: bool = True,
    rtol: float | None = None,
    v: float = 10.
):
    r"""Dynamic mode decomposition of a one-dimensional Burgers-type equation.

    This demo computes the dynamic mode decomposition (DMD) of the solution trajectory of the
    one-dimensional Burgers-type equation

    .. math::

        \partial_t u + \partial_x (v u^\mu) = 0, \qquad x \in [0, 2],\ t \in [0, 0.3],

    given by :func:`~pymor.analyticalproblems.burgers.burgers_problem` for a fixed exponent
    :math:`\mu`. The problem is discretized with finite volumes using the Engquist-Osher flux
    (:func:`~pymor.discretizers.builtin.fv.discretize_instationary_fv`). The full-order solution
    is computed and visualized, and its DMD is computed using
    :func:`~pymor.algorithms.dmd.dmd` (:cite:`TRLBK14`) in the standard and the exact variant,
    where the number of modes is determined by truncating the singular values of the snapshot
    matrix. The plots show the real and imaginary parts of the standard and exact DMD modes, as
    well as the DMD eigenvalues, either as discrete-time eigenvalues together with the unit
    circle or as continuous-time eigenvalues.

    For the identification of a linear system from trajectory data using DMD, see
    :ref:`cyclopts-pymor-demo-dmd_identification`. For the two-dimensional Burgers-type
    equation, see :ref:`cyclopts-pymor-demo-burgers`.

    Parameters
    ----------
    exp
        Exponent :math:`\mu` in the flux.
    atol
        Absolute truncation tolerance for the singular values of the snapshot matrix.
    continuous_time
        Show continuous time system eigenvalues.
    grid
        Use grid with 2*NI elements.
    initial_data
        Select the initial data (sin, bump).
    modes
        Maximum number of DMD modes.
    nt
        Number of time steps.
    periodic
        If `False`, use Dirichlet boundary conditions on the left and outflow boundary
        conditions on the right boundary.
    rtol
        Relative truncation tolerance for the singular values of the snapshot matrix.
    v
        Speed in x-direction.
    """
    print('Setup Problem ...')
    problem = burgers_problem(v=v, initial_data_type=initial_data, circle=periodic)

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
    app()
