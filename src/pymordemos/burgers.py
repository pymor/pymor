# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import math
import sys
import time
from typing import Literal

from cyclopts import App

from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import RectGrid, TriaGrid, discretize_instationary_fv

TAGS = ('instationary', 'nonlinear', 'parametric', 'builtin-discretizer')

app = App(help_on_error=True)

@app.default
def main(
    exp: float,
    /, *,
    grid: int = 60,
    grid_type: Literal['rect', 'tria'] = 'rect',
    initial_data: Literal['sin', 'bump'] = 'sin',
    lxf_lambda: float = 1.,
    periodic: bool = True,
    nt: int = 100,
    num_flux: Literal['lax_friedrichs', 'engquist_osher', 'simplified_engquist_osher'] = 'engquist_osher',
    vx: float = 1.,
    vy: float = 1.,
):
    r"""Solves a two-dimensional Burgers-type equation.

    This demo solves a two-dimensional Burgers-type equation for a given exponent :math:`\mu`
    in the flux. The analytical problem is given by
    :func:`~pymor.analyticalproblems.burgers.burgers_problem_2d`, where the initial data can be
    chosen as a sine or a bump function. It is discretized with finite volumes using
    :func:`~pymor.discretizers.builtin.fv.discretize_instationary_fv` on a rectangular
    (:class:`~pymor.discretizers.builtin.grids.rect.RectGrid`) or triangular
    (:class:`~pymor.discretizers.builtin.grids.tria.TriaGrid`) grid, and the numerical flux can
    be chosen from the Lax-Friedrichs, Engquist-Osher and simplified Engquist-Osher fluxes.
    The solution is visualized, and the time needed to solve the problem is printed.

    See :doc:`/tutorial_builtin_discretizer` for an introduction to pyMOR's builtin
    discretization toolkit. For model order reduction of this problem using empirical
    interpolation, see :ref:`cyclopts-pymor-demo-burgers_ei`, and for dynamic mode
    decomposition applied to a one-dimensional Burgers-type equation, see
    :ref:`cyclopts-pymor-demo-burgers_dmd`.

    Parameters
    ----------
    exp
        Exponent :math:`\mu` in the flux.
    grid
        Use grid with (2*NI)*NI elements.
    grid_type
        Type of grid to use.
    initial_data
        Select the initial data.
    lxf_lambda
        Parameter lambda in Lax-Friedrichs flux.
    periodic
        If `False`, use Dirichlet boundary conditions on the left and bottom boundary
        and outflow boundary conditions on the top and right boundary.
    nt
        Number of time steps.
    num_flux
        Numerical flux to use.
    vx
        Speed in x-direction.
    vy
        Speed in y-direction.
    """
    print('Setup Problem ...')
    problem = burgers_problem_2d(vx=vx, vy=vy, initial_data_type=initial_data,
                                 parameter_range=(0, 1e42), torus=periodic)

    print('Discretize ...')
    if grid_type == 'rect':
        grid *= 1. / math.sqrt(2)
    m, data = discretize_instationary_fv(
        problem,
        diameter=1. / grid,
        grid_type=RectGrid if grid_type == 'rect' else TriaGrid,
        num_flux=num_flux,
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


if __name__ == '__main__':
    app()
