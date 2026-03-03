# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from cyclopts import App

from pymor.basic import *

app = App(help_on_error=True)

@app.command
def heat(
    top: float,
    /, *,
    fv: bool = False,
    grid: int = 100,
    nt: int = 100,
    rect: bool = False,
):
    """Heat conduction problem.

    Parameters
    ----------
    top
        The heat diffusion coefficient for the top bars.
    fv
        Use finite volume discretization instead of finite elements.
    grid
        Use grid with NIxNI intervals.
    nt
        Number of time steps.
    rect
        Use RectGrid instead of TriaGrid.
    """
    problem = InstationaryProblem(

        StationaryProblem(
            domain=RectDomain(top='dirichlet', bottom='neumann'),

            diffusion=LincombFunction(
                [ConstantFunction(1., dim_domain=2),
                 ExpressionFunction('(0.45 < x[0] < 0.55) * (x[1] < 0.7) * 1.',
                                    dim_domain=2),
                 ExpressionFunction('(0.35 < x[0] < 0.40) * (x[1] > 0.3) * 1. + '
                                    '(0.60 < x[0] < 0.65) * (x[1] > 0.3) * 1.',
                                    dim_domain=2)],
                [1.,
                 100. - 1.,
                 ExpressionParameterFunctional('top - 1.', {'top': 1})]
            ),

            rhs=ConstantFunction(value=0., dim_domain=2),

            dirichlet_data=ConstantFunction(value=0., dim_domain=2),

            neumann_data=ExpressionFunction('(0.45 < x[0] < 0.55) * -1000.',
                                            dim_domain=2),
        ),

        T=1.,

        initial_data=ExpressionFunction('(0.45 < x[0] < 0.55) * (x[1] < 0.7) * 10.',
                                        dim_domain=2)
    )
    mu = {'top': top}
    solve(problem, mu, fv, rect, grid, nt)


@app.command()
def dar(
    speed: float,
    /, *,
    fv: bool = False,
    grid: int = 100,
    nt: int = 100,
    rect: bool = False,
):
    """Diffusion advection reaction problem.

    Parameters
    ----------
    speed
        The advection speed.
    fv
        Use finite volume discretization instead of finite elements.
    grid
        Use grid with NIxNI intervals.
    nt
        Number of time steps.
    rect
        Use RectGrid instead of TriaGrid.
    """
    problem = InstationaryProblem(

        StationaryProblem(
            domain=RectDomain(),

            diffusion=ConstantFunction(0.01, dim_domain=2),

            advection=LincombFunction([ConstantFunction(np.array([-1., 0]), dim_domain=2)],
                                      [ProjectionParameterFunctional('speed')]),

            reaction=ConstantFunction(0.5, dim_domain=2),

            rhs=ExpressionFunction('(0.3 < x[0] < 0.7) * (0.3 < x[1] < 0.7) * 0.',
                                   dim_domain=2),

            dirichlet_data=ConstantFunction(value=0., dim_domain=2),
        ),

        T=1.,

        initial_data=ExpressionFunction(
            '(0.3 < x[0] < 0.7) * (0.3 < x[1] < 0.7) * 10.',
            dim_domain=2),
    )
    mu = {'speed': speed}
    solve(problem, mu, fv, rect, grid, nt)


def solve(problem, mu, fv, rect, grid, nt):
    print('Discretize ...')
    discretizer = discretize_instationary_fv if fv else discretize_instationary_cg
    m, data = discretizer(
        analytical_problem=problem,
        grid_type=RectGrid if rect else TriaGrid,
        diameter=np.sqrt(2) / grid if rect else 1. / grid,
        nt=nt
    )
    grid = data['grid']
    print(grid)
    print()

    print('Solve ...')
    U = m.solve(mu)
    m.visualize(U, title='Solution')

    print('')


if __name__ == '__main__':
    app()
