#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from typer import Argument, Option, Typer

from pymor.basic import *


app = Typer(help="Solve parabolic equations using pyMOR's builtin discretization toolkit.")
FV = Option(False, help='Use finite volume discretization instead of finite elements.')
GRID = Option(100, help='Use grid with NIxNI intervals.')
NT = Option(100, help='Number of time steps.')
RECT = Option(False, help='Use RectGrid instead of TriaGrid.')


@app.command()
def heat(
    top: float = Argument(..., help='The heat diffusion coefficient for the top bars.'),

    fv: bool = FV,
    grid: int = GRID,
    nt: int = NT,
    rect: bool = RECT,
):
    problem = InstationaryProblem(

        StationaryProblem(
            domain=RectDomain(top='dirichlet', bottom='neumann'),

            diffusion=LincombFunction(
                [ConstantFunction(1., dim_domain=2),
                 ExpressionFunction('(x[0] > 0.45) * (x[0] < 0.55) * (x[1] < 0.7) * 1.',
                                    dim_domain=2),
                 ExpressionFunction('(x[0] > 0.35) * (x[0] < 0.40) * (x[1] > 0.3) * 1. + '
                                    '(x[0] > 0.60) * (x[0] < 0.65) * (x[1] > 0.3) * 1.',
                                    dim_domain=2)],
                [1.,
                 100. - 1.,
                 ExpressionParameterFunctional('top - 1.', {'top': 1})]
            ),

            rhs=ConstantFunction(value=0., dim_domain=2),

            dirichlet_data=ConstantFunction(value=0., dim_domain=2),

            neumann_data=ExpressionFunction('(x[0] > 0.45) * (x[0] < 0.55) * -1000.',
                                            dim_domain=2),
        ),

        T=1.,

        initial_data=ExpressionFunction('(x[0] > 0.45) * (x[0] < 0.55) * (x[1] < 0.7) * 10.',
                                        dim_domain=2)
    )
    mu = {'top': top}
    solve(problem, mu, fv, rect, grid, nt)


@app.command()
def dar(
    speed: float = Argument(..., help='The advection speed.'),

    fv: bool = FV,
    grid: int = GRID,
    nt: int = NT,
    rect: bool = RECT,
):
    problem = InstationaryProblem(

        StationaryProblem(
            domain=RectDomain(),

            diffusion=ConstantFunction(0.01, dim_domain=2),

            advection=LincombFunction([ConstantFunction(np.array([-1., 0]), dim_domain=2)],
                                      [ProjectionParameterFunctional('speed')]),

            reaction=ConstantFunction(0.5, dim_domain=2),

            rhs=ExpressionFunction('(x[0] > 0.3) * (x[0] < 0.7) * (x[1] > 0.3)*(x[1]<0.7) * 0.',
                                   dim_domain=2),

            dirichlet_data=ConstantFunction(value=0., dim_domain=2),
        ),

        T=1.,

        initial_data=ExpressionFunction(
            '(x[0] > 0.3) * (x[0] < 0.7) * (x[1]>0.3) * (x[1] < 0.7) * 10.',
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
