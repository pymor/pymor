#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from typer import Argument, Option, run

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction
from pymor.discretizers.builtin import discretize_stationary_cg, discretize_stationary_fv, RectGrid, TriaGrid


def main(
    problem_number: int = Argument(..., min=0, max=1, help='Selects the problem to solve [0 or 1].'),
    dirichlet_number: int = Argument(..., min=0, max=2, help='Selects the Dirichlet data function [0 to 2].'),
    neumann_number: int = Argument(..., min=0, max=2, help='Selects the Neumann data function.'),
    neumann_count: int = Argument(
        ...,
        min=0,
        max=3,
        help='0: no neumann boundary\n\n'
             '1: right edge is neumann boundary\n\n'
             '2: right+top edges are neumann boundary\n\n'
             '3: right+top+bottom edges are neumann boundary\n\n'
    ),

    fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
    rect: bool = Option(False, help='Use RectGrid instead of TriaGrid.'),
):
    """Solves the Poisson equation in 2D using pyMOR's builtin discreization toolkit."""
    rhss = [ExpressionFunction('10', 2),
            ExpressionFunction('(x[0] - 0.5) ** 2 * 1000', 2)]
    dirichlets = [ExpressionFunction('0', 2),
                  ExpressionFunction('1', 2),
                  ExpressionFunction('x[0]', 2)]
    neumanns = [None,
                ConstantFunction(3., dim_domain=2),
                ExpressionFunction('50*(0.1 <= x[1]) * (x[1] <= 0.2)'
                                   '+50*(0.8 <= x[1]) * (x[1] <= 0.9)', 2)]
    domains = [RectDomain(),
               RectDomain(right='neumann'),
               RectDomain(right='neumann', top='neumann'),
               RectDomain(right='neumann', top='neumann', bottom='neumann')]

    rhs = rhss[problem_number]
    dirichlet = dirichlets[dirichlet_number]
    neumann = neumanns[neumann_number]
    domain = domains[neumann_count]

    problem = StationaryProblem(
        domain=domain,
        diffusion=ConstantFunction(1, dim_domain=2),
        rhs=rhs,
        dirichlet_data=dirichlet,
        neumann_data=neumann,
        outputs=(('l2', ConstantFunction(1, dim_domain=2)),             # average over the domain
                 ('l2_boundary', ConstantFunction(0.25, dim_domain=2)))  # average over the boundary
    )

    for n in [32, 128]:
        print('Discretize ...')
        discretizer = discretize_stationary_fv if fv else discretize_stationary_cg
        m, data = discretizer(
            analytical_problem=problem,
            grid_type=RectGrid if rect else TriaGrid,
            diameter=np.sqrt(2) / n if rect else 1. / n
        )
        grid = data['grid']
        print(grid)
        print()

        print('Solve ...')
        U = m.solve()
        m.visualize(U, title=repr(grid))
        print()

        print('Computing outputs ...')
        S = m.output()
        print(f'  average solution over the domain:   {S[0, 0]}')
        print(f'  average solution over the boundary: {S[0, 1]}')


if __name__ == '__main__':
    run(main)
