#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving the Poisson equation in 2D using pyMOR's builtin discretizations.

Usage:
    elliptic.py [--fv] [--rect] PROBLEM-NUMBER DIRICHLET-NUMBER NEUMANN-NUMBER NEUMANN-COUNT

Arguments:
    PROBLEM-NUMBER    {0,1}, selects the problem to solve
    DIRICHLET-NUMBER  {0,1,2}, selects the Dirichlet data function
    NEUMANN-NUMBER    {0,1}, selects the Neumann data function

    NEUMANN-COUNT     0: no neumann boundary
                      1: right edge is neumann boundary
                      2: right+top edges are neumann boundary
                      3: right+top+bottom edges are neumann boundary

Options:
    -h, --help   Show this message.
    --fv         Use finite volume discretization instead of finite elements.
    --rect       Use RectGrid instead of TriaGrid.
"""

from docopt import docopt
import numpy as np

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction
from pymor.discretizers.builtin import discretize_stationary_cg, discretize_stationary_fv, RectGrid, TriaGrid


def elliptic_demo(args):
    args['PROBLEM-NUMBER'] = int(args['PROBLEM-NUMBER'])
    assert 0 <= args['PROBLEM-NUMBER'] <= 1, ValueError('Invalid problem number')
    args['DIRICHLET-NUMBER'] = int(args['DIRICHLET-NUMBER'])
    assert 0 <= args['DIRICHLET-NUMBER'] <= 2, ValueError('Invalid Dirichlet boundary number.')
    args['NEUMANN-NUMBER'] = int(args['NEUMANN-NUMBER'])
    assert 0 <= args['NEUMANN-NUMBER'] <= 2, ValueError('Invalid Neumann boundary number.')
    args['NEUMANN-COUNT'] = int(args['NEUMANN-COUNT'])
    assert 0 <= args['NEUMANN-COUNT'] <= 3, ValueError('Invalid Neumann boundary count.')

    rhss = [ExpressionFunction('ones(x.shape[:-1]) * 10', 2, ()),
            ExpressionFunction('(x[..., 0] - 0.5) ** 2 * 1000', 2, ())]
    dirichlets = [ExpressionFunction('zeros(x.shape[:-1])', 2, ()),
                  ExpressionFunction('ones(x.shape[:-1])', 2, ()),
                  ExpressionFunction('x[..., 0]', 2, ())]
    neumanns = [None,
                ConstantFunction(3., dim_domain=2),
                ExpressionFunction('50*(0.1 <= x[..., 1]) * (x[..., 1] <= 0.2)'
                                   '+50*(0.8 <= x[..., 1]) * (x[..., 1] <= 0.9)', 2, ())]
    domains = [RectDomain(),
               RectDomain(right='neumann'),
               RectDomain(right='neumann', top='neumann'),
               RectDomain(right='neumann', top='neumann', bottom='neumann')]

    rhs = rhss[args['PROBLEM-NUMBER']]
    dirichlet = dirichlets[args['DIRICHLET-NUMBER']]
    neumann = neumanns[args['NEUMANN-NUMBER']]
    domain = domains[args['NEUMANN-COUNT']]

    problem = StationaryProblem(
        domain=domain,
        diffusion=ConstantFunction(1, dim_domain=2),
        rhs=rhs,
        dirichlet_data=dirichlet,
        neumann_data=neumann
    )

    for n in [32, 128]:
        print('Discretize ...')
        discretizer = discretize_stationary_fv if args['--fv'] else discretize_stationary_cg
        m, data = discretizer(
            analytical_problem=problem,
            grid_type=RectGrid if args['--rect'] else TriaGrid,
            diameter=np.sqrt(2) / n if args['--rect'] else 1. / n
        )
        grid = data['grid']
        print(grid)
        print()

        print('Solve ...')
        U = m.solve()
        m.visualize(U, title=repr(grid))
        print()


if __name__ == '__main__':
    args = docopt(__doc__)
    elliptic_demo(args)
