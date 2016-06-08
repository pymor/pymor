#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
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

import math as m
from docopt import docopt
import numpy as np

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg, discretize_elliptic_fv
from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.domaindescriptions.basic import RectDomain
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.functions.basic import GenericFunction, ConstantFunction
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid


def elliptic_demo(args):
    args['PROBLEM-NUMBER'] = int(args['PROBLEM-NUMBER'])
    assert 0 <= args['PROBLEM-NUMBER'] <= 1, ValueError('Invalid problem number')
    args['DIRICHLET-NUMBER'] = int(args['DIRICHLET-NUMBER'])
    assert 0 <= args['DIRICHLET-NUMBER'] <= 2, ValueError('Invalid Dirichlet boundary number.')
    args['NEUMANN-NUMBER'] = int(args['NEUMANN-NUMBER'])
    assert 0 <= args['NEUMANN-NUMBER'] <= 2, ValueError('Invalid Neumann boundary number.')
    args['NEUMANN-COUNT'] = int(args['NEUMANN-COUNT'])
    assert 0 <= args['NEUMANN-COUNT'] <= 3, ValueError('Invalid Neumann boundary count.')

    rhss = [GenericFunction(lambda X: np.ones(X.shape[:-1]) * 10, 2),
            GenericFunction(lambda X: (X[..., 0] - 0.5) ** 2 * 1000, 2)]
    dirichlets = [GenericFunction(lambda X: np.zeros(X.shape[:-1]), 2),
                  GenericFunction(lambda X: np.ones(X.shape[:-1]), 2),
                  GenericFunction(lambda X: X[..., 0], 2)]
    neumanns = [None,
                ConstantFunction(3., dim_domain=2),
                GenericFunction(lambda X:  50*(0.1 <= X[..., 1]) * (X[..., 1] <= 0.2)
                                          +50*(0.8 <= X[..., 1]) * (X[..., 1] <= 0.9), 2)]
    domains = [RectDomain(),
               RectDomain(right=BoundaryType('neumann')),
               RectDomain(right=BoundaryType('neumann'), top=BoundaryType('neumann')),
               RectDomain(right=BoundaryType('neumann'), top=BoundaryType('neumann'), bottom=BoundaryType('neumann'))]

    rhs = rhss[args['PROBLEM-NUMBER']]
    dirichlet = dirichlets[args['DIRICHLET-NUMBER']]
    neumann = neumanns[args['NEUMANN-NUMBER']]
    domain = domains[args['NEUMANN-COUNT']]

    for n in [32, 128]:
        grid_name = '{1}(({0},{0}))'.format(n, 'RectGrid' if args['--rect'] else 'TriaGrid')
        print('Solving on {0}'.format(grid_name))

        print('Setup problem ...')
        problem = EllipticProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet, neumann_data=neumann)

        print('Discretize ...')
        if args['--rect']:
            grid, bi = discretize_domain_default(problem.domain, diameter=m.sqrt(2) / n, grid_type=RectGrid)
        else:
            grid, bi = discretize_domain_default(problem.domain, diameter=1. / n, grid_type=TriaGrid)
        discretizer = discretize_elliptic_fv if args['--fv'] else discretize_elliptic_cg
        discretization, _ = discretizer(analytical_problem=problem, grid=grid, boundary_info=bi)

        print('Solve ...')
        U = discretization.solve()

        print('Plot ...')
        discretization.visualize(U, title=grid_name)

        print('')


if __name__ == '__main__':
    args = docopt(__doc__)
    elliptic_demo(args)
