#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Proof of concept for solving the poisson equation in 2D using linear finite elements and our grid interface

Usage:
    cg.py PROBLEM-NUMBER DIRICHLET-NUMBER NEUMANN-COUNT

Arguments:
    PROBLEM-NUMBER    {0,1}, selects the problem to solve

    DIRICHLET-NUMBER  {0,1,2}, selects the dirichlet data function

    NEUMANN-COUNT  0: no neumann boundary
                   1: right edge is neumann boundary
                   2: right+top edges are neumann boundary
                   3: right+top+bottom edges are neumann boundary

Options:
    -h, --help    this message
'''

from __future__ import absolute_import, division, print_function

import sys
import math as m
from docopt import docopt
import numpy as np

from pymor.analyticalproblems import EllipticProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.domaindescriptions import BoundaryType
from pymor.domaindescriptions import RectDomain
from pymor.functions import GenericFunction


def cg_demo(nrhs, ndirichlet, nneumann):
    rhs0 = GenericFunction(lambda X: np.ones(X.shape[:-1]) * 10, 2)                      # NOQA
    rhs1 = GenericFunction(lambda X: (X[..., 0] - 0.5) ** 2 * 1000, 2)                   # NOQA
    dirichlet0 = GenericFunction(lambda X: np.zeros(X.shape[:-1]), 2)                    # NOQA
    dirichlet1 = GenericFunction(lambda X: np.ones(X.shape[:-1]), 2)                     # NOQA
    dirichlet2 = GenericFunction(lambda X: X[..., 0], 2)                                 # NOQA
    domain0 = RectDomain()                                                               # NOQA
    domain1 = RectDomain(right=BoundaryType('neumann'))                                  # NOQA
    domain2 = RectDomain(right=BoundaryType('neumann'), top=BoundaryType('neumann'))     # NOQA
    domain3 = RectDomain(right=BoundaryType('neumann'), top=BoundaryType('neumann'), bottom=BoundaryType('neumann'))  # NOQA

    assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
    rhs = eval('rhs{}'.format(nrhs))

    assert 0 <= ndirichlet <= 2, ValueError('Invalid boundary number.')
    dirichlet = eval('dirichlet{}'.format(ndirichlet))

    assert 0 <= nneumann <= 3, ValueError('Invalid neumann boundary count.')
    domain = eval('domain{}'.format(nneumann))

    for n in [32, 128]:
        print('Solving on TriaGrid(({0},{0}))'.format(n))

        print('Setup problem ...')
        problem = EllipticProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet)

        print('Discretize ...')
        discretization, _ = discretize_elliptic_cg(problem, diameter=m.sqrt(2) / n)

        print('Solve ...')
        U = discretization.solve()

        print('Plot ...')
        discretization.visualize(U, title='Triagrid(({0},{0}))'.format(n))

        print('')


if __name__ == '__main__':
    args = docopt(__doc__)
    nrhs = int(args['PROBLEM-NUMBER'])
    ndirichlet = int(args['DIRICHLET-NUMBER'])
    nneumann = int(args['NEUMANN-COUNT'])
    cg_demo(nrhs, ndirichlet, nneumann)
