#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''
Proof of concept for solving the poisson equation in 1D using linear finite elements and our grid interface
'''

from __future__ import absolute_import, division, print_function

import sys
import math as m

import numpy as np

from pymor.domaindescriptions import BoundaryType
from pymor.domaindescriptions import RectDomain
from pymor.analyticalproblems import EllipticProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.functions import GenericFunction


def cg_demo(nrhs, ndirichlet, nneumann):
    rhs0 = GenericFunction(lambda X: np.ones(X.shape[:-1]) * 10, 2)
    rhs1 = GenericFunction(lambda X: (X[..., 0] - 0.5) ** 2 * 1000, 2)
    dirichlet0 = GenericFunction(lambda X: np.zeros(X.shape[:-1]), 2)
    dirichlet1 = GenericFunction(lambda X: np.ones(X.shape[:-1]), 2)
    dirichlet2 = GenericFunction(lambda X: X[..., 0], 2)
    domain0 = RectDomain()
    domain1 = RectDomain(right=BoundaryType('neumann'))
    domain2 = RectDomain(right=BoundaryType('neumann'), top=BoundaryType('neumann'))
    domain3 = RectDomain(right=BoundaryType('neumann'), top=BoundaryType('neumann'), bottom=BoundaryType('neumann'))

    assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
    rhs = eval('rhs{}'.format(nrhs))

    assert 0 <= ndirichlet <= 2, ValueError('Invalid boundary number.')
    dirichlet = eval('dirichlet{}'.format(ndirichlet))

    assert 0 <= nneumann <= 3, ValueError('Invalid neumann boundary count.')
    domain = eval('domain{}'.format(nneumann))


    for n in [32, 128]
        print('Solving on TriaGrid(({0},{0}))'.format(n))

        print('Setup problem ...')
        problem = EllipticProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet)

        print('Discretize ...')
        discretization, _ = discretize_elliptic_cg(problem, diameter=m.sqrt(2) / n)

        print('Solve ...')
        U = discretization.solve()
        
        print('Plot ...')
        discretization.visualize(U)

        print('')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit('Usage: {} RHS-NUMBER BOUNDARY-DATA-NUMBER NEUMANN-COUNT'.format(sys.argv[0]))
    nrhs = int(sys.argv[1])
    ndirichlet = int(sys.argv[2])
    nneumann = int(sys.argv[3])
    cg_demo(nrhs, ndirichlet, nneumann)
