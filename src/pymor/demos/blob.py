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
from pymor.functions import GenericFunction, ConstantFunction
from pymor.grids import RectGrid, AllDirichletBoundaryInfo


def blob_demo(nrhs, ndirichlet, nneumann):
    rhs = GenericFunction(lambda X: (1. - 1. * ((X[...,0]-1.5)**2+(X[...,1]-0.5)**2 > 0.01) * ((X[...,0]-0.5)**2+(X[...,1]-0.25)**2 > 0.01) * ((X[...,0]-0.5)**2+(X[...,1]-0.75)**2 > 0.01)), dim_domain=2,name='RHS',dim_range=1)
    #rhs = GenericFunction(lambda X: 1. * ((X[...,0])**2 > 1))
    #rhs = GenericFunction(lambda X: (X[..., 0] + X[..., 1] - 0.5) ** 2 * 1000, 2)
    #dirichlet0 = GenericFunction(lambda X: np.zeros(X.shape[:-1]), 2)
    #dirichlet1 = GenericFunction(lambda X: np.ones(X.shape[:-1]), 2)
    #dirichlet2 = GenericFunction(lambda X: X[..., 0], 2)
    domain = RectDomain(domain=[[0,0],[2,1]])
    #domain1 = RectDomain(right=BoundaryType('neumann'))
    #domain2 = RectDomain(right=BoundaryType('neumann'), top=BoundaryType('neumann'))
    #domain3 = RectDomain(right=BoundaryType('neumann'), top=BoundaryType('neumann'), bottom=BoundaryType('neumann'))

    for n in [32, 128]:
        print('Solving on TriaGrid(({0},{0}))'.format(n))

        print('Setup problem ...')
        problem = EllipticProblem(domain=domain, rhs=rhs, diffusion_functions=(ConstantFunction(dim_domain=2),), diffusion_functionals=None, dirichlet_data=ConstantFunction(value=0, dim_domain=2))

        print('Discretize ...')
        discretization, _ = discretize_elliptic_cg(problem, diameter=m.sqrt(2) / (n*4))
        #gitter = RectGrid(num_intervals=(200,200),domain=[[0,0],[2,1]])
        #discretization, _ = discretize_elliptic_cg(problem, diameter=m.sqrt(2) / (n*4), grid=gitter, boundary_info=AllDirichletBoundaryInfo(gitter))


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
    blob_demo(nrhs, ndirichlet, nneumann)
