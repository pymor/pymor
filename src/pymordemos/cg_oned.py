#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Proof of concept for solving the poisson equation in 1D using linear finite elements and our grid interface

Usage:
    cg_oned.py PROBLEM-NUMBER N PLOT

Arguments:
    PROBLEM-NUMBER    {0,1}, selects the problem to solve

    N                 Grid interval count

    PLOT              plot solution after solve?

Options:
    -h, --help    this message
'''

from __future__ import absolute_import, division, print_function

from docopt import docopt
import numpy as np

from pymor.analyticalproblems import EllipticProblem
from pymor.core import getLogger
from pymor.discretizers import discretize_elliptic_cg
from pymor.domaindescriptions import LineDomain
from pymor.functions import GenericFunction, ConstantFunction
from pymor.parameters import CubicParameterSpace, ProjectionParameterFunctional, GenericParameterFunctional

getLogger('pymor.discretizations').setLevel('INFO')


def cg_oned_demo(nrhs, n, plot):
    rhs0 = GenericFunction(lambda X: np.ones(X.shape[:-1]) * 10, dim_domain=1)          # NOQA
    rhs1 = GenericFunction(lambda X: (X[..., 0] - 0.5) ** 2 * 1000, dim_domain=1)       # NOQA

    assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
    rhs = eval('rhs{}'.format(nrhs))

    d0 = GenericFunction(lambda X: 1 - X[..., 0], dim_domain=1)
    d1 = GenericFunction(lambda X: X[..., 0], dim_domain=1)

    parameter_space = CubicParameterSpace({'diffusionl': 0}, 0.1, 1)
    f0 = ProjectionParameterFunctional('diffusionl', 0)
    f1 = GenericParameterFunctional(lambda mu: 1, {})

    print('Solving on OnedGrid(({0},{0}))'.format(n))

    print('Setup Problem ...')
    problem = EllipticProblem(domain=LineDomain(), rhs=rhs, diffusion_functions=(d0, d1),
                              diffusion_functionals=(f0, f1), dirichlet_data=ConstantFunction(value=0, dim_domain=1),
                              name='1DProblem')

    print('Discretize ...')
    discretization, _ = discretize_elliptic_cg(problem, diameter=1 / n)

    print('The parameter type is {}'.format(discretization.parameter_type))

    U = discretization.type_solution.empty(discretization.dim_solution)
    for mu in parameter_space.sample_uniformly(10):
        U.append(discretization.solve(mu))

    if plot:
        print('Plot ...')
        discretization.visualize(U, title='Solution for diffusionl in [0.1, 1]')


if __name__ == '__main__':
    args = docopt(__doc__)
    nrhs = int(args['PROBLEM-NUMBER'])
    n = int(args['N'])
    plot = bool(args['PLOT'])
    cg_oned_demo(nrhs, n, plot)
