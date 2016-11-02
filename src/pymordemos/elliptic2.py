#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving the Poisson equation in 2D using pyMOR's builtin discretizations.

Usage:
    elliptic2.py [--fv] PROBLEM-NUMBER N

Arguments:
    PROBLEM-NUMBER    {0,1}, selects the problem to solve

    N                 Triangle count per direction

Options:
    -h, --help   Show this message.

    --fv         Use finite volume discretization instead of finite elements.
"""

from docopt import docopt

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg, discretize_elliptic_fv
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ExpressionFunction, LincombFunction
from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace


def elliptic2_demo(args):
    args['PROBLEM-NUMBER'] = int(args['PROBLEM-NUMBER'])
    assert 0 <= args['PROBLEM-NUMBER'] <= 1, ValueError('Invalid problem number.')
    args['N'] = int(args['N'])

    rhss = [ExpressionFunction('ones(x.shape[:-1]) * 10', 2, ()),
            ExpressionFunction('(x[..., 0] - 0.5)**2 * 1000', 2, ())]
    rhs = rhss[args['PROBLEM-NUMBER']]

    print('Solving on TriaGrid(({0},{0}))'.format(args['N']))

    print('Setup Problem ...')
    problem = EllipticProblem(
        domain=RectDomain(),
        rhs=rhs,
        diffusion=LincombFunction(
            [ExpressionFunction('1 - x[..., 0]', 2, ()), ExpressionFunction('x[..., 0]', 2, ())],
            [ProjectionParameterFunctional('diffusionl', 0), ExpressionParameterFunctional('1', {})]
        ),
        parameter_space=CubicParameterSpace({'diffusionl': 0}, 0.1, 1),
        name='2DProblem')

    print('Discretize ...')
    discretizer = discretize_elliptic_fv if args['--fv'] else discretize_elliptic_cg
    discretization, _ = discretizer(problem, diameter=1. / args['N'])

    print('The parameter type is {}'.format(discretization.parameter_type))

    U = discretization.solution_space.empty()
    for mu in discretization.parameter_space.sample_uniformly(10):
        U.append(discretization.solve(mu))

    print('Plot ...')
    discretization.visualize(U, title='Solution for diffusionl in [0.1, 1]')

if __name__ == '__main__':
    args = docopt(__doc__)
    elliptic2_demo(args)
