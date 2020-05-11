#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
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

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, LincombFunction, ConstantFunction
from pymor.discretizers.builtin import discretize_stationary_cg, discretize_stationary_fv
from pymor.parameters.functionals import ProjectionParameterFunctional


def elliptic2_demo(args):
    args['PROBLEM-NUMBER'] = int(args['PROBLEM-NUMBER'])
    assert 0 <= args['PROBLEM-NUMBER'] <= 1, ValueError('Invalid problem number.')
    args['N'] = int(args['N'])

    rhss = [ExpressionFunction('ones(x.shape[:-1]) * 10', 2, ()),
              LincombFunction(
              [ExpressionFunction('ones(x.shape[:-1]) * 10', 2, ()), ConstantFunction(1.,2)],
              [ProjectionParameterFunctional('mu'), 0.1])]

    dirichlets = [ExpressionFunction('zeros(x.shape[:-1])', 2, ()),
                  LincombFunction(
                  [ExpressionFunction('2 * x[..., 0]', 2, ()), ConstantFunction(1.,2)],
                  [ProjectionParameterFunctional('mu'), 0.5])]

    neumanns = [None,
                  LincombFunction(
                  [ExpressionFunction('1 - x[..., 1]', 2, ()), ConstantFunction(1.,2)],
                  [ProjectionParameterFunctional('mu'), 0.5**2])]

    robins = [None,
                (LincombFunction(
                [ExpressionFunction('x[..., 1]', 2, ()), ConstantFunction(1.,2)],
                [ProjectionParameterFunctional('mu'), 1]),
                 ConstantFunction(1.,2))]

    domains = [RectDomain(),
               RectDomain(right='neumann', top='dirichlet', bottom='robin')]

    rhs = rhss[args['PROBLEM-NUMBER']]
    dirichlet = dirichlets[args['PROBLEM-NUMBER']]
    neumann = neumanns[args['PROBLEM-NUMBER']]
    domain = domains[args['PROBLEM-NUMBER']]
    robin = robins[args['PROBLEM-NUMBER']]
    
    problem = StationaryProblem(
        domain=domain,
        rhs=rhs,
        diffusion=LincombFunction(
            [ExpressionFunction('1 - x[..., 0]', 2, ()), ExpressionFunction('x[..., 0]', 2, ())],
            [ProjectionParameterFunctional('mu'), 1]
        ),
        dirichlet_data=dirichlet,
        neumann_data=neumann,
        robin_data=robin,
        parameter_ranges=(0.1, 1),
        name='2DProblem'
    )

    print('Discretize ...')
    discretizer = discretize_stationary_fv if args['--fv'] else discretize_stationary_cg
    m, data = discretizer(problem, diameter=1. / args['N'])
    print(data['grid'])
    print()

    print('Solve ...')
    U = m.solution_space.empty()
    for mu in problem.parameter_space.sample_uniformly(10):
        U.append(m.solve(mu))
    m.visualize(U, title='Solution for mu in [0.1, 1]')


if __name__ == '__main__':
    args = docopt(__doc__)
    elliptic2_demo(args)
