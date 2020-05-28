#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Proof of concept for solving the Poisson equation in 1D using linear finite elements and our grid interface

Usage:
    elliptic_oned.py [--fv] PROBLEM-NUMBER N

Arguments:
    PROBLEM-NUMBER    {0,1}, selects the problem to solve
    N                 Grid interval count

Options:
    -h, --help   Show this message.
    --fv         Use finite volume discretization instead of finite elements.
"""

from docopt import docopt

from pymor.analyticalproblems.domaindescriptions import LineDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction, LincombFunction
from pymor.discretizers.builtin import discretize_stationary_cg, discretize_stationary_fv
from pymor.parameters.functionals import ProjectionParameterFunctional


def elliptic_oned_demo(args):
    args['PROBLEM-NUMBER'] = int(args['PROBLEM-NUMBER'])
    assert 0 <= args['PROBLEM-NUMBER'] <= 1, ValueError('Invalid problem number.')
    args['N'] = int(args['N'])

    rhss = [ExpressionFunction('ones(x.shape[:-1]) * 10', 1, ()),
            ExpressionFunction('(x - 0.5)**2 * 1000', 1, ())]
    rhs = rhss[args['PROBLEM-NUMBER']]

    d0 = ExpressionFunction('1 - x', 1, ())
    d1 = ExpressionFunction('x', 1, ())

    f0 = ProjectionParameterFunctional('diffusionl')
    f1 = 1.

    problem = StationaryProblem(
        domain=LineDomain(),
        rhs=rhs,
        diffusion=LincombFunction([d0, d1], [f0, f1]),
        dirichlet_data=ConstantFunction(value=0, dim_domain=1),
        name='1DProblem'
    )

    parameter_space = problem.parameters.space(0.1, 1)

    print('Discretize ...')
    discretizer = discretize_stationary_fv if args['--fv'] else discretize_stationary_cg
    m, data = discretizer(problem, diameter=1 / args['N'])
    print(data['grid'])
    print()

    print('Solve ...')
    U = m.solution_space.empty()
    for mu in parameter_space.sample_uniformly(10):
        U.append(m.solve(mu))
    m.visualize(U, title='Solution for diffusionl in [0.1, 1]')


if __name__ == '__main__':
    args = docopt(__doc__)
    elliptic_oned_demo(args)
