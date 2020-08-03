#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving the Poisson equation in 2D using pyMOR's builtin discretizations.

Usage:
    elliptic2.py [--fv] PROBLEM-NUMBER N PRODUCT-COUNT

Arguments:
    PROBLEM-NUMBER  {0,1}, selects the problem to solve
    N               Triangle count per direction

    PRODUCT-COUNT   0: equip the model with standard products
                    1: use a fixed parameter instance with random parameter values to also
                    add an energy product to the model

Options:
    -h, --help   Show this message.
    --fv         Use finite volume discretization instead of finite elements.
"""

from docopt import docopt
import numpy as np

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, LincombFunction, ConstantFunction
from pymor.discretizers.builtin import discretize_stationary_cg, discretize_stationary_fv
from pymor.parameters.functionals import ProjectionParameterFunctional


def elliptic2_demo(args):
    args['PROBLEM-NUMBER'] = int(args['PROBLEM-NUMBER'])
    assert 0 <= args['PROBLEM-NUMBER'] <= 1, ValueError('Invalid problem number.')
    args['N'] = int(args['N'])
    args['PRODUCT-COUNT'] = int(args['PRODUCT-COUNT'])
    assert 0 <= args['PRODUCT-COUNT'] <= 1, ValueError('Invalid product count.')

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

    if args['PRODUCT-COUNT'] and not args['--fv']:
        # use a random parameter to construct an energy product
        mu_bar = problem.parameter_space.sample_randomly(1)[0]
    else:
        mu_bar = None

    print('Discretize ...')
    discretizer = discretize_stationary_fv if args['--fv'] else discretize_stationary_cg
    if mu_bar is not None:
        m, data = discretizer(problem, diameter=1. / args['N'], mu_energy_product=mu_bar)
    else:
        m, data = discretizer(problem, diameter=1. / args['N'])
    print(data['grid'])
    print()

    print('Solve ...')
    U = m.solution_space.empty()
    for mu in problem.parameter_space.sample_uniformly(10):
        U.append(m.solve(mu))
    if mu_bar is not None:
        # use the given energy product
        energy_norm_squared = m.products['energy'].apply2(U[-1], U[-1])
        print('Energy norm of the last snapshot: ', np.sqrt(energy_norm_squared)[0][0])
    if not args['--fv']:
        h1_0_norm_squared = m.products['h1_0_semi'].apply2(U[-1], U[-1])
        print('H^1_0 semi norm of the last snapshot: ', np.sqrt(h1_0_norm_squared)[0][0])
    m.visualize(U, title='Solution for mu in [0.1, 1]')


if __name__ == '__main__':
    args = docopt(__doc__)
    elliptic2_demo(args)
