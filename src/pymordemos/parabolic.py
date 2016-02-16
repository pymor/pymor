#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving the heat equation in 2D using pyMOR's builtin discretizations.

Usage:
    parabolic.py [-h] [--help] [--fv] [--rect] [--grid=NI] [--nt=COUNT] DIFF

Arguments:
    DIFF         The diffusion constant

Options:
    -h, --help   Show this message.

    --fv         Use finite volume discretization instead of finite elements.

    --rect       Use RectGrid instead of TriaGrid.

    --grid=NI    Use grid with NIxNI intervals [default: 100].

    --nt=COUNT   Number of time steps [default: 10].
"""

from __future__ import absolute_import, division, print_function

import math as m
from docopt import docopt
import numpy as np

from pymor.analyticalproblems.parabolic import ParabolicProblem
from pymor.discretizers.parabolic import discretize_parabolic_cg, discretize_parabolic_fv
from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.domaindescriptions.basic import RectDomain
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.functions.basic import GenericFunction, ConstantFunction
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid


def parabolic_demo(args):
    args['DIFF'] = float(args['DIFF'])
    args['--nt'] = int(args['--nt'])
    args['--grid'] = int(args['--grid'])

    grid_name = '{1}(({0},{0}))'.format(args['--grid'], 'RectGrid' if args['--rect'] else 'TriaGrid')
    print('Solving on {0}'.format(grid_name))

    print('Setup problem ...')
    domain = RectDomain(top=BoundaryType('neumann'), bottom=BoundaryType('neumann'))
    rhs = ConstantFunction(value=0, dim_domain=2)
    diffusion_functional = GenericParameterFunctional(mapping=lambda mu: mu['diffusion'],
                                                      parameter_type={'diffusion': 0})
    neumann = ConstantFunction(value=-1., dim_domain=2)
    initial = GenericFunction(lambda X: np.cos(np.pi*X[..., 0])*np.sin(np.pi*X[..., 1]), dim_domain=2)

    problem = ParabolicProblem(domain=domain, rhs=rhs, diffusion_functionals=[diffusion_functional],
                               dirichlet_data=initial, neumann_data=neumann, initial_data=initial)

    print('Discretize ...')
    if args['--rect']:
        grid, bi = discretize_domain_default(problem.domain, diameter=m.sqrt(2) / args['--grid'], grid_type=RectGrid)
    else:
        grid, bi = discretize_domain_default(problem.domain, diameter=1. / args['--grid'], grid_type=TriaGrid)
    discretizer = discretize_parabolic_fv if args['--fv'] else discretize_parabolic_cg
    discretization, _ = discretizer(analytical_problem=problem, grid=grid, boundary_info=bi, nt=args['--nt'])

    print('The parameter type is {}'.format(discretization.parameter_type))

    mu = {'diffusion': args['DIFF']}
    print('Solving for diffusion = {} ... '.format(mu['diffusion']))
    U = discretization.solve(mu)

    print('Plot ...')
    discretization.visualize(U, title=grid_name)

    print('')


if __name__ == '__main__':
    args = docopt(__doc__)
    parabolic_demo(args)
