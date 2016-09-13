#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving a parabolic equation with diffusion, reaction, and advection using pyMOR's
builtin discretizations.

Usage:
    parabolic_extended.py [-h] [--help] [--fv] [--rect] [--grid=NI] [--nt=COUNT]

Arguments:


Options:
    -h, --help   Show this message.

    --fv         Use finite volume discretization instead of finite elements.

    --rect       Use RectGrid instead of TriaGrid.

    --grid=NI    Use grid with NIxNI intervals [default: 100].

    --nt=COUNT   Number of time steps [default: 100].
"""

from docopt import docopt
import numpy as np

from pymor.basic import *
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper

def parabolic_demo(args):

    epsilon = np.array([[0.01,0],[0,0.01]])
    b = -0.2
    c = 0.5

    args['--nt'] = int(args['--nt'])
    args['--grid'] = int(args['--grid'])

    grid_name = '{1}(({0},{0}))'.format(args['--grid'], 'RectGrid' if args['--rect'] else 'TriaGrid')
    print('Solving on {0}'.format(grid_name))

    problem = ParabolicProblem(
        domain=RectDomain(),

        diffusion_functions=[ConstantFunction(epsilon, dim_domain=2),],

        advection_functions=[ConstantFunction(np.array([b,0]), dim_domain=2),],

        reaction_functions=[ConstantFunction(c,dim_domain=2),],

        rhs=ExpressionFunction('(x[..., 0] > 0.3) * (x[..., 0] < 0.7) * (x[..., 1] > 0.3)*(x[...,1]<0.7) * 0.',
                                        dim_domain=2),
        dirichlet_data=ConstantFunction(value=0., dim_domain=2),

        initial_data=ExpressionFunction('(x[..., 0] > 0.3) * (x[..., 0] < 0.7) * (x[...,1]>0.3) * (x[..., 1] < 0.7) * 10.',
                                        dim_domain=2),
        )

    print('Discretize ...')
    if args['--rect']:
        grid, bi = discretize_domain_default(problem.domain, diameter=np.sqrt(2) / args['--grid'], grid_type=RectGrid)
    else:
        grid, bi = discretize_domain_default(problem.domain, diameter=1. / args['--grid'], grid_type=TriaGrid)
    discretizer = discretize_parabolic_fv if args['--fv'] else discretize_parabolic_cg
    discretization, _ = discretizer(analytical_problem=problem, grid=grid, boundary_info=bi, nt=args['--nt'])


    print('The parameter type is {}'.format(discretization.parameter_type))

    U = discretization.solve()

    print('Plot ...')
    discretization.visualize(U, title=grid_name)

    print('')


if __name__ == '__main__':
    args = docopt(__doc__)
    parabolic_demo(args)
