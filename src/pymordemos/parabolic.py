#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving the heat equation using pyMOR's builtin discretizations.

Usage:
    parabolic.py [-h] [--help] [--fv] [--rect] [--grid=NI] [--nt=COUNT] TOP

Arguments:
    TOP          The heat diffusion coefficient for the top bars.

Options:
    -h, --help   Show this message.

    --fv         Use finite volume discretization instead of finite elements.

    --rect       Use RectGrid instead of TriaGrid.

    --grid=NI    Use grid with NIxNI intervals [default: 100].

    --nt=COUNT   Number of time steps [default: 10].
"""

from docopt import docopt
import numpy as np

from pymor.basic import *


def parabolic_demo(args):
    args['TOP'] = float(args['TOP'])
    args['--nt'] = int(args['--nt'])
    args['--grid'] = int(args['--grid'])

    grid_name = '{1}(({0},{0}))'.format(args['--grid'], 'RectGrid' if args['--rect'] else 'TriaGrid')
    print('Solving on {0}'.format(grid_name))

    problem = ParabolicProblem(
        domain=RectDomain(top=BoundaryType('dirichlet'), bottom=BoundaryType('neumann')),

        diffusion_functions=[ConstantFunction(1., dim_domain=2),
                             ExpressionFunction('(x[..., 0] > 0.45) * (x[..., 0] < 0.55) * (x[..., 1] < 0.7) * 1.',
                                                dim_domain=2),
                             ExpressionFunction('(x[..., 0] > 0.35) * (x[..., 0] < 0.40) * (x[..., 1] > 0.3) * 1. + ' +
                                                '(x[..., 0] > 0.60) * (x[..., 0] < 0.65) * (x[..., 1] > 0.3) * 1.',
                                                dim_domain=2)],

        diffusion_functionals=[1.,
                               100. - 1.,
                               ExpressionParameterFunctional('top - 1.', {'top': 0})],

        rhs=ConstantFunction(value=0., dim_domain=2),

        dirichlet_data=ConstantFunction(value=0., dim_domain=2),

        neumann_data=ExpressionFunction('(x[..., 0] > 0.45) * (x[..., 0] < 0.55) * -1000.',
                                        dim_domain=2),

        initial_data=ExpressionFunction('(x[..., 0] > 0.45) * (x[..., 0] < 0.55) * (x[..., 1] < 0.7) * 10.',
                                        dim_domain=2),

        parameter_space=CubicParameterSpace({'top': 0}, minimum=1, maximum=100.)
    )

    print('Discretize ...')
    if args['--rect']:
        grid, bi = discretize_domain_default(problem.domain, diameter=np.sqrt(2) / args['--grid'], grid_type=RectGrid)
    else:
        grid, bi = discretize_domain_default(problem.domain, diameter=1. / args['--grid'], grid_type=TriaGrid)
    discretizer = discretize_parabolic_fv if args['--fv'] else discretize_parabolic_cg
    discretization, _ = discretizer(analytical_problem=problem, grid=grid, boundary_info=bi, nt=args['--nt'])

    print('The parameter type is {}'.format(discretization.parameter_type))

    U = discretization.solve({'top': args['TOP']})

    print('Plot ...')
    discretization.visualize(U, title=grid_name)

    print('')


if __name__ == '__main__':
    args = docopt(__doc__)
    parabolic_demo(args)
