#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving parabolic equations using pyMOR's builtin discretization toolkit.

Usage:
    parabolic.py [options] heat TOP
    parabolic.py [options] dar SPEED

Arguments:
    TOP          The heat diffusion coefficient for the top bars.
    SPEED        The advection speed.

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


def parabolic_demo(args):
    args['--nt'] = int(args['--nt'])
    args['--grid'] = int(args['--grid'])

    if args['heat']:
        args['TOP'] = float(args['TOP'])
        problem = InstationaryProblem(

            StationaryProblem(
                domain=RectDomain(top='dirichlet', bottom='neumann'),

                diffusion=LincombFunction(
                    [ConstantFunction(1., dim_domain=2),
                     ExpressionFunction('(x[..., 0] > 0.45) * (x[..., 0] < 0.55) * (x[..., 1] < 0.7) * 1.',
                                        dim_domain=2),
                     ExpressionFunction('(x[..., 0] > 0.35) * (x[..., 0] < 0.40) * (x[..., 1] > 0.3) * 1. + '
                                        '(x[..., 0] > 0.60) * (x[..., 0] < 0.65) * (x[..., 1] > 0.3) * 1.',
                                        dim_domain=2)],
                    [1.,
                     100. - 1.,
                     ExpressionParameterFunctional('top - 1.', {'top': 1})]
                ),

                rhs=ConstantFunction(value=0., dim_domain=2),

                dirichlet_data=ConstantFunction(value=0., dim_domain=2),

                neumann_data=ExpressionFunction('(x[..., 0] > 0.45) * (x[..., 0] < 0.55) * -1000.',
                                                dim_domain=2),
            ),

            T=1.,

            initial_data=ExpressionFunction('(x[..., 0] > 0.45) * (x[..., 0] < 0.55) * (x[..., 1] < 0.7) * 10.',
                                            dim_domain=2)
        )
    else:
        args['SPEED'] = float(args['SPEED'])
        problem = InstationaryProblem(

            StationaryProblem(
                domain=RectDomain(),

                diffusion=ConstantFunction(0.01, dim_domain=2),

                advection=LincombFunction([ConstantFunction(np.array([-1., 0]), dim_domain=2)],
                                          [ProjectionParameterFunctional('speed')]),

                reaction=ConstantFunction(0.5, dim_domain=2),

                rhs=ExpressionFunction('(x[..., 0] > 0.3) * (x[..., 0] < 0.7) * (x[..., 1] > 0.3)*(x[...,1]<0.7) * 0.',
                                       dim_domain=2),

                dirichlet_data=ConstantFunction(value=0., dim_domain=2),
            ),

            T=1.,

            initial_data=ExpressionFunction('(x[..., 0] > 0.3) * (x[..., 0] < 0.7) * (x[...,1]>0.3) * (x[..., 1] < 0.7) * 10.',
                                            dim_domain=2),
        )

    print('Discretize ...')
    discretizer = discretize_instationary_fv if args['--fv'] else discretize_instationary_cg
    m, data = discretizer(
        analytical_problem=problem,
        grid_type=RectGrid if args['--rect'] else TriaGrid,
        diameter=np.sqrt(2) / args['--grid'] if args['--rect'] else 1. / args['--grid'],
        nt=args['--nt']
    )
    grid = data['grid']
    print(grid)
    print()

    print('Solve ...')
    U = m.solve({'top': args['TOP']} if args['heat'] else {'speed': args['SPEED']})
    m.visualize(U, title='Solution')

    print('')


if __name__ == '__main__':
    args = docopt(__doc__)
    parabolic_demo(args)
