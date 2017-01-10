#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving the Poisson equation in 2D on a circular sector
domain of radius 1 using an unstructured mesh.

Note that Gmsh (http://geuz.org/gmsh/) is required for meshing.

Usage:
    elliptic_unstructured.py [--fv] ANGLE NUM_POINTS CLSCALE

Arguments:
    ANGLE        The angle of the circular sector.

    NUM_POINTS   The number of points that form the arc of the circular sector.

    CLSCALE      Mesh element size scaling factor.


Options:
    -h, --help   Show this message.

    --fv         Use finite volume discretization instead of finite elements.
"""

import numpy as np
from docopt import docopt

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg, discretize_elliptic_fv
from pymor.domaindescriptions.polygonal import CircularSectorDomain
from pymor.functions.basic import ConstantFunction, ExpressionFunction


def elliptic_gmsh_demo(args):
    args['ANGLE'] = float(args['ANGLE'])
    args['NUM_POINTS'] = int(args['NUM_POINTS'])
    args['CLSCALE'] = float(args['CLSCALE'])

    print('Setup problem ...')
    problem = EllipticProblem(
        domain=CircularSectorDomain(args['ANGLE'], radius=1, num_points=args['NUM_POINTS']),
        diffusion=ConstantFunction(1, dim_domain=2),
        rhs=ConstantFunction(np.array(0.), dim_domain=2, name='rhs'),
        dirichlet_data=ExpressionFunction('sin(polar(x)[1] * pi/angle)', 2, (),
                                          {}, {'angle': args['ANGLE']}, name='dirichlet')
    )

    print('Discretize ...')
    discretizer = discretize_elliptic_fv if args['--fv'] else discretize_elliptic_cg
    discretization, data = discretizer(analytical_problem=problem, diameter=args['CLSCALE'])

    print('Solve ...')
    U = discretization.solve()

    print('Plot ...')

    solution = ExpressionFunction('(lambda r, phi: r**(pi/angle) * sin(phi * pi/angle))(*polar(x))', 2, (),
                                  {}, {'angle': args['ANGLE']})
    grid = data['grid']
    U_ref = U.space.make_array(solution(grid.centers(0)) if args['--fv'] else solution(grid.centers(2)))
    discretization.visualize((U, U_ref, U-U_ref),
                             legend=('Solution', 'Analytical solution (circular boundary)', 'Error'),
                             separate_colorbars=True)


if __name__ == '__main__':
    args = docopt(__doc__)
    elliptic_gmsh_demo(args)
