#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving the Poisson equation in 2D on a circular sector
domain of radius 1 using an unstructured mesh.

Note that Gmsh (http://geuz.org/gmsh/) is required for meshing.

Usage:
    elliptic_unstructured.py ANGLE NUM_POINTS CLSCALE

Arguments:
    ANGLE        The angle of the circular sector.
    NUM_POINTS   The number of points that form the arc of the circular sector.
    CLSCALE      Mesh element size scaling factor.

Options:
    -h, --help   Show this message.
"""

from docopt import docopt
import numpy as np

from pymor.analyticalproblems.domaindescriptions import CircularSectorDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
from pymor.discretizers.builtin import discretize_stationary_cg


def elliptic_gmsh_demo(args):
    args['ANGLE'] = float(args['ANGLE'])
    args['NUM_POINTS'] = int(args['NUM_POINTS'])
    args['CLSCALE'] = float(args['CLSCALE'])

    problem = StationaryProblem(
        domain=CircularSectorDomain(args['ANGLE'], radius=1, num_points=args['NUM_POINTS']),
        diffusion=ConstantFunction(1, dim_domain=2),
        rhs=ConstantFunction(np.array(0.), dim_domain=2, name='rhs'),
        dirichlet_data=ExpressionFunction('sin(polar(x)[1] * pi/angle)', 2, (),
                                          {}, {'angle': args['ANGLE']}, name='dirichlet')
    )

    print('Discretize ...')
    m, data = discretize_stationary_cg(analytical_problem=problem, diameter=args['CLSCALE'])
    grid = data['grid']
    print(grid)
    print()

    print('Solve ...')
    U = m.solve()

    solution = ExpressionFunction('(lambda r, phi: r**(pi/angle) * sin(phi * pi/angle))(*polar(x))', 2, (),
                                  {}, {'angle': args['ANGLE']})
    U_ref = U.space.make_array(solution(grid.centers(2)))

    m.visualize((U, U_ref, U-U_ref),
                legend=('Solution', 'Analytical solution (circular boundary)', 'Error'),
                separate_colorbars=True)


if __name__ == '__main__':
    args = docopt(__doc__)
    elliptic_gmsh_demo(args)
