#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

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

from __future__ import absolute_import, division, print_function

from docopt import docopt
import numpy as np

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg, discretize_elliptic_fv
from pymor.domaindescriptions.polygonal import CircularSectorDomain
from pymor.functions.basic import GenericFunction, ConstantFunction
from pymor.vectorarrays.numpy import NumpyVectorArray


def elliptic_gmsh_demo(args):
    args['ANGLE'] = float(args['ANGLE'])
    args['NUM_POINTS'] = int(args['NUM_POINTS'])
    args['CLSCALE'] = float(args['CLSCALE'])

    domain = CircularSectorDomain(args['ANGLE'], radius=1, num_points=args['NUM_POINTS'])

    rhs = ConstantFunction(np.array(0.), dim_domain=2, name='rhs')

    def dirichlet(X):
        _, phi = polar(X)
        return np.sin(phi*np.pi/args['ANGLE'])

    dirichlet_data = GenericFunction(dirichlet, dim_domain=2, name='dirichlet')

    print('Setup problem ...')
    problem = EllipticProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet_data)

    print('Discretize ...')
    discretizer = discretize_elliptic_fv if args['--fv'] else discretize_elliptic_cg
    discretization, data = discretizer(analytical_problem=problem, diameter=args['CLSCALE'])

    print('Solve ...')
    U = discretization.solve()

    print('Plot ...')

    def ref_sol(X):
        r, phi = polar(X)
        return np.power(r, np.pi/args['ANGLE']) * np.sin(phi*np.pi/args['ANGLE'])

    solution = GenericFunction(ref_sol, 2)
    grid = data['grid']
    U_ref = NumpyVectorArray(solution(grid.centers(0))) if args['--fv'] else NumpyVectorArray(solution(grid.centers(2)))
    discretization.visualize((U, U_ref, U-U_ref),
                             legend=('Solution', 'Analytical solution (circular boundary)', 'Error'),
                             separate_colorbars=True)


def polar(X):
    r = np.sqrt(X[..., 0]**2 + X[..., 1]**2)
    phi = np.zeros(X.shape[:-1])
    part1 = np.all([X[..., 1] >= 0, r > 0], axis=0)
    part2 = np.all([X[..., 1] < 0, r > 0], axis=0)
    phi[part1] = np.arccos(X[..., 0][part1] / r[part1])
    phi[part2] = 2*np.pi - np.arccos(X[..., 0][part2] / r[part2])

    return r, phi


if __name__ == '__main__':
    args = docopt(__doc__)
    elliptic_gmsh_demo(args)
