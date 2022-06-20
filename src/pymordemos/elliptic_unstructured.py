#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from typer import Argument, run

from pymor.analyticalproblems.domaindescriptions import CircularSectorDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
from pymor.discretizers.builtin import discretize_stationary_cg


def main(
    angle: float = Argument(..., help='The angle of the circular sector.'),
    num_points: int = Argument(..., help='The number of points that form the arc of the circular sector.'),
    clscale: float = Argument(..., help='Mesh element size scaling factor.'),
):
    """Solves the 2D-Poisson equation on a circular sector of radius 1 with an unstructured mesh.

    Note that Gmsh (https://gmsh.info/) is required for meshing.
    """
    problem = StationaryProblem(
        domain=CircularSectorDomain(angle, radius=1, num_points=num_points),
        diffusion=ConstantFunction(1, dim_domain=2),
        rhs=ConstantFunction(np.array(0.), dim_domain=2, name='rhs'),
        dirichlet_data=ExpressionFunction('sin(angle(x) * pi/rho)', 2,
                                          {}, {'rho': angle}, name='dirichlet')
    )

    print('Discretize ...')
    m, data = discretize_stationary_cg(analytical_problem=problem, diameter=clscale)
    grid = data['grid']
    print(grid)
    print()

    print('Solve ...')
    U = m.solve()

    solution = ExpressionFunction('norm(x)**(pi/rho) * sin(angle(x) * pi/rho)', 2,
                                  {}, {'rho': angle})
    U_ref = U.space.make_array(solution(grid.centers(2)))

    m.visualize((U, U_ref, U-U_ref),
                legend=('Solution', 'Analytical solution (circular boundary)', 'Error'),
                separate_colorbars=True)


if __name__ == '__main__':
    run(main)
