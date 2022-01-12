#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from typer import Argument, Option, run

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, LincombFunction, ConstantFunction
from pymor.discretizers.builtin import discretize_stationary_cg, discretize_stationary_fv
from pymor.parameters.functionals import ProjectionParameterFunctional


def main(
    problem_number: int = Argument(..., min=0, max=1, help='Selects the problem to solve [0 or 1].'),
    n: int = Argument(..., help='Triangle count per direction'),
    norm: str = Argument(
        ...,
        help="h1: compute the h1-norm of the last snapshot.\n\n"
             "l2: compute the l2-norm of the last snapshot.\n\n"
             "k: compute the energy norm of the last snapshot, where the energy-product"
             "is constructed with a parameter {'mu': k}."
    ),

    fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
):
    """Solves the Poisson equation in 2D using pyMOR's builtin discreization toolkit."""
    norm = float(norm) if not norm.lower() in ('h1', 'l2') else norm.lower()

    rhss = [ExpressionFunction('10', 2),
            LincombFunction(
                [ExpressionFunction('10', 2), ConstantFunction(1., 2)],
                [ProjectionParameterFunctional('mu'), 0.1])]

    dirichlets = [ExpressionFunction('0', 2),
                  LincombFunction(
                  [ExpressionFunction('2 * x[0]', 2), ConstantFunction(1., 2)],
                  [ProjectionParameterFunctional('mu'), 0.5])]

    neumanns = [None,
                LincombFunction(
                    [ExpressionFunction('1 - x[1]', 2), ConstantFunction(1., 2)],
                    [ProjectionParameterFunctional('mu'), 0.5**2])]

    robins = [None,
              (LincombFunction(
                  [ExpressionFunction('x[1]', 2), ConstantFunction(1., 2)],
                  [ProjectionParameterFunctional('mu'), 1]), ConstantFunction(1., 2))]

    domains = [RectDomain(),
               RectDomain(right='neumann', top='dirichlet', bottom='robin')]

    rhs = rhss[problem_number]
    dirichlet = dirichlets[problem_number]
    neumann = neumanns[problem_number]
    domain = domains[problem_number]
    robin = robins[problem_number]

    problem = StationaryProblem(
        domain=domain,
        rhs=rhs,
        diffusion=LincombFunction(
            [ExpressionFunction('1 - x[0]', 2), ExpressionFunction('x[0]', 2)],
            [ProjectionParameterFunctional('mu'), 1]
        ),
        dirichlet_data=dirichlet,
        neumann_data=neumann,
        robin_data=robin,
        parameter_ranges=(0.1, 1),
        name='2DProblem'
    )

    if isinstance(norm, float) and not fv:
        # use a random parameter to construct an energy product
        mu_bar = problem.parameters.parse(norm)
    else:
        mu_bar = None

    print('Discretize ...')
    if fv:
        m, data = discretize_stationary_fv(problem, diameter=1. / n)
    else:
        m, data = discretize_stationary_cg(problem, diameter=1. / n, mu_energy_product=mu_bar)
    print(data['grid'])
    print()

    print('Solve ...')
    U = m.solution_space.empty()
    for mu in problem.parameter_space.sample_uniformly(10):
        U.append(m.solve(mu))
    if mu_bar is not None:
        # use the given energy product
        norm_squared = U[-1].norm(m.products['energy'])[0]
        print('Energy norm of the last snapshot: ', np.sqrt(norm_squared))
    if not fv:
        if norm == 'h1':
            norm_squared = U[-1].norm(m.products['h1_0_semi'])[0]
            print('H^1_0 semi norm of the last snapshot: ', np.sqrt(norm_squared))
        if norm == 'l2':
            norm_squared = U[-1].norm(m.products['l2_0'])[0]
            print('L^2_0 norm of the last snapshot: ', np.sqrt(norm_squared))
    m.visualize(U, title='Solution for mu in [0.1, 1]')


if __name__ == '__main__':
    run(main)
