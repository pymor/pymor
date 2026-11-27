# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import Literal

import numpy as np
from cyclopts import App

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
from pymor.discretizers.builtin import discretize_stationary_cg, discretize_stationary_fv
from pymor.parameters.functionals import ProjectionParameterFunctional

TAGS = ('stationary', 'linear', 'parametric', 'builtin-discretizer')

app = App(help_on_error=True)

@app.default
def main(
    problem_number: Literal[0, 1],
    n: int,
    norm: Literal['h1', 'l2'] | float,
    /, *,
    fv: bool = False
):
    r"""Solves a parametric elliptic problem in 2D using pyMOR's builtin discretization toolkit.

    This demo solves the parametric elliptic problem

    .. math::

        -\nabla \cdot \big(((1 - x_1)\mu + x_1) \nabla u\big) = f

    on the unit square, where :math:`\mu \in [0.1, 1]` is the parameter. Problem number 0 uses
    :math:`f = 10` and homogeneous Dirichlet boundary conditions. In problem number 1, the
    right-hand side as well as the Dirichlet data on the left and top edges, the Neumann data on
    the right edge and the Robin data on the bottom edge depend affinely on :math:`\mu`. The
    problem is discretized with finite elements
    (:func:`~pymor.discretizers.builtin.cg.discretize_stationary_cg`) or finite volumes
    (:func:`~pymor.discretizers.builtin.fv.discretize_stationary_fv`) and solved for ten
    uniformly sampled parameters, and the solutions are visualized. For finite elements, the
    :math:`H^1_0`-seminorm, the :math:`L^2`-norm or the energy norm induced by the operator for
    a fixed parameter value of the last solution is printed.

    See :doc:`/tutorial_builtin_discretizer` for an introduction to pyMOR's builtin
    discretization toolkit. For the non-parametric Poisson equation, see
    :ref:`cyclopts-pymor-demo-elliptic`, and for a one-dimensional parametric problem, see
    :ref:`cyclopts-pymor-demo-elliptic_oned`.

    Parameters
    ----------
    problem_number
        Selects the problem to solve [0 or 1].
    n
        Triangle count per direction.
    norm
        Only used for finite elements.
        h1: compute the h1-norm of the last snapshot;
        l2: compute the l2-norm of the last snapshot;
        k: compute the energy norm of the last snapshot, where the energy-product is constructed
           with a parameter {'mu': k}.
    fv
        Use finite volume discretization instead of finite elements.
    """
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
    app()
