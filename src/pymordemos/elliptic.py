# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import Literal

import numpy as np
from cyclopts import App

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
from pymor.discretizers.builtin import RectGrid, TriaGrid, discretize_stationary_cg, discretize_stationary_fv

TAGS = ('stationary', 'linear', 'builtin-discretizer')

app = App(help_on_error=True)

@app.default
def main(
    problem_number: Literal[0, 1],
    dirichlet_number: Literal[0, 1, 2],
    neumann_number: Literal[0, 1, 2],
    neumann_count: Literal[0, 1, 2, 3],
    /, *,
    fv: bool = False,
    rect: bool = False
):
    r"""Solves the Poisson equation in 2D using pyMOR's builtin discretization toolkit.

    This demo solves a Poisson equation :math:`-\Delta u = f` on the unit square with Dirichlet
    and Neumann boundary conditions. The analytical problem is given as a
    :class:`~pymor.analyticalproblems.elliptic.StationaryProblem`, where the data can be chosen
    as follows:

    - right-hand side: :math:`f = 10` (0) or :math:`f = 1000 (x_1 - 0.5)^2` (1),
    - Dirichlet data: :math:`0` (0), :math:`1` (1) or :math:`x_1` (2),
    - Neumann data: none (0), the constant :math:`3` (1), or :math:`50` on the strips
      :math:`0.1 \leq x_2 \leq 0.2` and :math:`0.8 \leq x_2 \leq 0.9` and zero elsewhere (2).

    The edges of the domain that form the Neumann boundary are selected by `neumann_count`. The
    problem is discretized with finite elements
    (:func:`~pymor.discretizers.builtin.cg.discretize_stationary_cg`) or finite volumes
    (:func:`~pymor.discretizers.builtin.fv.discretize_stationary_fv`) on a triangular or
    rectangular grid at two different resolutions. For each resolution, the solution is
    visualized, and its averages over the domain and over the boundary, which are defined as
    outputs of the problem, are printed.

    See :doc:`/tutorial_builtin_discretizer` for an introduction to pyMOR's builtin
    discretization toolkit. For parametric elliptic problems, see
    :ref:`cyclopts-pymor-demo-elliptic2` and :ref:`cyclopts-pymor-demo-elliptic_oned`, and for
    an unstructured mesh, see :ref:`cyclopts-pymor-demo-elliptic_unstructured`.

    Parameters
    ----------
    problem_number
        Selects the problem to solve.
    dirichlet_number
        Selects the Dirichlet data function.
    neumann_number
        Selects the Neumann data function.
    neumann_count
        0: no neumann boundary,
        1: right edge is neumann boundary,
        2: right+top edges are neumann boundary,
        3: right+top+bottom edges are neumann boundary.
    fv
        Use finite volume discretization instead of finite elements.
    rect
        Use RectGrid instead of TriaGrid.
    """
    rhss = [ExpressionFunction('10', 2),
            ExpressionFunction('(x[0] - 0.5) ** 2 * 1000', 2)]
    dirichlets = [ExpressionFunction('0', 2),
                  ExpressionFunction('1', 2),
                  ExpressionFunction('x[0]', 2)]
    neumanns = [None,
                ConstantFunction(3., dim_domain=2),
                ExpressionFunction('50*(0.1 <= x[1] <= 0.2)'
                                   '+50*(0.8 <= x[1] <= 0.9)', 2)]
    domains = [RectDomain(),
               RectDomain(right='neumann'),
               RectDomain(right='neumann', top='neumann'),
               RectDomain(right='neumann', top='neumann', bottom='neumann')]

    rhs = rhss[problem_number]
    dirichlet = dirichlets[dirichlet_number]
    neumann = neumanns[neumann_number]
    domain = domains[neumann_count]

    problem = StationaryProblem(
        domain=domain,
        diffusion=ConstantFunction(1, dim_domain=2),
        rhs=rhs,
        dirichlet_data=dirichlet,
        neumann_data=neumann,
        outputs=(('l2', ConstantFunction(1, dim_domain=2)),             # average over the domain
                 ('l2_boundary', ConstantFunction(0.25, dim_domain=2)))  # average over the boundary
    )

    for n in [32, 128]:
        print('Discretize ...')
        discretizer = discretize_stationary_fv if fv else discretize_stationary_cg
        m, data = discretizer(
            analytical_problem=problem,
            grid_type=RectGrid if rect else TriaGrid,
            diameter=np.sqrt(2) / n if rect else 1. / n
        )
        grid = data['grid']
        print(grid)
        print()

        print('Solve ...')
        U = m.solve()
        m.visualize(U, title=repr(grid))
        print()

        print('Computing outputs ...')
        S = m.output()
        print(f'  average solution over the domain:   {S[0, 0]}')
        print(f'  average solution over the boundary: {S[1, 0]}')


if __name__ == '__main__':
    app()
