# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: lucas-ca <lucascamp@web.de>

from __future__ import absolute_import, division, print_function


from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizations.basic import StationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.grids.oned import OnedGrid
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid
from pymor.gui.qt import PatchVisualizer, Matplotlib1DVisualizer
from pymor.operators.cg import DiffusionOperatorP1, L2ProductFunctionalP1, L2ProductP1,\
    DiffusionOperatorQ1, L2ProductFunctionalQ1, L2ProductQ1
from pymor.operators.constructions import LincombOperator


def discretize_elliptic_cg(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None):
    """Discretizes an |EllipticProblem| using finite elements.

    Parameters
    ----------
    analytical_problem
        The |EllipticProblem| to discretize.
    diameter
        If not None, is passed to the domain_discretizer.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If further arguments should be passed to the discretizer, use
        :func:`functools.partial`. If `None`, |discretize_domain_default| is used.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is provided.

    Returns
    -------
    discretization
        The discretization that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
    """

    assert isinstance(analytical_problem, EllipticProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    assert isinstance(grid, (OnedGrid, TriaGrid, RectGrid))

    if isinstance(grid, RectGrid):
        Operator = DiffusionOperatorQ1
        Functional = L2ProductFunctionalQ1
    else:
        Operator = DiffusionOperatorP1
        Functional = L2ProductFunctionalP1

    p = analytical_problem

    if p.diffusion_functionals is not None or len(p.diffusion_functions) > 1:
        L0 = Operator(grid, boundary_info, diffusion_constant=0, name='diffusion_boundary_part')

        Li = [Operator(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                       name='diffusion_{}'.format(i))
              for i, df in enumerate(p.diffusion_functions)]

        if p.diffusion_functionals is None:
            L = LincombOperator(operators=Li + [L0], name='diffusion', num_coefficients=len(Li),
                                coefficients_name='diffusion_coefficients')
        else:
            L = LincombOperator(operators=[L0] + Li, coefficients=[1.] + list(p.diffusion_functionals),
                                name='diffusion')
    else:
        L = Operator(grid, boundary_info, diffusion_function=p.diffusion_functions[0],
                     name='diffusion')

    F = Functional(grid, p.rhs, boundary_info, dirichlet_data=p.dirichlet_data, neumann_data=p.neumann_data)

    if isinstance(grid, (TriaGrid, RectGrid)):
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.domain, codim=2)
    else:
        visualizer = Matplotlib1DVisualizer(grid=grid, codim=1)

    empty_bi = EmptyBoundaryInfo(grid)
    l2_product = L2ProductQ1(grid, empty_bi) if isinstance(grid, RectGrid) else L2ProductP1(grid, empty_bi)
    h1_semi_product = Operator(grid, empty_bi)
    products = {'h1': l2_product + h1_semi_product,
                'h1_semi': h1_semi_product,
                'l2': l2_product}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_CG'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}
