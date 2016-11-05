# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizations.basic import StationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.functions.basic import LincombFunction
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.grids.referenceelements import line, triangle, square
from pymor.gui.qt import PatchVisualizer, Matplotlib1DVisualizer
from pymor.operators import cg, fv
from pymor.operators.constructions import LincombOperator


def discretize_elliptic_cg(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None):
    """Discretizes an |EllipticProblem| using finite elements.

    Parameters
    ----------
    analytical_problem
        The |EllipticProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed as an argument to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If `None`, |discretize_domain_default| is used.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.
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

    assert grid.reference_element in (line, triangle, square)

    if grid.reference_element is square:
        DiffusionOperator = cg.DiffusionOperatorQ1
        AdvectionOperator = cg.AdvectionOperatorQ1
        ReactionOperator  = cg.L2ProductQ1
        Functional = cg.L2ProductFunctionalQ1
    else:
        DiffusionOperator = cg.DiffusionOperatorP1
        AdvectionOperator = cg.AdvectionOperatorP1
        ReactionOperator  = cg.L2ProductP1
        Functional = cg.L2ProductFunctionalP1

    p = analytical_problem

    Li = [DiffusionOperator(grid, boundary_info, diffusion_constant=0, name='boundary_part')]
    coefficients = [1.]

    # diffusion part
    if isinstance(p.diffusion, LincombFunction):
        Li += [DiffusionOperator(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                                 name='diffusion_{}'.format(i))
               for i, df in enumerate(p.diffusion.functions)]
        coefficients += list(p.diffusion.coefficients)
    elif p.diffusion is not None:
        Li += [DiffusionOperator(grid, boundary_info, diffusion_function=p.diffusion,
                                 dirichlet_clear_diag=True, name='diffusion')]
        coefficients.append(1.)

    # advection part
    if isinstance(p.advection, LincombFunction):
        Li += [AdvectionOperator(grid, boundary_info, advection_function=af, dirichlet_clear_diag=True,
                                 name='advection_{}'.format(i))
               for i, af in enumerate(p.advection.functions)]
        coefficients += list(p.advection.coefficients)
    elif p.advection is not None:
        Li += [AdvectionOperator(grid, boundary_info, advection_function=p.advection,
                                 dirichlet_clear_diag=True, name='advection')]
        coefficients.append(1.)

    # reaction part
    if isinstance(p.reaction, LincombFunction):
        Li += [ReactionOperator(grid, boundary_info, coefficient_function=rf, dirichlet_clear_diag=True,
                                name='reaction_{}'.format(i))
               for i, rf in enumerate(p.reaction.functions)]
        coefficients += list(p.reaction.coefficients)
    elif p.reaction is not None:
        Li += [ReactionOperator(grid, boundary_info, coefficient_function=p.reaction,
                                dirichlet_clear_diag=True, name='reaction')]
        coefficients.append(1.)

    # robin boundaries
    if p.robin_data is not None:
        Li += [cg.RobinBoundaryOperator(grid, boundary_info, robin_data=p.robin_data, order=2, name='robin')]
        coefficients.append(1.)

    L = LincombOperator(operators=Li, coefficients=coefficients, name='ellipticOperator')

    F = Functional(grid, p.rhs, boundary_info, dirichlet_data=p.dirichlet_data, neumann_data=p.neumann_data)

    if grid.reference_element in (triangle, square):
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.bounding_box(), codim=2)
    elif grid.reference_element is line:
        visualizer = Matplotlib1DVisualizer(grid=grid, codim=1)
    else:
        visualizer = None

    Prod = cg.L2ProductQ1 if grid.reference_element is square else cg.L2ProductP1
    empty_bi = EmptyBoundaryInfo(grid)
    l2_product = Prod(grid, empty_bi, name='l2')
    l2_0_product = Prod(grid, boundary_info, dirichlet_clear_columns=True, name='l2_0')
    h1_semi_product = DiffusionOperator(grid, empty_bi, name='h1_semi')
    h1_0_semi_product = DiffusionOperator(grid, boundary_info, dirichlet_clear_columns=True, name='h1_0_semi')
    products = {'h1': l2_product + h1_semi_product,
                'h1_semi': h1_semi_product,
                'l2': l2_product,
                'h1_0': l2_0_product + h1_0_semi_product,
                'h1_0_semi': h1_0_semi_product,
                'l2_0': l2_0_product}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_CG'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}


def discretize_elliptic_fv(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None):
    """Discretizes an |EllipticProblem| using the finite volume method.

    Parameters
    ----------
    analytical_problem
        The |EllipticProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed as an argument to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If `None`, |discretize_domain_default| is used.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
    """

    assert isinstance(analytical_problem, EllipticProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if analytical_problem.advection is not None:
        raise NotImplementedError
    if analytical_problem.reaction is not None:
        raise NotImplementedError
    if analytical_problem.robin_data is not None:
        raise NotImplementedError

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    p = analytical_problem

    if isinstance(p.diffusion, LincombFunction):
        Li = [fv.DiffusionOperator(grid, boundary_info, diffusion_function=df, name='diffusion_{}'.format(i))
              for i, df in enumerate(p.diffusion.functions)]
        L = LincombOperator(operators=Li, coefficients=list(p.diffusion.coefficients),
                            name='diffusion')

        F0 = fv.L2ProductFunctional(grid, p.rhs, boundary_info=boundary_info, neumann_data=p.neumann_data)
        if p.dirichlet_data is not None:
            Fi = [fv.L2ProductFunctional(grid, None, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data,
                                         diffusion_function=df, name='dirichlet_{}'.format(i))
                  for i, df in enumerate(p.diffusion.functions)]
            F = LincombOperator(operators=[F0] + Fi, coefficients=[1.] + list(p.diffusion.coefficients),
                                name='rhs')
        else:
            F = F0

    else:
        L = fv.DiffusionOperator(grid, boundary_info, diffusion_function=p.diffusion, name='diffusion')

        F = fv.L2ProductFunctional(grid, p.rhs, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data,
                                   diffusion_function=p.diffusion, neumann_data=p.neumann_data)

    if grid.reference_element in (triangle, square):
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.bounding_box(), codim=0)
    elif grid.reference_element is line:
        visualizer = Matplotlib1DVisualizer(grid=grid, codim=0)
    else:
        visualizer = None

    l2_product = fv.L2Product(grid, name='l2')
    products = {'l2': l2_product}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_FV'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}
