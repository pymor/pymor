# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function


from pymor.analyticalproblems import EllipticProblem
from pymor.domaindiscretizers import discretize_domain_default
from pymor.operators.cg import DiffusionOperatorP1, L2ProductFunctionalP1, L2ProductP1
from pymor.discretizations import StationaryDiscretization
from pymor.gui.qt import GlumpyPatchVisualizer, Matplotlib1DVisualizer
from pymor.grids import TriaGrid, OnedGrid, EmptyBoundaryInfo
from pymor.la import induced_norm


def discretize_elliptic_cg(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None):
    '''Discretize an `EllipticProblem` using finite elements.

    Since operators are not assembled during instatiation, calling this function is
    cheap if the domain discretization proceeds quickly.

    Parameters
    ----------
    analytical_problem
        The `EllipticProblem` to discretize.
    diameter
        If not None, is passed to the domain_discretizer.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be function `domain_discretizer(domain_description, diameter=...)`.
        If further arguments should be passed to the discretizer, use
        functools.partial. If None, `discretize_domain_default` is used.
    grid
        Instead of using a domain discretizer, the grid can be passed directly.
    boundary_info
        A `BoundaryInfo` specifying the boundary types of the grid boundary
        entities. Must be provided is `grid` is provided.

    Returns
    -------
    discretization
        The discretization that has been generated.
    data
        Dict with the following entries:
            grid
                The generated grid.
            boundary_info
                The generated `BoundaryInfo`.
    '''

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

    assert isinstance(grid, (OnedGrid, TriaGrid))

    Operator = DiffusionOperatorP1
    Functional = L2ProductFunctionalP1
    p = analytical_problem

    if p.diffusion_functionals is not None or len(p.diffusion_functions) > 1:
        L0 = Operator(grid, boundary_info, diffusion_constant=0, name='diffusion_boundary_part')

        Li = [Operator(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                       name='diffusion_{}'.format(i))
                   for i, df in enumerate(p.diffusion_functions)]

        if p.diffusion_functionals is None:
            L = type(L0).lincomb(operators=Li + [L0], name='diffusion', num_coefficients=len(Li),
                                 global_names={'coefficients': 'diffusion_coefficients'})
        else:
            L = type(L0).lincomb(operators=[L0] + Li, coefficients=[1.] + list(p.diffusion_functionals),
                                 name='diffusion')
    else:
        L = Operator(grid, boundary_info, diffusion_function=p.diffusion_functions[0],
                     name='diffusion')

    F = Functional(grid, p.rhs, boundary_info, dirichlet_data=p.dirichlet_data)

    if isinstance(grid, TriaGrid):
        visualizer = GlumpyPatchVisualizer(grid=grid, bounding_box=grid.domain, codim=2)
        # def visualize(U):
        #     assert len(U) == 1
        #     pl.tripcolor(grid.centers(2)[:, 0], grid.centers(2)[:, 1], grid.subentities(0, 2), U.data.ravel())
        #     pl.colorbar()
        #     pl.show()
    else:
        visualizer = Matplotlib1DVisualizer(grid=grid, codim=1)

    products = {'h1': Operator(grid, boundary_info),
                'l2': L2ProductP1(grid, boundary_info)}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_CG'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}
