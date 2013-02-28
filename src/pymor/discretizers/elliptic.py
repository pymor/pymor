from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as pl

from pymor.analyticalproblems import EllipticProblem
from pymor.domaindiscretizers import DefaultDomainDiscretizer
from pymor.discreteoperators.cg import DiffusionOperatorP1, L2ProductFunctionalP1, L2ProductP1
from pymor.discreteoperators.affine import LinearAffinelyDecomposedOperator
from pymor.discreteoperators import add_operators
from pymor.discretizations import StationaryLinearDiscretization
from pymor.grids import TriaGrid, OnedGrid, EmptyBoundaryInfo
from pymor.la import induced_norm


def discretize_elliptic_cg(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None):

    assert isinstance(analytical_problem, EllipticProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if grid is None:
        domain_discretizer = domain_discretizer or DefaultDomainDiscretizer(analytical_problem.domain)
        if diameter is None:
            grid, boundary_info = domain_discretizer.discretize()
        else:
            grid, boundary_info = domain_discretizer.discretize(diameter=diameter)

    assert isinstance(grid, (OnedGrid, TriaGrid))

    Operator = DiffusionOperatorP1
    Functional = L2ProductFunctionalP1
    p = analytical_problem

    if p.diffusion_functionals is not None or len(p.diffusion_functions) > 1:
        L0 = Operator(grid, boundary_info, diffusion_constant=0, name='diffusion_boundary_part')

        Li = tuple(Operator(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                                         name='diffusion_{}'.format(i))
                   for i, df in enumerate(p.diffusion_functions))

        if p.diffusion_functionals is None:
            L = LinearAffinelyDecomposedOperator(Li, L0, name='diffusion')
            L.rename_parameter({'.coefficients':'.diffusion_coefficients'})
        else:
            L = LinearAffinelyDecomposedOperator(Li, L0, p.diffusion_functionals, name='diffusion')
    else:
        L = Operator(grid, boundary_info, diffusion_function=p.diffusion_functions[0],
                                  name='diffusion')

    F = Functional(grid, p.rhs, boundary_info, dirichlet_data=p.dirichlet_data)

    if isinstance(grid, TriaGrid):
        def visualize(U):
            pl.tripcolor(grid.centers(2)[:, 0], grid.centers(2)[:, 1], grid.subentities(0, 2), U)
            pl.colorbar()
            pl.show()
    else:
        def visualize(U):
            pl.plot(grid.centers(1), U)
            pl.show()
            pass

    discr = StationaryLinearDiscretization(L, F, visualizer=visualize, name='{}_CG'.format(p.name))

    discr.h1_product = add_operators((Operator(grid, EmptyBoundaryInfo(grid)), L2ProductP1(grid)),
                                     name='h1_product')
    discr.h1_norm = induced_norm(discr.h1_product)

    if hasattr(p, 'parameter_space'):
        discr.parameter_space = p.parameter_space

    return discr, {'grid':grid, 'boundary_info':boundary_info}
