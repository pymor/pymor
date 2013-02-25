from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.sparse.linalg import bicg
import matplotlib.pyplot as pl

import pymor.core as core
from pymor.analyticalproblems import EllipticProblem
from pymor.domaindescriptions import BoundaryType
from pymor.domaindiscretizers import DefaultDomainDiscretizer
from pymor.discreteoperators.cg import DiffusionOperatorP1, L2ProductFunctionalP1, L2ProductP1
from pymor.discreteoperators.affine import LinearAffinelyDecomposedOperator
from pymor.discreteoperators import GenericLinearOperator, add_operators
from pymor.discretizations import StationaryLinearDiscretization
from pymor.grids import TriaGrid, OnedGrid, EmptyBoundaryInfo
from pymor.la import induced_norm


class EllipticCGDiscretizer(object):

    def __init__(self, analytical_problem):
        assert isinstance(analytical_problem, EllipticProblem)
        self.analytical_problem = analytical_problem

    def discretize_domain(self, domain_discretizer=None, diameter=None):
        domain_discretizer = domain_discretizer or DefaultDomainDiscretizer(self.analytical_problem.domain)
        if diameter is None:
            return domain_discretizer.discretize()
        else:
            return domain_discretizer.discretize(diameter=diameter)

    def discretize(self, domain_discretizer=None, diameter=None, grid=None, boundary_info=None,
                   h1_product=True):
        assert grid is None or boundary_info is None
        assert boundary_info is None or grid is None
        assert grid is None or domain_discretizer is None
        if grid is None:
            grid, boundary_info = self.discretize_domain(domain_discretizer, diameter)

        assert isinstance(grid, (OnedGrid, TriaGrid))

        Operator = DiffusionOperatorP1
        Functional = L2ProductFunctionalP1

        p = self.analytical_problem

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

        F = Functional(grid, boundary_info, p.rhs, dirichlet_data=p.dirichlet_data)

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

        if h1_product:
            discr.h1_product = add_operators((Operator(grid, EmptyBoundaryInfo(grid)), L2ProductP1(grid)),
                                             name='h1_product')
            discr.h1_norm = induced_norm(discr.h1_product)

        if hasattr(p, 'parameter_space'):
            discr.parameter_space = p.parameter_space

        return discr
