from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.sparse.linalg import bicg
import matplotlib.pyplot as pl

import pymor.core as core
from pymor.common import BoundaryType
from pymor import domaindiscretizers
from pymor.discreteoperators.cg import DiffusionOperatorP1D2, L2ProductFunctionalP1D2
from pymor.discreteoperators.affine import LinearAffinelyDecomposedOperator
from pymor import discretizations
from pymor.grids.tria import TriaGrid


class PoissonCGDiscretizer(object):

    def __init__(self, domain_discretizer=None, diameter=None):
        assert domain_discretizer is None or diameter is None, 'Cannot specifiy domain_discretizer and diameter'
        if diameter is not None:
            domain_discretizer = domaindiscretizers.DefaultDomainDiscretizer(diameter=diameter)
        self.domain_discretizer = domain_discretizer

    def discretize(self, analytical_problem):
        grid, boundary_info = self.domain_discretizer.discretize(analytical_problem.domain)

        assert isinstance(grid, TriaGrid)

        if analytical_problem.parameter_dependent:
            L0 = DiffusionOperatorP1D2(grid, boundary_info, diffusion_constant=0, name='diffusion_boundary_part')

            Li = tuple(DiffusionOperatorP1D2(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                                             name='diffusion_{}'.format(i))
                       for i, df in enumerate(analytical_problem.diffusion_functions))

            L = LinearAffinelyDecomposedOperator(Li, L0, name='diffusion')
        else:
            L = DiffusionOperatorP1D2(grid, boundary_info, diffusion_function=analytical_problem.diffusion_functions[0],
                                      name='diffusion')

        F = L2ProductFunctionalP1D2(grid, boundary_info, analytical_problem.rhs,
                                    dirichlet_data=analytical_problem.dirichlet_data)

        def visualize(U):
            pl.tripcolor(grid.centers(2)[:, 0], grid.centers(2)[:, 1], grid.subentities(0, 2), U)
            pl.colorbar()
            pl.show()

        discr = discretizations.EllipticDiscretization(L, F, visualizer=visualize)

        return discr
