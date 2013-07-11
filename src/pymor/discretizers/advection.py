# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.domaindiscretizers import discretize_domain_default
from pymor.operators.fv import (NonlinearAdvectionLaxFriedrichs, NonlinearAdvectionEngquistOsher, L2Product,
                                L2ProductFunctional)
from pymor.operators import NumpyLinearOperator
from pymor.grids import RectGrid
from pymor.gui.qt import GlumpyPatchVisualizer
from pymor.discretizations import InstationaryNonlinearDiscretization
from pymor.la import induced_norm
from pymor.la import NumpyVectorArray


def discretize_nonlinear_instationary_advection_fv(analytical_problem, diameter=None, nt=100, num_flux='lax_friedrichs',
                                                   lxf_lambda=1., domain_discretizer=None, grid=None, boundary_info=None):

    assert isinstance(analytical_problem, InstationaryAdvectionProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert num_flux in ('lax_friedrichs', 'engquist_osher')

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    p = analytical_problem

    if num_flux == 'lax_friedrichs':
        L = NonlinearAdvectionLaxFriedrichs(grid, boundary_info, p.flux_function, dirichlet_data=p.dirichlet_data,
                                            lxf_lambda=lxf_lambda)
    else:
        L = NonlinearAdvectionEngquistOsher(grid, boundary_info, p.flux_function, p.flux_function_derivative,
                                            dirichlet_data=p.dirichlet_data)
    F = L2ProductFunctional(grid, p.rhs)
    I = p.initial_data.evaluate(grid.quadrature_points(0, order=2)).squeeze()
    I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
    I = NumpyVectorArray(I)

    products = {'l2': L2Product(grid, boundary_info)}
    visualizer = GlumpyPatchVisualizer(grid=grid, bounding_box=grid.domain, codim=0)
    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = InstationaryNonlinearDiscretization(L, F, I, p.T, nt, products=products,
                                                         parameter_space=parameter_space, visualizer=visualizer,
                                                         name='{}_FV'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}
