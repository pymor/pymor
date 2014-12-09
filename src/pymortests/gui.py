# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import multiprocessing
from pymor.grids.oned import OnedGrid
from time import sleep
from pymor.gui.qt import visualize_patch

import pytest
import numpy as np
from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.rect import RectGrid

from pymortests.base import runmodule
from pymor.domaindescriptions.basic import RectDomain, LineDomain
from pymor.functions.basic import GenericFunction


@pytest.fixture(params=(('matplotlib', RectGrid), ('gl', RectGrid), ('matplotlib', OnedGrid)))
def backend_gridtype(request):
    return request.param


def test_visualize_patch(backend_gridtype):
    backend, gridtype = backend_gridtype
    domain = LineDomain() if gridtype is OnedGrid else RectDomain()
    dim = 1 if gridtype is OnedGrid else 2
    rhs = GenericFunction(lambda X: np.ones(X.shape[:-1]) * 10, dim)  # NOQA
    dirichlet = GenericFunction(lambda X: np.zeros(X.shape[:-1]), dim)  # NOQA
    diffusion = GenericFunction(lambda X: np.ones(X.shape[:-1]), dim)  # NOQA
    problem = EllipticProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet, diffusion_functions=(diffusion,))
    grid, bi = discretize_domain_default(problem.domain, grid_type=gridtype)
    discretization, data = discretize_elliptic_cg(analytical_problem=problem, grid=grid, boundary_info=bi)
    U = discretization.solve()
    visualize_patch(data['grid'], U=U, backend=backend)
    sleep(2)  # so gui has a chance to popup
    for child in multiprocessing.active_children():
        child.terminate()


if __name__ == "__main__":
    runmodule(filename=__file__)
