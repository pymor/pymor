# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from time import sleep

import numpy as np
import pytest

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.core.exceptions import PySideMissing
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.domaindescriptions.basic import LineDomain, RectDomain
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.functions.basic import GenericFunction
from pymor.grids.oned import OnedGrid
from pymor.grids.rect import RectGrid
from pymor.gui.qt import stop_gui_processes, visualize_patch
from pymortests.base import runmodule


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
    problem = EllipticProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet, diffusion=diffusion)
    grid, bi = discretize_domain_default(problem.domain, grid_type=gridtype)
    discretization, data = discretize_elliptic_cg(analytical_problem=problem, grid=grid, boundary_info=bi)
    U = discretization.solve()
    try:
        visualize_patch(data['grid'], U=U, backend=backend)
    except PySideMissing as ie:
        pytest.xfail("PySide missing")
    finally:
        stop_gui_processes()


if __name__ == "__main__":
    runmodule(filename=__file__)
