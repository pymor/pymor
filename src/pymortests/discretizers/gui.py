# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.analyticalproblems.domaindescriptions import LineDomain, RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import GenericFunction
from pymor.core.exceptions import QtMissingError
from pymor.discretizers.builtin import RectGrid, discretize_stationary_cg
from pymor.discretizers.builtin.domaindiscretizers.default import discretize_domain_default
from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymortests.base import runmodule

pytestmark = pytest.mark.builtin


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
    problem = StationaryProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet, diffusion=diffusion)
    grid, bi = discretize_domain_default(problem.domain, grid_type=gridtype)
    m, data = discretize_stationary_cg(analytical_problem=problem, grid=grid, boundary_info=bi)
    U = m.solve()
    try:
        if dim == 1:
            from pymor.discretizers.builtin.gui.qt import visualize_matplotlib_1d
            visualize_matplotlib_1d(data['grid'], U=U)
        else:
            from pymor.discretizers.builtin.gui.qt import visualize_patch
            visualize_patch(data['grid'], U=U, backend=backend)
    except QtMissingError:
        pytest.xfail('Qt missing')


if __name__ == '__main__':
    runmodule(filename=__file__)
