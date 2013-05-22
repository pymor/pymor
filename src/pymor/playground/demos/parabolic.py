#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.gui.qt import visualize_glumpy_patch
from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.algorithms.timestepping import implicit_euler

def parabolic_demo():
    p = ThermalBlockProblem(parameter_range=(0.01, 1))
    d, d_data = discretize_elliptic_cg(p, diameter=1./100)

    mu = next(d.parameter_space.sample_randomly(1))
    U0 = NumpyVectorArray(np.zeros(d.operator.dim_source))
    R = implicit_euler(d.operator, d.rhs, d.l2_product, U0, 0, 1, 10, mu)

    visualize_glumpy_patch(d.rhs.grid, R)

    # mu = next(d.parameter_space.sample_randomly(1))
    # U = d.solve(mu)
    # visualize_glumpy_patch(d.rhs.grid, U)

if __name__ == '__main__':
    parabolic_demo()
