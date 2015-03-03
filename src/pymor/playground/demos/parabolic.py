#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.analyticalproblems.thermalblock import ThermalBlockProblem
from pymor.discretizations.basic import InstationaryDiscretization
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.vectorarrays.numpy import NumpyVectorArray


def parabolic_demo():
    p = ThermalBlockProblem(parameter_range=(0.01, 1))
    d_stat, d_data = discretize_elliptic_cg(p, diameter=1./100)
    U0 = NumpyVectorArray(np.zeros(d_stat.operator.source.dim))
    time_stepper = ImplicitEulerTimeStepper(50)

    d = InstationaryDiscretization(operator=d_stat.operator, rhs=d_stat.rhs, mass=d_stat.l2_product,
                                   initial_data=U0, T=1, products=d_stat.products, time_stepper=time_stepper,
                                   parameter_space=d_stat.parameter_space, visualizer=d_stat.visualizer)

    mu = next(d.parameter_space.sample_randomly(1))
    R = d.solve(mu)
    d.visualize(R)


if __name__ == '__main__':
    parabolic_demo()
