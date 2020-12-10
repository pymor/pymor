# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymortests.base import runmodule
import pytest

import numpy as np
from pymor.algorithms.greedy import rb_greedy
from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
from pymordemos.linear_optimization import create_fom

def test_output_estimate():
    grid_intervals = 10
    training_samples = 10

    fom, mu_bar = create_fom(grid_intervals, vector_valued_output=False)

    parameter_space = fom.parameters.space(0, np.pi)
    training_set = parameter_space.sample_uniformly(training_samples)
    random_set = parameter_space.sample_randomly(training_samples)
    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)

    RB_reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    RB_greedy_data = rb_greedy(fom, RB_reductor, training_set, atol=1e-2)
    rom = RB_greedy_data['rom']

    for mu in random_set:
        s_rom, s_est = rom.output(return_error_estimate=True, mu=mu)
        s_fom = fom.output(mu=mu)
        for s_r, s_f, s_e in np.dstack((s_rom, s_fom, s_est))[0]:
            assert np.abs(s_r-s_f) < s_e
