# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pytest
import numpy as np

from pymor.algorithms.tr import trust_region
from pymor.core.exceptions import TRError
from pymordemos.linear_optimization import create_fom
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.parameters.functionals import MinThetaParameterFunctional


def test_tr():
    fom, mu_bar = create_fom(10)
    parameter_space = fom.parameters.space(0, np.pi)
    initial_guess = fom.parameters.parse([0.25, 0.5])
    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)
    reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)

    # successful run
    _, data = trust_region(parameter_space, reductor, radius=.1, initial_guess=initial_guess)
    assert len(data['mus']) == 22

    # failing run
    # reset reductor
    reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    with pytest.raises(TRError):
        mu, _ = trust_region(parameter_space, reductor, radius=.1, initial_guess=initial_guess, maxiter=10)
