# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from pymordemos.linear_optimization import create_fom

from pymor.algorithms.tr import trust_region
from pymor.core.exceptions import TRError
from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor


def test_tr():
    fom, mu_bar = create_fom(10)
    mu_opt = [1.42454, np.pi]
    parameter_space = fom.parameters.space(0, np.pi)
    initial_guess = fom.parameters.parse([0.25, 0.5])
    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)
    reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)

    # successful run
    mu_opt_tr, data = trust_region(reductor, parameter_space, radius=.1, initial_guess=initial_guess)
    assert len(data['mus']) == 4
    assert np.allclose(mu_opt_tr, mu_opt)

    # failing run
    # reset reductor
    reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    with pytest.raises(TRError):
        mu, _ = trust_region(reductor, parameter_space, radius=.1, initial_guess=initial_guess, maxiter=10, rtol=1e-6)
