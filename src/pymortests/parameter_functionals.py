# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymortests.base import runmodule
import pytest

import numpy as np

from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional, LincombParameterFunctional
from pymor.basic import Mu


def test_LincombParameterFunctional():
    dict_of_d_mus = {'mu': ['200 * mu[0]', '2 * mu[0]'], 'nu': ['cos(nu[0])']}

    epf = ExpressionParameterFunctional('100 * mu[0]**2 + 2 * mu[1] * mu[0] + sin(nu[0])',
                                        {'mu': 2, 'nu': 1},
                                        'functional_with_derivative_and_second_derivative',
                                        dict_of_d_mus)
    pf = ProjectionParameterFunctional('mu', 2, 0)
    mu = Mu({'mu': [10,2], 'nu': [3]})

    zero = pf - pf 
    two_pf = pf + pf
    three_pf = pf + 2*pf
    pf_plus_one = pf + 1
    sum_ = epf + pf 
    pf_squared =  (pf + 2*epf) * (pf - 2*epf) + 4 * epf * epf  

    assert zero(mu) == 0
    assert two_pf(mu) == 2 * pf(mu)
    assert three_pf(mu) == 3 * pf(mu)
    assert pf_plus_one(mu) == pf(mu) + 1
    assert sum_(mu) == epf(mu) + pf(mu)
    assert pf_squared(mu) == pf(mu) * pf(mu)
    assert sum_.d_mu('mu', 0)(mu) == epf.d_mu('mu', 0)(mu) + pf.d_mu('mu', 0)(mu)
    assert sum_.d_mu('mu', 1)(mu) == epf.d_mu('mu', 1)(mu) + pf.d_mu('mu', 1)(mu)
    assert sum_.d_mu('nu', 0)(mu) == epf.d_mu('nu', 0)(mu) + pf.d_mu('nu', 0)(mu)
