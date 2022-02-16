# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.basic import Mu


def test_LincombParameterFunctional():
    dict_of_d_mus = {'mu': ['200 * mu[0]', '2 * mu[0]'], 'nu': ['cos(nu[0])']}

    epf = ExpressionParameterFunctional('100 * mu[0]**2 + 2 * mu[1] * mu[0] + sin(nu[0])',
                                        {'mu': 2, 'nu': 1},
                                        'functional_with_derivative_and_second_derivative',
                                        dict_of_d_mus)
    pf = ProjectionParameterFunctional('mu', 2, 0)
    mu = Mu({'mu': [10, 2], 'nu': [3]})

    zero = pf - pf
    two_pf = pf + pf
    two_pf_named = two_pf.with_(name='some_interesting_quantity')
    three_pf = pf + 2*pf
    three_pf_ = pf + two_pf
    three_pf_named = pf + two_pf_named
    pf_plus_one = pf + 1
    pf_plus_one_ = 1 + pf
    sum_ = epf + pf + 1 - 3
    pf_squared_ = pf * pf
    pf_squared_named = pf_squared_.with_(name='some_interesting_quantity')
    pf_times_pf_squared = pf * pf_squared_
    pf_times_pf_squared_named = pf * pf_squared_named
    pf_squared = 4 * epf * epf * 1 + (pf + 2*epf) * (pf - 2*epf)

    assert zero(mu) == 0
    assert two_pf(mu) == 2 * pf(mu)
    assert three_pf(mu) == 3 * pf(mu)
    assert three_pf_(mu) == 3 * pf(mu)
    assert pf_plus_one(mu) == pf(mu) + 1
    assert pf_plus_one_(mu) == pf(mu) + 1
    assert sum_(mu) == epf(mu) + pf(mu) + 1 - 3
    assert pf_squared_(mu) == pf(mu) * pf(mu)
    assert pf_squared(mu) == pf(mu) * pf(mu)
    assert pf_times_pf_squared(mu) == pf(mu) * pf(mu) * pf(mu)
    # derivatives are linear
    assert sum_.d_mu('mu', 0)(mu) == epf.d_mu('mu', 0)(mu) + pf.d_mu('mu', 0)(mu)
    assert sum_.d_mu('mu', 1)(mu) == epf.d_mu('mu', 1)(mu) + pf.d_mu('mu', 1)(mu)
    assert sum_.d_mu('nu', 0)(mu) == epf.d_mu('nu', 0)(mu) + pf.d_mu('nu', 0)(mu)
    # sums and products are not nested
    assert len(sum_.coefficients) == 4
    assert len(pf_squared.functionals[0].factors) == 4
    # named functions will not be merged and still be nested
    assert three_pf_named(mu) == three_pf_(mu)
    assert len(three_pf_named.coefficients) != len(three_pf_.coefficients)
    assert pf_times_pf_squared_named(mu) == pf_times_pf_squared(mu)
    assert len(pf_times_pf_squared_named.factors) != len(pf_times_pf_squared.factors)
