# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from pymor.parameters.functionals import (ConstantParameterFunctional, ExpressionParameterFunctional,
                                          MinThetaParameterFunctional, BaseMaxThetaParameterFunctional,
                                          MaxThetaParameterFunctional, ParameterFunctional)
from pymortests.base import runmodule


def test_min_theta_parameter_functional():
    thetas = (ExpressionParameterFunctional('2*mu[0]', {'mu': 1}),
              ConstantParameterFunctional(1),
              1)
    mu_bar = 3
    alpha_mu_bar = 10
    theta = MinThetaParameterFunctional(thetas, mu_bar, alpha_mu_bar)
    thetas = [ConstantParameterFunctional(t) if not isinstance(t, ParameterFunctional) else t
              for t in thetas]
    mu = theta.parameters.parse(1)
    mu_bar = theta.parameters.parse(mu_bar)
    expected_value = alpha_mu_bar * np.min(np.array([t(mu) for t in thetas])/np.array([t(mu_bar) for t in thetas]))
    actual_value = theta.evaluate(mu)
    assert expected_value == actual_value


def test_min_theta_parameter_functional_fails_for_wrong_input():
    thetas = (ExpressionParameterFunctional('2*mu[0]', {'mu': 1}),
              ConstantParameterFunctional(1),
              -1)
    mu_bar = -3
    alpha_mu_bar = 10
    with pytest.raises(AssertionError):
        MinThetaParameterFunctional(thetas, mu_bar, alpha_mu_bar)


def test_max_theta_parameter_functional():
    thetas = (ExpressionParameterFunctional('2*mu[0]', {'mu': 1}),
              ConstantParameterFunctional(1),
              -1)
    mu_bar = -3
    gamma_mu_bar = 10
    theta = MaxThetaParameterFunctional(thetas, mu_bar, gamma_mu_bar)
    thetas = [ConstantParameterFunctional(t) if not isinstance(t, ParameterFunctional) else t
              for t in thetas]
    mu = theta.parameters.parse(1)
    mu_bar = theta.parameters.parse(mu_bar)
    expected_value = gamma_mu_bar * np.abs(np.max(np.array([t(mu) for t in thetas])
                                                  / np.array([t(mu_bar) for t in thetas])))
    actual_value = theta.evaluate(mu)
    assert expected_value == actual_value


def test_base_max_theta_parameter_functional():
    thetas = (ExpressionParameterFunctional('2*mu[0]', {'mu': 1}, derivative_expressions={'mu': ['2']}),
              ConstantParameterFunctional(1))

    # theta prime can for example be the derivatives of all theta
    thetas_prime = tuple([theta.d_mu('mu', 0) for theta in thetas])

    mu_bar = -3
    gamma_mu_bar = 10
    theta = BaseMaxThetaParameterFunctional(thetas_prime, thetas, mu_bar, gamma_mu_bar)
    thetas = [ConstantParameterFunctional(t) if not isinstance(t, ParameterFunctional) else t
              for t in thetas]
    thetas_prime = [ConstantParameterFunctional(t) if not isinstance(t, ParameterFunctional) else t
                    for t in thetas_prime]
    mu = theta.parameters.parse(1)
    mu_bar = theta.parameters.parse(mu_bar)
    expected_value = gamma_mu_bar * np.abs(np.max(np.array([t(mu) for t in thetas_prime])
                                                  / np.array([np.abs(t(mu_bar)) for t in thetas])))
    actual_value = theta.evaluate(mu)
    assert expected_value == actual_value


if __name__ == "__main__":
    runmodule(filename=__file__)
