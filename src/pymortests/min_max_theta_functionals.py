import numpy as np
import pytest
from pymor.parameters.functionals import (
        ConstantParameterFunctional,
        ExpressionParameterFunctional,
        MinThetaParameterFunctional,
        MaxThetaParameterFunctional,
        ParameterFunctional)
from pymortests.base import runmodule


def test_min_theta_parameter_functional():
    thetas = (ExpressionParameterFunctional('2*mu', {'mu': ()}),
              ConstantParameterFunctional(1),
              1)
    mu_bar = 3
    alpha_mu_bar = 10
    theta = MinThetaParameterFunctional(thetas, mu_bar, alpha_mu_bar)
    thetas = [ConstantParameterFunctional(t) if not isinstance(t, ParameterFunctional) else t
              for t in thetas]
    mu = 1
    expected_value = alpha_mu_bar * np.min(np.array([t(mu) for t in thetas])/np.array([t(mu_bar) for t in thetas]))
    actual_value = theta.evaluate(mu)
    assert expected_value == actual_value


def test_min_theta_parameter_functional_fails_for_wrong_input():
    thetas = (ExpressionParameterFunctional('2*mu', {'mu': ()}),
              ConstantParameterFunctional(1),
              -1)
    mu_bar = -3
    alpha_mu_bar = 10
    with pytest.raises(AssertionError):
        theta = MinThetaParameterFunctional(thetas, mu_bar, alpha_mu_bar)


def test_max_theta_parameter_functional():
    thetas = (ExpressionParameterFunctional('2*mu', {'mu': ()}),
              ConstantParameterFunctional(1),
              -1)
    mu_bar = -3
    gamma_mu_bar = 10
    theta = MaxThetaParameterFunctional(thetas, mu_bar, gamma_mu_bar)
    thetas = [ConstantParameterFunctional(t) if not isinstance(t, ParameterFunctional) else t
              for t in thetas]
    mu = 1
    expected_value = gamma_mu_bar * np.abs(np.max(np.array([t(mu) for t in thetas])/np.array([t(mu_bar) for t in
        thetas])))
    actual_value = theta.evaluate(mu)
    assert expected_value == actual_value


if __name__ == "__main__":
    runmodule(filename=__file__)
