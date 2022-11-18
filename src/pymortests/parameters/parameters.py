# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from hypothesis import given

from pymor.analyticalproblems.functions import ConstantFunction
from pymor.parameters.base import Parameters, Mu
from pymortests.base import runmodule
import pymortests.strategies as pyst


num_samples = 100


@pytest.fixture(scope='module')
def space():
    return Parameters({'diffusionl': 1}).space(0.1, 1)


def test_uniform(space):
    values = space.sample_uniformly(num_samples)
    assert len(values) == num_samples
    for value in values:
        assert space.contains(value)


def test_randomly(space):
    values = space.sample_randomly(num_samples)
    assert len(values) == num_samples
    for value in values:
        assert space.contains(value)


def test_randomly_without_count(space):
    mu = space.sample_randomly()
    assert isinstance(mu, Mu)


def test_parse_parameter():
    parameters = Parameters(b=2, a=1)
    mu_as_list = [1, 2, 3]
    mu_as_parameter_and_back = list(parameters.parse(mu_as_list).to_numpy())
    assert mu_as_list == mu_as_parameter_and_back


@given(pyst.mus)
def test_parse_mu(mu):
    parameters = mu.parameters
    assert parameters.parse(mu) == mu


@given(pyst.mus)
def test_mu_parameters(mu):
    params = mu.parameters
    assert isinstance(params, Parameters)
    assert mu.keys() == params.keys()
    assert params.is_compatible(mu)


@given(pyst.mus)
def test_mu_values(mu):
    assert all(isinstance(v, np.ndarray) for v in mu.values())
    assert all(v.ndim == 1 for v in mu.values())
    assert all(len(v) > 0 for v in mu.values())


@given(pyst.mus)
def test_mu_time_dependent(mu):
    for param in mu:
        func = mu.get_time_dependent_value(param)
        if mu.is_time_dependent(param):
            assert np.all(mu[param] == func(mu.get('t', 0)))
        else:
            assert isinstance(func, ConstantFunction)
            assert np.all(mu[param] == func.value)


@given(pyst.mus)
def test_mu_with_changed_time(mu):
    mu2 = mu.with_(t=42)
    for param in mu:
        if param == 't':
            assert mu2['t'].item() == 42
            continue
        func = mu.get_time_dependent_value(param)
        if mu.is_time_dependent(param):
            assert np.all(mu2[param] == func(42))
        else:
            assert np.all(mu[param] == mu2[param])


@given(pyst.mus)
def test_mu_to_numpy(mu):
    mu_array = mu.to_numpy()
    mu2 = mu.parameters.parse(mu_array)
    assert mu == mu2


def test_mu_t_wrong_value():
    with pytest.raises(Exception):
        Mu(t=ConstantFunction(np.array([3])))
    with pytest.raises(Exception):
        Mu(t=np.array([1, 2]))


if __name__ == "__main__":
    runmodule(filename=__file__)
