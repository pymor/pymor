# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.parameters.spaces import CubicParameterSpace
from pymortests.base import runmodule

import pytest


num_samples = 100


@pytest.fixture(scope='module')
def space():
    return CubicParameterSpace({'diffusionl': 1}, 0.1, 1)


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


if __name__ == "__main__":
    runmodule(filename=__file__)
