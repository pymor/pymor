# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.core.pickle import dumps, loads
from pymortests.fixtures.discretization import discretization, picklable_discretization
from pymortests.base import runmodule
from pymortests.pickling import assert_picklable, assert_picklable_without_dumps_function


def test_pickle(discretization):
    assert_picklable(discretization)


def test_pickle_without_dumps_function(picklable_discretization):
    assert_picklable_without_dumps_function(picklable_discretization)


def test_pickle_by_solving(discretization):
    d = discretization
    d2 = loads(dumps(d))
    d.disable_caching()
    d2.disable_caching()
    for mu in d.parameter_space.sample_randomly(3, seed=234):
        assert np.all(almost_equal(d.solve(mu), d2.solve(mu)))


if __name__ == "__main__":
    runmodule(filename=__file__)
