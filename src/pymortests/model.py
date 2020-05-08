# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.core.pickle import dumps, loads
from pymortests.fixtures.model import model, picklable_model
from pymortests.base import runmodule
from pymortests.pickling import assert_picklable, assert_picklable_without_dumps_function


def test_pickle(model):
    assert_picklable(model)


def test_pickle_without_dumps_function(picklable_model):
    assert_picklable_without_dumps_function(picklable_model)


def test_pickle_by_solving(model):
    m = model
    m2 = loads(dumps(m))
    m.disable_caching()
    m2.disable_caching()
    for mu in m.parameters.space(1, 2).sample_randomly(3, seed=234):
        assert np.all(almost_equal(m.solve(mu), m2.solve(mu)))


if __name__ == "__main__":
    runmodule(filename=__file__)
