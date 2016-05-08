# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import random
from math import sin, exp

from pymortests.base import runmodule


@pytest.mark.xfail
def test_eval():
    from pymor.playground.expression_function import ExpressionFunction

    FUNCTIONS = [(ExpressionFunction(['x**2'], 'x'), lambda x: np.array([x ** 2])),
                 (ExpressionFunction(['x**2', 'sin(x)'], 'x'), lambda x: np.array([x ** 2, sin(x)])),
                 (ExpressionFunction(['exp(xp)'], 'xp'), lambda x: np.array([exp(x)]))]
    for (fn, fe) in FUNCTIONS:
        for x in (random.uniform(0, 1) for _ in range(9000)):
            np.testing.assert_array_almost_equal(fn([x]), fe(x))

if __name__ == "__main__":
    random.seed()
    runmodule(filename=__file__)
