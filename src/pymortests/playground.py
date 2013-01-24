from __future__ import absolute_import, division, print_function

import numpy as np
import random
from math import pow, factorial, sin, pi, exp

from pymortests.base import TestBase, runmodule
from pymor.playground.expression_function import ExpressionFunction

FUNCTIONS = [(ExpressionFunction(['x**2'], 'x'), lambda x: np.array([x**2])),
             (ExpressionFunction(['x**2', 'sin(x)'], 'x'), lambda x: np.array([x**2,sin(x)])),
             (ExpressionFunction(['exp(xp)'], 'xp'), lambda x: np.array([exp(x)]))]

class TestExpressionFunction(TestBase):
    
    def test_eval(self):
        for (fn, fe) in FUNCTIONS:
            for x in (random.uniform(0, 1) for _ in xrange(9000)):
                np.testing.assert_array_almost_equal(fn([x]), fe(x))
        

if __name__ == "__main__":
    random.seed()
    runmodule(name='pymortests.playground')