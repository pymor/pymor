# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from pymortests.base import runmodule
from pymor.functions.basic import ConstantFunction, GenericFunction


def test_lincomb_function():
    for steps in (1, 10):
        x = np.linspace(0, 1, num=steps)
        zero = ConstantFunction(0.0, dim_domain=steps)
        for zero in (ConstantFunction(0.0, dim_domain=steps),
                     GenericFunction(lambda X: np.zeros(X.shape[:-1]), dim_domain=steps)):
            for one in (ConstantFunction(1.0, dim_domain=steps),
                        GenericFunction(lambda X: np.ones(X.shape[:-1]), dim_domain=steps), 1.0):
                add = (zero + one) + 0
                sub = (zero - one) + np.zeros(())
                neg = - zero
                assert np.allclose(sub(x), [-1])
                assert np.allclose(add(x), [1.0])
                assert np.allclose(neg(x), [0.0])
                (repr(add), str(add), repr(one), str(one))  # just to cover the respective special funcs too
                mul = neg * 1.
                assert np.allclose(mul(x), [0.0])
        with pytest.raises(AssertionError):
            zero + ConstantFunction(dim_domain=steps + 1)
        with pytest.raises(AssertionError):
            zero * ConstantFunction(dim_domain=steps)
    with pytest.raises(AssertionError):
        ConstantFunction(dim_domain=0)


if __name__ == "__main__":
    runmodule(filename=__file__)
