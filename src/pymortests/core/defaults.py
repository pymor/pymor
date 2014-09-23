# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import pytest

from pymortests.base import TestInterface, runmodule
from pymor.core.defaults import defaults, set_defaults


@defaults('c', 'd')
def func(a, b, c=2, d=3, e=4):
    return a, b, c, d, e


def test_defaults():
    assert func(0, 1) == (0, 1, 2, 3, 4)
    assert func(0, 1, None, d=None) == (0, 1, 2, 3, 4)
    assert func(0, 1, 5, d=None) == (0, 1, 5, 3, 4)
    with pytest.raises(TypeError):
        assert func(0, c=2, d=3)
    set_defaults({__name__ + '.func.c': 42})
    assert func(0, 1) == (0, 1, 42, 3, 4)
    assert func(0, 1, None, d=None) == (0, 1, 42, 3, 4)
    assert func(0, 1, 5, d=None) == (0, 1, 5, 3, 4)
    set_defaults({__name__ + '.func.c': 43})
    assert func(0, 1) == (0, 1, 43, 3, 4)
    assert func(0, 1, None, d=None) == (0, 1, 43, 3, 4)
    assert func(0, 1, 5, d=None) == (0, 1, 5, 3, 4)
