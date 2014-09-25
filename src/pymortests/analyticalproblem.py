# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function


from pymortests.fixtures.analyticalproblem import analytical_problem
from pymortests.pickle import assert_picklable


def test_pickle(analytical_problem):
    assert_picklable(analytical_problem)
