# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymortests.fixtures.analyticalproblem import analytical_problem, picklable_analytical_problem
from pymortests.pickling import assert_picklable, assert_picklable_without_dumps_function


def test_pickle(analytical_problem):
    assert_picklable(analytical_problem)

def test_pickle_without_dumps_function(picklable_analytical_problem):
    assert_picklable_without_dumps_function(picklable_analytical_problem)
