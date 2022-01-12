# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import warnings

from pymor.analyticalproblems.text import text_problem
from pymortests.base import runmodule
from pymortests.pickling import assert_picklable, assert_picklable_without_dumps_function


def test_pickle(analytical_problem):
    assert_picklable(analytical_problem)


def test_pickle_without_dumps_function(picklable_analytical_problem):
    assert_picklable_without_dumps_function(picklable_analytical_problem)


def test_missing_font():
    name = 'ThisFontIsMissing'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        text_problem(text='pyMOR', font_name=name)
        assert len(w) == 1
        assert issubclass(w[-1].category, ResourceWarning)
        assert name in str(w[-1].message)


if __name__ == "__main__":
    runmodule(filename=__file__)
