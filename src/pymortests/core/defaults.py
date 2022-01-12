# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import pytest

from pymor.core.defaults import defaults, set_defaults, print_defaults, load_defaults_from_file, write_defaults_to_file
from pymor.tools.io import safe_temporary_filename


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


def test_print_defaults():
    print_defaults()


def test_write_defaults_to_file():
    with safe_temporary_filename('defaults.py') as filename:
        write_defaults_to_file(filename)


def test_load_defaults_from_file():
    with safe_temporary_filename('defaults.py') as filename:
        write_defaults_to_file(filename)
        load_defaults_from_file(filename)
