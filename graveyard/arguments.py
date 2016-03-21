# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import inspect


def method_arguments(func):
    """Returns the names of the arguments of a given method (without `self`)."""
    args = inspect.getargspec(func)[0]
    try:
        args.remove('self')
    except ValueError:
        pass
    return args
