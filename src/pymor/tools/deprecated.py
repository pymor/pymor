# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import functools
import types
import inspect
import warnings


class Deprecated:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    def __init__(self, alt='no alternative given'):
        self._alt = alt

    def __call__(self, func):
        func.decorated = self

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            frame = inspect.currentframe().f_back
            msg = 'DeprecationWarning. Call to deprecated function {} in {}:{}\nUse {} instead'.format(
                func.__name__, frame.f_code.co_filename, frame.f_code.co_firstlineno, self._alt)
            warnings.warn(msg, DeprecationWarning)
            return func(*args, **kwargs)
        return new_func

    def __get__(self, obj, ownerClass=None):
        """Return a wrapper that binds self as a method of obj (!)"""
        self.obj = obj
        return types.MethodType(self, obj)
