# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import functools
import types
import inspect
import warnings


class Deprecated:
    """Decorator for marking functions as deprecated.

    It will result in a warning being emitted when the function is used.
    """

    def __init__(self, alt='no alternative given'):
        if hasattr(alt, '__qualname__'):
            alt = f'{alt.__module__}.{alt.__qualname__}'
        self._alt = alt

    def __call__(self, func):
        func.decorated = self

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            func_name = f'{func.__module__}.{func.__qualname__}'
            frame = inspect.currentframe().f_back
            msg = f'DeprecationWarning. Call to deprecated function {func_name}  in ' \
                  f'{frame.f_code.co_filename}:{frame.f_code.co_firstlineno}\n' \
                  f'Use {self._alt} instead'
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return new_func

    def __get__(self, obj, ownerClass=None):
        """Return a wrapper that binds self as a method of obj (!)"""
        self.obj = obj
        return types.MethodType(self, obj)
