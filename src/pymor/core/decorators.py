# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

"""
Created on Fri Nov  2 10:12:55 2012
Collection of function/class based decorators.

"""
from __future__ import absolute_import, division, print_function
import functools
import types
import inspect
import copy
import warnings


def fixup_docstring(doc):
    return doc


def _is_decorated(func):
    return 'decorated' in dir(func)


class DecoratorBase(object):
    """A base for all decorators that does the common automagic"""
    def __init__(self, func):
        functools.wraps(func)(self)
        func.decorated = self
        self.func = func
        assert _is_decorated(func)

    def __get__(self, obj, ownerClass=None):
        """Return a wrapper that binds self as a method of obj (!)"""
        self.obj = obj
        return types.MethodType(self, obj)


class DecoratorWithArgsBase(object):
    """A base for all decorators with args that sadly can do little common automagic"""
    def mark(self, func):
        functools.wraps(func)
        func.decorated = self

    def __get__(self, obj, ownerClass=None):
        # Return a wrapper that binds self as a method of obj (!)
        self.obj = obj
        return types.MethodType(self, obj)


class Deprecated(DecoratorBase):
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

