# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import functools
import inspect
import itertools

from pymor.core.interfaces import BasicInterface, ImmutableMeta, _calculate_sid


class generate_sid(object):

    def __init__(self, func=None, ignore=None):
        if func is not None and not hasattr(func, '__call__'):
            assert ignore is None
            ignore = func
            func = None
        if isinstance(ignore, str):
            ignore = (ignore,)
        self.ignore = ignore if ignore is not None else tuple()
        self.set_func(func)

    def set_func(self, func):
        self.func = func
        if func is not None:
            # Beware! The following will probably break in python 3 if there are
            # keyword-only arguments
            args, varargs, keywords, defaults  = inspect.getargspec(func)
            if varargs:
                raise NotImplementedError
            assert args[0] == 'self'
            self.func_arguments = tuple(args[1:])
            functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        if self.func is None:
            assert len(kwargs) == 0 and len(args) == 1
            self.set_func(args[0])
            return self
        else:
            r = self.func(*args, **kwargs)
            assert isinstance(r, BasicInterface)

            if not hasattr(r, 'sid'):
                r.unlock()
                try:
                    kwargs.update((k, o) for k, o in itertools.izip(self.func_arguments, args[1:]))
                    kwarg_sids = tuple((k, _calculate_sid(o, k))
                                       for k, o in sorted(kwargs.iteritems())
                                       if k not in self.ignore)
                    r.sid = (type(r), args[0].sid, self.__name__,  kwarg_sids)
                    ImmutableMeta.sids_created += 1
                except (ValueError, AttributeError) as e:
                    r.sid_failure = str(e)
                r.lock()

            return r

    def __get__(self, obj, obj_type):
        return functools.partial(self.__call__, obj)
