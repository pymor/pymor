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

try:
    import contracts
    HAVE_CONTRACTS = True
except ImportError:
    HAVE_CONTRACTS = False


def fixup_docstring(doc):
    """replaces all dots with underscores in contract lines
    this is necessary to circumvent type identifier checking
    in pycontracts itself
    """
    if doc is None:
        return None
    ret = []
    for line in doc.split('\n'):
        stripped_line = line.lstrip()
        if stripped_line.startswith(':type'):
            # line's like: :type ParamName: Some.Module.Classname
            tokens = stripped_line.split()
            idx = 2
            if len(tokens) > idx and tokens[idx].startswith('pymor'):
                line = ' %s %s %s' % (tokens[idx - 2], tokens[idx - 1], tokens[idx].replace('.', '_'))
        ret.append(line)
    return '\n'.join(ret)


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


def contract(*arg, **kwargs):
    """
        Decorator for adding contracts to functions.

        It is smart enough to support functions with variable number of
        arguments and keyword arguments.

        There are three ways to specify the contracts. In order of precedence:

        - As arguments to this decorator. For example: ::

              @contract(a='int,>0',b='list[N],N>0',returns='list[N]')
              def my_function(a, b):
                  # ...
                  pass

        - As annotations (supported only in Python 3): ::

              @contract
              def my_function(a:'int,>0', b:'list[N],N>0') -> 'list[N]':
                  # ...
                  pass

        - Using ``:type:`` and ``:rtype:`` tags in the function's docstring: ::

              @contract
              def my_function(a, b):
                  """

    if not HAVE_CONTRACTS:
        if isinstance(arg[0], types.FunctionType):
            return arg[0]
        else:
            return lambda f: f

    # this bit tags function as decorated
    def tag_and_decorate(function, **kwargs):
        @functools.wraps(function)
        def __functools_wrap(function, **kwargs):
            new_f = contracts.main.contracts_decorate(function, **kwargs)
            cargs = copy.deepcopy(kwargs)
            new_f.contract_kwargs = cargs or dict()
            new_f.decorated = 'contract'
            return new_f
        return __functools_wrap(function, **kwargs)

    # OK, this is black magic. You are not expected to understand this.
    if arg:
        if isinstance(arg[0], types.FunctionType):
            # We were called without parameters
            function = arg[0]
            function.__doc__ = fixup_docstring(function.__doc__)
            if contracts.all_disabled():
                return function
            try:
                return tag_and_decorate(function, **kwargs)
            except contracts.ContractSyntaxError as e:
                # Erase the stack
                raise contracts.ContractSyntaxError(e.error, e.where)
        else:
            msg = ('I expect that  contracts() is called with '
                   'only keyword arguments (passed: %r)' % arg)
            raise contracts.ContractException(msg)
    else:
        function.__doc__ = fixup_docstring(function.__doc__)
        # We were called *with* parameters.
        if contracts.all_disabled():
            def tmp_wrap(function):
                return function
        else:
            def tmp_wrap(function):
                try:
                    return tag_and_decorate(function, **kwargs)
                except contracts.ContractSyntaxError as e:
                    # Erase the stack
                    raise contracts.ContractSyntaxError(e.error, e.where)
        return tmp_wrap

# alias this so we need no contracts import outside this module
if HAVE_CONTRACTS:
    contracts_decorate = contracts.main.contracts_decorate


def contains_contract(string):
    if not HAVE_CONTRACTS:
        return False
    try:
        contracts.parse_contract_string(string)
        return True
    except:
        return False
