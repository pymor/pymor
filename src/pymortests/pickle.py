# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import issparse
from types import FunctionType, MethodType

from pymor.core.interfaces import BasicInterface
from pymor.core.pickle import dumps, loads, dumps_function, PicklingError
from pymor.grids.subgrid import SubGrid
from pymor.operators.numpy import NumpyMatrixBasedOperator


is_equal_ignored_attributes = \
    ((SubGrid, {'_uid', '_CacheableInterface__cache_region', '_SubGrid__parent_grid'}),
     (NumpyMatrixBasedOperator, {'_uid', '_CacheableInterface__cache_region', '_assembled_operator'}),
     (BasicInterface, {'_name', '_uid', '_CacheableInterface__cache_region'}))

is_equal_dispatch_table = {}


def func_with_closure_generator():
    x = 42
    def func_with_closure():
        print(x)
    return func_with_closure

cell_type = type(func_with_closure_generator().func_closure[0])


def assert_is_equal(first, second):

    seen = []

    def _assert_is_equal(first, second):

        for x in seen:
            if x is first:
                return

        seen.append(first)

        for c, m in is_equal_dispatch_table.iteritems():
            if type(first) == c:
                assert m(first, second)
                return

        ignored_attributes = set()
        for c, v in is_equal_ignored_attributes:
            if isinstance(first, c):
                ignored_attributes = v
                break

        assert type(first) == type(second)

        if isinstance(first, np.ndarray):
            assert np.all(first == second)
        elif issparse(first):
            ne = first != second
            if isinstance(ne, bool):
                return not ne
            else:
                assert not np.any(ne.data)
        elif isinstance(first, (list, tuple)):
            assert len(first) == len(second)
            for u, v in zip(first, second):
                _assert_is_equal(u, v)
        elif isinstance(first, dict):
            assert set(first.keys()) == set(second.keys())
            for k, u in first.iteritems():
                _assert_is_equal(u, second.get(k))
        elif isinstance(first, FunctionType):
            for k in ['func_closure', 'func_code', 'func_dict', 'func_doc', 'func_name']:
                _assert_is_equal(getattr(first, k), getattr(second, k))
        elif isinstance(first, MethodType):
            _assert_is_equal(first.im_func, second.im_func)
            _assert_is_equal(first.im_self, second.im_self)
        elif isinstance(first, cell_type):
            _assert_is_equal(first.cell_contents, second.cell_contents)
        elif not isinstance(first, BasicInterface):
            assert first == second
        else:
            assert (set(first.__dict__.keys()) - ignored_attributes) == (set(second.__dict__.keys()) - ignored_attributes)
            for k, v in first.__dict__.iteritems():
                if not k in ignored_attributes:
                    _assert_is_equal(v, second.__dict__.get(k))

    _assert_is_equal(first, second)


def assert_picklable(o):
    s = dumps(o)
    o2 = loads(s)
    assert_is_equal(o, o2)


def assert_picklable_without_dumps_function(o):

    def dumps_function_raise(function):
        raise PicklingError('Cannot pickle function {}'.format(function))

    old_code = dumps_function.func_code
    dumps_function.func_code = dumps_function_raise.func_code
    try:
        s = dumps(o)
        o2 = loads(s)
        assert_is_equal(o, o2)
    finally:
        dumps_function.func_code = old_code
