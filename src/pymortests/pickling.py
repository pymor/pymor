# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.sparse import issparse
from types import FunctionType, MethodType

from pymor.core.base import BasicObject
from pymor.core.config import config
from pymor.core.pickle import dumps, loads, dumps_function, PicklingError
from pymor.discretizers.builtin.grids.subgrid import SubGrid
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.parameters.base import ParametricObject

is_equal_ignored_attributes = \
    ((SubGrid, {'_uid', '_CacheableObject__cache_region', '_SubGrid__parent_grid'}),
     (NumpyMatrixBasedOperator, {'_uid', '_CacheableObject__cache_region', '_parameters', '_assembled_operator'}),
     (ParametricObject, {'_name', '_uid', '_CacheableObject__cache_region', '_parameters'}),
     (BasicObject, {'_name', '_uid', '_CacheableObject__cache_region'}))

is_equal_dispatch_table = {}

if config.HAVE_DUNEGDT:
    from dune.xt.la import IstlVector

    def _assert_IstlVector_equal(first, second):
        if not isinstance(first, IstlVector):
            return False
        if not isinstance(second, IstlVector):
            return False
        if not len(first) == len(second):
            return False
        assert_is_equal(np.array(first, copy=False), np.array(second, copy=False))
        return True

    is_equal_dispatch_table[IstlVector] = _assert_IstlVector_equal


def func_with_closure_generator():
    x = 42

    def func_with_closure():
        print(x)
    return func_with_closure


cell_type = type(func_with_closure_generator().__closure__[0])


def assert_is_equal(first, second):

    seen = []

    def _assert_is_equal(first, second):

        for x in seen:
            if x is first:
                return

        seen.append(first)

        for c, m in is_equal_dispatch_table.items():
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
            if first.dtype == object:
                assert first.shape == second.shape
                [_assert_is_equal(f, s) for f, s in zip(first.ravel(), second.ravel())]
            else:
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
            for k, u in first.items():
                _assert_is_equal(u, second.get(k))
        elif isinstance(first, FunctionType):
            for k in ['__closure__', '__code__', '__dict__', '__doc__', '__name__',
                      '__qualname__', '__kwdefaults__', '__annotations__']:
                _assert_is_equal(getattr(first, k), getattr(second, k))
        elif isinstance(first, MethodType):
            _assert_is_equal(first.__func__, second.__func__)
            _assert_is_equal(first.__self__, second.__self__)
        elif isinstance(first, cell_type):
            _assert_is_equal(first.cell_contents, second.cell_contents)
        elif not isinstance(first, BasicObject):
            assert first == second
        else:
            assert ((set(first.__dict__.keys()) - ignored_attributes)
                    == (set(second.__dict__.keys()) - ignored_attributes))
            for k, v in first.__dict__.items():
                if k not in ignored_attributes:
                    _assert_is_equal(v, second.__dict__.get(k))

    _assert_is_equal(first, second)


def assert_picklable(o):
    s = dumps(o)
    o2 = loads(s)
    assert_is_equal(o, o2)


def assert_picklable_without_dumps_function(o):

    def dumps_function_raise(function):
        raise PicklingError(f'Cannot pickle function {function}')

    old_code = dumps_function.__code__
    dumps_function.__code__ = dumps_function_raise.__code__
    try:
        s = dumps(o)
        o2 = loads(s)
        assert_is_equal(o, o2)
    finally:
        dumps_function.__code__ = old_code
