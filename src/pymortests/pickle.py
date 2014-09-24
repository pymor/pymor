# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import BasicInterface
from pymor.core.pickle import dumps, loads
from pymor.grids.subgrid import SubGrid


is_equal_ignored_attributes = \
    {BasicInterface: {'_uid', '_cache_region'},
     SubGrid: {'_uid', '_cache_region', '_SubGrid__parent_grid'}}

is_equal_dispatch_table = {}


function_type = type(lambda:())


def assert_is_equal(first, second):

    for c, m in is_equal_dispatch_table.iteritems():
        if type(first) == c:
            assert m(first, second)
            return

    ignored_attributes = set()
    for c, v in is_equal_ignored_attributes.iteritems():
        if type(first) == c:
            ignored_attributes = v
            break

    assert type(first) == type(second)

    if isinstance(first, np.ndarray):
        assert np.all(first == second)
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for u, v in zip(first, second):
            assert_is_equal(u, v)
    elif isinstance(first, dict):
        assert set(first.keys()) == set(second.keys())
        for k, u in first.iteritems():
            assert_is_equal(u, second.get(k))
    elif isinstance(first, function_type):
        for k in ['func_closure', 'func_code', 'func_code', 'func_dict', 'func_doc', 'func_name']:
            assert getattr(first, k) == getattr(second, k)
    elif not isinstance(first, BasicInterface):
        assert first == second
    else:
        assert (set(first.__dict__.keys()) - ignored_attributes) == (set(second.__dict__.keys()) - ignored_attributes)
        for k, v in first.__dict__.iteritems():
            if not k in ignored_attributes:
                assert_is_equal(v, second.__dict__.get(k))


def assert_picklable(o):
    s = dumps(o)
    o2 = loads(s)
    assert_is_equal(o, o2)
