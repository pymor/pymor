# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import hashlib
import pprint
import pkgutil
import os
import sys
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from math import factorial
from pickle import dumps, dump, load
from pkg_resources import resource_filename, resource_stream

from pymor.core import logger
from pymor.operators.basic import OperatorBase
from pymor.vectorarrays.numpy import NumpyVectorSpace


class TestInterface(object):
    logger = logger.getLogger(__name__)


TestInterface = TestInterface


def _load_all():
    import pymor

    ignore_playground = True
    fails = []
    for _, module_name, _ in pkgutil.walk_packages(pymor.__path__, pymor.__name__ + '.',
                                                   lambda n: fails.append((n, ''))):
        if ignore_playground and 'playground' in module_name:
            continue
        try:
            __import__(module_name, level=0)
        except (TypeError, ImportError) as t:
            fails.append((module_name, t))
    if len(fails) > 0:
        logger.getLogger(__name__).fatal('Failed imports: {}'.format(pprint.pformat(fails)))
        raise ImportError(__name__)


def subclassForImplemetorsOf(InterfaceType, TestCase):
    """dynamically creates subclasses of the decorated base test class
    for all implementors of a given Interface
    """
    try:
        _load_all()
    except ImportError:
        pass


    test_types = set(sorted([T for T in InterfaceType.implementors(True) if not (T.has_interface_name() or issubclass(T, TestInterface))], key=lambda g: g.__name__))
    for Type in test_types:
        cname = 'DynamicTest_{}_{}'.format(Type.__name__, TestCase.__name__.replace('Interface', ''))
        yield type(cname, (TestCase,), {'Type': Type})


def runmodule(filename):
    import pytest

    sys.exit(pytest.main(sys.argv[1:] + [filename]))


def polynomials(max_order):
    for n in range(max_order + 1):
        def f(x):
            return np.power(x, n)

        def deri(k):
            if k > n:
                return lambda _: 0
            return lambda x: (factorial(n) / factorial(n - k)) * np.power(x, n - k)

        integral = (1 / (n + 1))
        yield (n, f, deri, integral)


class MonomOperator(OperatorBase):
    source = range = NumpyVectorSpace(1)

    def __init__(self, order, monom=None):
        self.monom = monom if monom else Polynomial(np.identity(order + 1)[order])
        assert isinstance(self.monom, Polynomial)
        self.order = order
        self.derivative = self.monom.deriv()
        self.linear = order == 1

    def apply(self, U, mu=None):
        return self.source.make_array(self.monom(U.to_numpy()))

    def jacobian(self, U, mu=None):
        return MonomOperator(self.order - 1, self.derivative)

    def apply_inverse(self, V, mu=None, least_squares=False):
        return self.range.make_array(1. / V.to_numpy())


def check_results(test_name, params, results, *args):
    params = str(params)
    tols = (1e-13, 1e-13)
    keys = {}
    for arg in args:
        if isinstance(arg, tuple):
            assert len(arg) == 2
            tols = arg
        else:
            keys[arg] = tols

    assert results is not None
    assert set(keys.keys()) <= set(results.keys()), \
        'Keys {} missing in results dict'.format(set(keys.keys()) - set(results.keys()))
    results = {k: np.asarray(results[k]) for k in keys.keys()}
    assert all(v.dtype != object for v in results.values())

    basepath = resource_filename('pymortests', 'testdata/check_results')
    arg_id = hashlib.sha1(params.encode()).hexdigest()
    filename = resource_filename('pymortests', 'testdata/check_results/{}/{}'.format(test_name, arg_id))
    testname_dir = os.path.join(basepath, test_name)

    def _dump_results(fn, res):
        with open(fn, 'wb') as f:
            f.write((params + '\n').encode())
            res = {k: v.tolist() for k, v in res.items()}
            dump(res, f, protocol=2)

    try:
        with resource_stream('pymortests', 'testdata/check_results/{}/{}'.format(test_name, arg_id)) as f:
            f.readline()
            old_results = load(f)
    except FileNotFoundError:
        if not os.path.exists(testname_dir):
            os.mkdir(testname_dir)
        _dump_results(filename, results)
        assert False, \
            'No results found for test {} ({}), saved current results. Remember to check in {}.'.format(
                test_name, params, filename)

    for k, (atol, rtol) in keys.items():
        if not np.all(np.allclose(old_results[k], results[k], atol=atol, rtol=rtol)):
            abs_errs = np.abs(results[k] - old_results[k])
            rel_errs = abs_errs / np.abs(old_results[k])
            _dump_results(filename + '_changed', results)
            assert False, 'Results for test {}({}, key: {}) have changed.\n (maximum error: {} abs / {} rel).\nSaved new results in {}'.format(
                test_name, params, k, np.max(abs_errs), np.max(rel_errs), filename + '_changed')
