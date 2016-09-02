# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
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

from pymor.core import logger
from pymor.operators.basic import OperatorBase
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace


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


def SubclassForImplemetorsOf(InterfaceType):
    """A decorator that dynamically creates subclasses of the decorated base test class
    for all implementors of a given Interface
    """
    try:
        _load_all()
    except ImportError:
        pass

    def decorate(TestCase):
        """saves a new type called cname with correct bases and class dict in globals"""
        import pymor.core.dynamic

        test_types = set([T for T in InterfaceType.implementors(True) if not (T.has_interface_name()
                                                                              or issubclass(T, TestInterface))])
        for Type in test_types:
            cname = 'Test_{}_{}'.format(Type.__name__, TestCase.__name__.replace('Interface', ''))
            pymor.core.dynamic.__dict__[cname] = type(cname, (TestCase,), {'Type': Type})
        return TestCase

    return decorate


def runmodule(filename):
    import pytest

    sys.exit(pytest.main(sys.argv[1:] + [filename]))


def polynomials(max_order):
    for n in range(max_order + 1):
        f = lambda x: np.power(x, n)

        def deri(k):
            if k > n:
                return lambda _: 0
            return lambda x: (factorial(n) / factorial(n - k)) * np.power(x, n - k)

        integral = (1 / (n + 1))
        yield (n, f, deri, integral)


class MonomOperator(OperatorBase):
    source = range = NumpyVectorSpace(1)
    type_source = type_range = NumpyVectorArray

    def __init__(self, order, monom=None):
        self.monom = monom if monom else Polynomial(np.identity(order + 1)[order])
        assert isinstance(self.monom, Polynomial)
        self.order = order
        self.derivative = self.monom.deriv()
        self.linear = order == 1

    def apply(self, U, ind=None, mu=None):
        return NumpyVectorArray(self.monom(U.data))

    def jacobian(self, U, mu=None):
        return MonomOperator(self.order - 1, self.derivative)

    def apply_inverse(self, V, ind=None, mu=None, least_squares=False):
        return NumpyVectorArray(1. / V.data)


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

    basepath = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'testdata', 'check_results')
    arg_id = hashlib.sha1(params.encode()).hexdigest()
    filename = os.path.normpath(os.path.join(basepath, test_name, arg_id))

    if not os.path.exists(os.path.join(basepath, test_name)):
        os.mkdir(os.path.join(basepath, test_name))
    if not os.path.exists(filename):
        with open(filename, 'wb') as f:
            f.write((params + '\n').encode())
            results = {k: v.tolist() for k, v in results.items()}
            dump(results, f, protocol=2)
        assert False, \
            'No results found for test {} ({}), saved current results. Remember to check in {}.'.format(
                test_name, params, filename)

    with open(filename, 'rb') as f:
        f.readline()
        old_results = load(f)

    for k, (atol, rtol) in keys.items():
        if not np.all(np.allclose(old_results[k], results[k], atol=atol, rtol=rtol)):
            abs_errs = np.abs(results[k] - old_results[k])
            rel_errs = abs_errs / np.abs(old_results[k])
            with open(filename + '_changed', 'wb') as f:
                f.write((params + '\n').encode())
                dump(results, f, protocol=2)
            assert False, 'Results for test {}({}, key: {}) have changed.\n (maximum error: {} abs / {} rel).\nSaved new results in {}'.format(
                test_name, params, k, np.max(abs_errs), np.max(rel_errs), filename + '_changed')
