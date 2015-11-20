# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import pprint
import pkgutil
import sys
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from math import factorial

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
    for n in xrange(max_order + 1):
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
