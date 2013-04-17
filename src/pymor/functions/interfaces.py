# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import pymor.core as core
from pymor.tools import Named
from pymor.parameters import Parametric


class FunctionInterface(core.BasicInterface, Parametric, Named):
    '''Interface for parameter dependent analytical functions.

    Every function is a map of the form ::

       f(μ): Ω ⊆ R^d -> R^r

    While the function can raise an error if it is evaluated for
    an argument not in Ω, the exact behavior is currently undefined.

    Attributes
    ----------
    dim_domain
        The dimension d.
    dim_range
        The dimension r.
    '''

    dim_domain = 0
    dim_range = 0

    def __init__(self):
        Parametric.__init__(self)

    @core.interfaces.abstractmethod
    def evaluate(self, x, mu={}):
        pass

    def __call__(self, x, mu={}):
        return self.evaluate(x, mu)
