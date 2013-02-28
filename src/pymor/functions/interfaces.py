# -*- coding: utf-8 -*-
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

    Inherits
    --------
    BasicInterface, Parametric, Named
    '''

    dim_domain = 0
    dim_range = 0

    @core.interfaces.abstractmethod
    def evaluate(self, x, mu={}):
        pass

    def __call__(self, x, mu={}):
        return self.evaluate(x, mu)
