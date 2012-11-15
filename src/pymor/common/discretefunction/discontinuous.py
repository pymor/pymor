#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from pymor import core
from pymor.common import function


class Interface(core.BaseInterface):

    def name(self):
        return 'common.discretefunction.discontinuous'


class Legendre(Interface):

    def __init__(self, order=0, f=function.nonparametric.Constant(0)):
        self._name = 'common.discretefunction.discontinuous.legendre'
        self._order = order

    def order(self):
        return self._order


if __name__ == '__main__':
    f = function.nonparametric.Constant(1)
    df = Legendre(0, f)