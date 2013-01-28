from __future__ import absolute_import, division, print_function, unicode_literals

import pymor.core as core
from pymor.tools import Named
from pymor.parameters import Parametric

class FunctionInterface(core.BasicInterface, Parametric, Named):

    dim_domain = 0
    dim_range = 0

    @core.interfaces.abstractmethod
    def evaluate(self, x, mu={}):
        pass

    def __call__(self, x, mu={}):
        return self.evaluate(x, mu)
