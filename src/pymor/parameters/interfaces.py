from __future__ import absolute_import, division, print_function, unicode_literals

import pymor.core as core
from pymor.tools import Named

class ParameterSpaceInterface(core.BasicInterface):

    parameter_type = None

from .base import Parametric

class ParameterFunctionalInterface(core.BasicInterface, Parametric, Named):

    @core.interfaces.abstractmethod
    def evaluate(self, mu={}):
        pass

    def __call__(self, mu={}):
        return self.evaluate(mu)
