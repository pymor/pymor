from __future__ import absolute_import, division, print_function

import pymor.core as core
from pymor.tools import Named
from pymor.parameters.base import Parametric


class ParameterSpaceInterface(core.BasicInterface):

    parameter_type = None


class ParameterFunctionalInterface(core.BasicInterface, Parametric, Named):

    @core.interfaces.abstractmethod
    def evaluate(self, mu={}):
        pass

    def __call__(self, mu={}):
        return self.evaluate(mu)
