from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from pymor.core.cache import Cachable, NO_CACHE_CONFIG
from pymor.discreteoperators import project_operator


class GenericRBReconstructor(core.BasicInterface):

    def __init__(self, RB):
        self.RB = RB

    def reconstruct(self, U):
        return np.dot(U, self.RB)


class GenericRBReductor(core.BasicInterface):

    def __init__(self, discretization, product=None, disable_caching=True):
        self.discretization = discretization
        self.product = product
        self.disable_caching = disable_caching

    def reduce(self, RB):
        rd = self.discretization.copy()
        for k, op in rd.operators.iteritems():
            rd.operators[k] = project_operator(op, RB, product=self.product)
        if self.disable_caching and isinstance(rd, Cachable):
            Cachable.__init__(rd, config=NO_CACHE_CONFIG)
        rd.name += '_reduced'
        rd.disable_logging = True
        rc = GenericRBReconstructor(RB)
        return rd, rc
