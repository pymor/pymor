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


def reduce_generic_rb(discretization, RB, product=None, disable_caching=True):
    rd = discretization.copy()
    if RB is None:
        RB = np.zeros((0, next(rd.operators.itervalues()).dim_source))
    for k, op in rd.operators.iteritems():
        rd.operators[k] = project_operator(op, RB, product=product)
    if disable_caching and isinstance(rd, Cachable):
        Cachable.__init__(rd, config=NO_CACHE_CONFIG)
    rd.name += '_reduced'
    rd.disable_logging = True
    rc = GenericRBReconstructor(RB)
    return rd, rc
