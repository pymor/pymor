from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.discreteoperators import project_operator


class GenericRBReconstructor(object):

    def __init__(self, RB):
        self.RB = RB

    def reconstruct(self, U):
        return np.dot(U, self.RB)


class GenericRBReductor(object):

    def __init__(self, discretization, product=None):
        self.discretization = discretization
        self.product = product

    def reduce(self, RB):
        rd = self.discretization.copy()
        for k, op in rd.operators.iteritems():
            rd.operators[k] = project_operator(op, RB, product=self.product)
        rc = GenericRBReconstructor(RB)
        return rd, rc
