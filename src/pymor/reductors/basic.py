# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

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
    '''Generic reduced basis reductor.

    Reduces a discretization by applying `discreteoperators.project_operator` to
    each of its `operators`.

    Parameters
    ----------
    discretization
        The discretization which is to be reduced.
    RB
        The reduced basis (i.e. an array of vectors) on which to project.
    product
        Scalar product for the projection. (See
        `discreteoperators.constructions.ProjectedOperator`)
    disable_caching
        If `True`, caching of the solutions of the reduced discretization
        is disabled.

    Returns
    -------
    rd
        The reduced discretization.
    rc
        The reconstructor providing a `reconstruct(U)` method which reconstructs
        high-dimensional solutions from solutions U of the reduced discretization.
    '''

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
