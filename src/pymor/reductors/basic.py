# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from pymor.la import NumpyVectorArray
from pymor.core.cache import Cachable, NO_CACHE_CONFIG
from pymor.operators import rb_project_operator


class GenericRBReconstructor(core.BasicInterface):

    def __init__(self, RB):
        self.RB = RB

    def reconstruct(self, U):
        assert isinstance(U, NumpyVectorArray)
        return self.RB.lincomb(U._array)


def reduce_generic_rb(discretization, RB, product=None, disable_caching=True):
    '''Generic reduced basis reductor.

    Reduces a discretization by applying `operators.project_operator` to
    each of its `operators`.

    Parameters
    ----------
    discretization
        The discretization which is to be reduced.
    RB
        The reduced basis (i.e. an array of vectors) on which to project.
    product
        Scalar product for the projection. (See
        `operators.constructions.ProjectedOperator`)
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

    if RB is None:
        RB = NumpyVectorArray(np.zeros((0, next(discretization.operators.itervalues()).dim_source)))

    projected_operators = {k: rb_project_operator(op, RB, product=product)
                           for k, op in discretization.operators.iteritems()}
    rd = discretization.with_projected_operators(projected_operators)

    if disable_caching and isinstance(rd, Cachable):
        Cachable.__init__(rd, config=NO_CACHE_CONFIG)
    rd.name += '_reduced'
    rd.disable_logging = True
    rc = GenericRBReconstructor(RB)
    return rd, rc
