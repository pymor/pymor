# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from pymor.la import NumpyVectorArray
from pymor.operators import rb_project_operator


class GenericRBReconstructor(core.BasicInterface):

    def __init__(self, RB):
        self.RB = RB

    def reconstruct(self, U):
        assert isinstance(U, NumpyVectorArray)
        return self.RB.lincomb(U.data)

    def restricted_to_subbasis(self, dim):
        assert dim <= len(self.RB)
        return GenericRBReconstructor(self.RB.copy(ind=range(dim)))


def reduce_generic_rb(discretization, RB, product=None, disable_caching=True,
                      extends=None):
    '''Generic reduced basis reductor.

    Reduces a discretization by applying `operators.rb_project_operator` to
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
    assert extends is None or len(extends) == 3

    if RB is None:
        RB = discretization.type_solution.empty(discretization.dim_solution)

    projected_operators = {k: rb_project_operator(op, RB, product=product)
                           for k, op in discretization.operators.iteritems()}

    if discretization.products is not None:
        projected_products = {k: rb_project_operator(op, RB, product=product)
                              for k, op in discretization.products.iteritems()}
    else:
        projected_products = None

    caching = None if disable_caching else discretization.caching

    rd = discretization.with_(operators=projected_operators, products=projected_products, visualizer=None,
                              estimator=None, caching=caching, name=discretization.name + '_reduced')
    rd.disable_logging()
    rc = GenericRBReconstructor(RB)

    return rd, rc, {}


class SubbasisReconstructor(core.BasicInterface):

    def __init__(self, dim, dim_subbasis, old_recontructor=None):
        self.dim = dim
        self.dim_subbasis = dim_subbasis
        self.old_recontructor = old_recontructor

    def reconstruct(self, U):
        assert isinstance(U, NumpyVectorArray)
        UU = np.zeros((len(U), self.dim))
        UU[:, :self.dim_subbasis] = U.data
        UU = NumpyVectorArray(UU, copy=False)
        if self.old_recontructor:
            return self.old_recontructor.reconstruct(UU)
        else:
            return UU


def reduce_to_subbasis(discretization, dim, reconstructor=None):

    dim_solution = discretization.dim_solution

    projected_operators = {k: op.projected_to_subbasis(dim_source=dim if op.dim_source == dim_solution else None,
                                                       dim_range=dim if op.dim_range == dim_solution else None)
                               if op is not None else None
                           for k, op in discretization.operators.iteritems()}

    if discretization.products is not None:
        projected_products = {k: op.projected_to_subbasis(dim_source=dim, dim_range=dim)
                              for k, op in discretization.products.iteritems()}
    else:
        projected_products = None

    if hasattr(discretization, 'estimator') and hasattr(discretization.estimator, 'restricted_to_subbasis'):
        estimator = discretization.estimator.restricted_to_subbasis(dim, discretization=discretization)
    elif hasattr(discretization, 'estimate'):
        class FakeEstimator(object):
            rd = discretization
            rc = SubbasisReconstructor(next(discretization.operators.itervalues()).dim_source, dim)
            def estimate(self, U, mu=None, discretization=None):
                return self.rd.estimate(self.rc.reconstruct(U), mu=mu)
        estimator = FakeEstimator()
    else:
        estimator = None

    rd = discretization.with_(operators=projected_operators, products=projected_products, visualizer=None,
                              estimator=estimator, name=discretization.name + '_reduced_to_subbasis')
    rd.disable_logging()

    if reconstructor is not None and hasattr(reconstructor, 'restricted_to_subbasis'):
        rc = reconstructor.restricted_to_subbasis(dim)
    else:
        rc = SubbasisReconstructor(next(discretization.operators.itervalues()).dim_source, dim,
                                   old_recontructor=reconstructor)

    return rd, rc, {}
