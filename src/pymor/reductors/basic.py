# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from pymor.la import NumpyVectorArray


class GenericRBReconstructor(core.BasicInterface):

    def __init__(self, RB):
        self.RB = RB

    def reconstruct(self, U):
        assert isinstance(U, NumpyVectorArray)
        return self.RB.lincomb(U.data)

    def restricted_to_subbasis(self, dim):
        assert dim <= len(self.RB)
        return GenericRBReconstructor(self.RB.copy(ind=range(dim)))


def reduce_generic_rb(discretization, RB, operator_product=None, vector_product=None,
                      disable_caching=True, extends=None):
    '''Generic reduced basis reductor.

    Parameters
    ----------
    discretization
        The discretization which is to be reduced.
    RB
        The reduced basis (i.e. an array of vectors) on which to project.
    operator_product
        Scalar product for the projection of operators. (See
        `operators.constructions.ProjectedOperator`)
    vector_product
        Scalar product for the projection of vector_operators
        (A typical case would be the `initial_data` operator holding the
        initial data of a Cauchy problem.)
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

    projected_operators = {k: op.projected(source_basis=RB, range_basis=RB, product=operator_product) if op else None
                           for k, op in discretization.operators.iteritems()}
    projected_functionals = {k: f.projected(source_basis=RB, range_basis=None, product=operator_product) if f else None
                             for k, f in discretization.functionals.iteritems()}
    projected_vector_operators = {k: (op.projected(source_basis=None, range_basis=RB, product=vector_product) if op
                                      else None)
                                  for k, op in discretization.vector_operators.iteritems()}

    if discretization.products is not None:
        projected_products = {k: p.projected(source_basis=RB, range_basis=RB) for k, p in discretization.products.iteritems()}
    else:
        projected_products = None

    caching = None if disable_caching else discretization.caching

    rd = discretization.with_(operators=projected_operators, functionals=projected_functionals,
                              vector_operators=projected_vector_operators,
                              products=projected_products, visualizer=None, estimator=None,
                              caching=caching, name=discretization.name + '_reduced')
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

    projected_operators = {k: op.projected_to_subbasis(dim_source=dim, dim_range=dim) if op is not None else None
                           for k, op in discretization.operators.iteritems()}
    projected_functionals = {k: f.projected_to_subbasis(dim_source=dim, dim_range=None) if f is not None else None
                            for k, f in discretization.functionals.iteritems()}
    projected_vector_operators = {k: op.projected_to_subbasis(dim_source=None, dim_range=dim) if op else None
                                  for k, op in discretization.vector_operators.iteritems()}

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

    rd = discretization.with_(operators=projected_operators, functionals=projected_functionals,
                              vector_operators = projected_vector_operators,
                              products=projected_products, visualizer=None, estimator=estimator,
                              name=discretization.name + '_reduced_to_subbasis')
    rd.disable_logging()

    if reconstructor is not None and hasattr(reconstructor, 'restricted_to_subbasis'):
        rc = reconstructor.restricted_to_subbasis(dim)
    else:
        rc = SubbasisReconstructor(next(discretization.operators.itervalues()).dim_source, dim,
                                   old_recontructor=reconstructor)

    return rd, rc, {}
