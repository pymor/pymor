# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import BasicInterface
from pymor.vectorarrays.numpy import NumpyVectorArray


class GenericRBReconstructor(BasicInterface):
    """Simple reconstructor forming linear combinations with a reduced basis."""

    def __init__(self, RB):
        self.RB = RB.copy()

    def reconstruct(self, U):
        """Reconstruct high-dimensional vector from reduced vector `U`."""
        assert isinstance(U, NumpyVectorArray)
        return self.RB.lincomb(U.data)

    def restricted_to_subbasis(self, dim):
        """See :meth:`~pymor.operators.numpy.NumpyMatrixOperator.projected_to_subbasis`."""
        assert dim <= len(self.RB)
        return GenericRBReconstructor(self.RB.copy(ind=list(range(dim))))


def reduce_generic_rb(discretization, RB, vector_product=None, disable_caching=True, extends=None):
    """Generic reduced basis reductor.

    Replaces each |Operator| of the given |Discretization| with the Galerkin
    projection onto the span of the given reduced basis.

    Parameters
    ----------
    discretization
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    vector_product
        Inner product for the projection of vector-like |Operators|.
        (A typical vector-like operator would be the `initial_data`
        |Operator| of an |InstationaryDiscretization| holding
        the initial data of a Cauchy problem.)
    disable_caching
        If `True`, caching of solutions is disabled for the reduced |Discretization|.
    extends
        Set by :meth:`~pymor.algorithms.greedy.greedy` to the result of the
        last reduction in case the basis extension was `hierarchic` (ignored).

    Returns
    -------
    rd
        The reduced |Discretization|.
    rc
        The :class:`reconstructor <GenericRBReconstructor>` providing a
        `reconstruct(U)` method which reconstructs high-dimensional solutions
        from solutions `U` of the reduced |Discretization|.
    reduction_data
        Additional data produced by the reduction process (empty).
    """
    assert extends is None or len(extends) == 3

    if RB is None:
        RB = discretization.solution_space.empty()

    projected_operators = {k: op.projected(range_basis=RB, source_basis=RB, product=None) if op else None
                           for k, op in discretization.operators.items()}
    projected_functionals = {k: f.projected(range_basis=None, source_basis=RB, product=None) if f else None
                             for k, f in discretization.functionals.items()}
    projected_vector_operators = {k: (op.projected(range_basis=RB, source_basis=None, product=vector_product) if op
                                      else None)
                                  for k, op in discretization.vector_operators.items()}

    if discretization.products is not None:
        projected_products = {k: p.projected(range_basis=RB, source_basis=RB)
                              for k, p in discretization.products.items()}
    else:
        projected_products = None

    cache_region = None if disable_caching else discretization.caching

    rd = discretization.with_(operators=projected_operators, functionals=projected_functionals,
                              vector_operators=projected_vector_operators,
                              products=projected_products, visualizer=None, estimator=None,
                              cache_region=cache_region, name=discretization.name + '_reduced')
    rd.disable_logging()
    rc = GenericRBReconstructor(RB)

    return rd, rc, {}


class SubbasisReconstructor(BasicInterface):
    """Returned by :meth:`reduce_to_subbasis`."""

    def __init__(self, dim, dim_subbasis, old_recontructor=None):
        self.dim = dim
        self.dim_subbasis = dim_subbasis
        self.old_recontructor = old_recontructor

    def reconstruct(self, U):
        """Reconstruct high-dimensional vector from reduced vector `U`."""
        assert isinstance(U, NumpyVectorArray)
        UU = np.zeros((len(U), self.dim))
        UU[:, :self.dim_subbasis] = U.data
        UU = NumpyVectorArray(UU, copy=False)
        if self.old_recontructor:
            return self.old_recontructor.reconstruct(UU)
        else:
            return UU


def reduce_to_subbasis(discretization, dim, reconstructor=None):
    """Further reduce a |Discretization| to the subbasis formed by the first `dim` basis vectors.

    This is achieved by calling :meth:`~pymor.operators.numpy.NumpyMatrixOperator.projected_to_subbasis`
    for each operator of the given |Discretization|. Additionally, if a reconstructor
    for the |Discretization| is provided, its :meth:`restricted_to_subbasis` method is also
    called to obtain a reconstructor for the further reduced |Discretization|. Otherwise
    :class:`SubbasisReconstructor` is used (which will be less efficient).

    Parameters
    ----------
    discretization
        The |Discretization| to further reduce.
    dim
        The dimension of the subbasis.
    reconstructor
        Reconstructor for `discretization` or `None`.

    Returns
    -------
    rd
        The further reduced |Discretization|.
    rc
        Reconstructor for `rd`.
    """

    projected_operators = {k: op.projected_to_subbasis(dim_range=dim, dim_source=dim) if op is not None else None
                           for k, op in discretization.operators.items()}
    projected_functionals = {k: f.projected_to_subbasis(dim_range=None, dim_source=dim) if f is not None else None
                             for k, f in discretization.functionals.items()}
    projected_vector_operators = {k: op.projected_to_subbasis(dim_range=dim, dim_source=None) if op else None
                                  for k, op in discretization.vector_operators.items()}

    if discretization.products is not None:
        projected_products = {k: op.projected_to_subbasis(dim_range=dim, dim_source=dim)
                              for k, op in discretization.products.items()}
    else:
        projected_products = None

    if hasattr(discretization, 'estimator') and hasattr(discretization.estimator, 'restricted_to_subbasis'):
        estimator = discretization.estimator.restricted_to_subbasis(dim, discretization=discretization)
    elif hasattr(discretization, 'estimate'):
        # noinspection PyShadowingNames
        class FakeEstimator(object):
            rd = discretization
            rc = SubbasisReconstructor(discretization.solution_space.dim, dim)

            def estimate(self, U, mu=None, discretization=None):
                return self.rd.estimate(self.rc.reconstruct(U), mu=mu)
        estimator = FakeEstimator()
    else:
        estimator = None

    rd = discretization.with_(operators=projected_operators, functionals=projected_functionals,
                              vector_operators=projected_vector_operators,
                              products=projected_products, visualizer=None, estimator=estimator,
                              name=discretization.name + '_reduced_to_subbasis')
    rd.disable_logging()

    if reconstructor is not None and hasattr(reconstructor, 'restricted_to_subbasis'):
        rc = reconstructor.restricted_to_subbasis(dim)
    else:
        rc = SubbasisReconstructor(discretization.solution_space.dim, dim,
                                   old_recontructor=reconstructor)

    return rd, rc, {}
