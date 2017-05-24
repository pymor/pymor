# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.projection import project, project_to_subbasis
from pymor.core.interfaces import BasicInterface
from pymor.vectorarrays.numpy import NumpyVectorSpace


class GenericRBReconstructor(BasicInterface):
    """Simple reconstructor forming linear combinations with a reduced basis."""

    def __init__(self, RB):
        self.RB = RB.copy()

    def reconstruct(self, U):
        """Reconstruct high-dimensional vector from reduced vector `U`."""
        RB = self.RB
        assert U in NumpyVectorSpace(len(RB), RB.space.id)
        return self.RB.lincomb(U.data)

    def restricted_to_subbasis(self, dim):
        """See :meth:`~pymor.algorithms.projection.project_to_subbasis`."""
        assert dim <= len(self.RB)
        return GenericRBReconstructor(self.RB[:dim])


def reduce_generic_rb(discretization, RB, orthogonal_projection=('initial_data',), product=None,
                      disable_caching=True, extends=None):
    """Generic reduced basis reductor.

    Replaces each |Operator| of the given |Discretization| with the Galerkin
    projection onto the span of the given reduced basis.

    Parameters
    ----------
    discretization
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    orthogonal_projection
        List of keys in `discretization.operators` for which the corresponding |Operator|
        should be orthogonally projected (i.e. operators which map to vectors in
        contrast to bilinear forms which map to functionals).
    product
        Inner product for the projection of the |Operators| given by
        `orthogonal_projection`.
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

    def project_operator(k, op):
        return project(op,
                       range_basis=RB if RB in op.range else None,
                       source_basis=RB if RB in op.source else None,
                       product=product if k in orthogonal_projection else None)

    projected_operators = {k: project_operator(k, op) if op else None for k, op in discretization.operators.items()}

    projected_products = {k: project_operator(k, p) for k, p in discretization.products.items()}

    cache_region = None if disable_caching else discretization.caching

    rd = discretization.with_(operators=projected_operators, products=projected_products,
                              visualizer=None, estimator=None,
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
        assert isinstance(U.space, NumpyVectorSpace)
        UU = np.zeros((len(U), self.dim))
        UU[:, :self.dim_subbasis] = U.data
        UU = NumpyVectorSpace.make_array(UU, U.space.id)
        if self.old_recontructor:
            return self.old_recontructor.reconstruct(UU)
        else:
            return UU


def reduce_to_subbasis(discretization, dim, reconstructor=None):
    """Further reduce a |Discretization| to the subbasis formed by the first `dim` basis vectors.

    This is achieved by calling :meth:`~pymor.algorithms.projection.project_to_subbasis`
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

    def project_operator(op):
        return project_to_subbasis(op,
                                     dim_range=dim if op.range == discretization.solution_space else None,
                                     dim_source=dim if op.source == discretization.solution_space else None)

    projected_operators = {k: project_operator(op) if op else None for k, op in discretization.operators.items()}

    projected_products = {k: project_operator(op) for k, op in discretization.products.items()}

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

    rd = discretization.with_(operators=projected_operators, products=projected_products,
                              visualizer=None, estimator=estimator,
                              name=discretization.name + '_reduced_to_subbasis')
    rd.disable_logging()

    if reconstructor is not None and hasattr(reconstructor, 'restricted_to_subbasis'):
        rc = reconstructor.restricted_to_subbasis(dim)
    else:
        rc = SubbasisReconstructor(discretization.solution_space.dim, dim,
                                   old_recontructor=reconstructor)

    return rd, rc, {}
