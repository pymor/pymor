# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.core.exceptions import ExtensionError
from pymor.core.interfaces import BasicInterface


class GenericRBReductor(BasicInterface):
    """Generic reduced basis reductor.

    Replaces each |Operator| of the given |Discretization| with the Galerkin
    projection onto the span of the given reduced basis.

    Parameters
    ----------
    d
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the (initial) reduced basis on which to project.
    orthogonal_projection
        List of keys in `d.operators` for which the corresponding |Operator|
        should be orthogonally projected (i.e. operators which map to vectors in
        contrast to bilinear forms which map to functionals).
    product
        Inner product for the projection of the |Operators| given by
        `orthogonal_projection`.
    """

    def __init__(self, d, RB=None, orthogonal_projection=('initial_data',), product=None):
        self.d = d
        self.RB = d.solution_space.empty() if RB is None else RB
        self.orthogonal_projection = orthogonal_projection
        self.product = product

    def reduce(self):
        """Perform the reduced basis projection.

        Returns
        -------
        rd
            The reduced |Discretization|.
        """

        d = self.d
        RB = self.RB

        def project_operator(k, op):
            return project(op,
                           range_basis=RB if RB in op.range else None,
                           source_basis=RB if RB in op.source else None,
                           product=self.product if k in self.orthogonal_projection else None)

        projected_operators = {k: project_operator(k, op) if op else None for k, op in d.operators.items()}

        projected_products = {k: project_operator(k, p) for k, p in d.products.items()}

        rd = d.with_(operators=projected_operators, products=projected_products,
                     visualizer=None, estimator=None,
                     cache_region=None, name=d.name + '_reduced')
        rd.disable_logging()

        return rd

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.RB[:u.dim].lincomb(u.data)

    def extend_basis(self, U, method='gram_schmidt', pod_modes=1, pod_orthonormalize=True, copy_U=True):
        """Extend basis by simply appending the new vectors.

        We check if the new vectors are already contained in the basis, but we do
        not check for linear independence.

        Parameters
        ----------
        U
            |VectorArray| containing the new basis vectors.
        method
            Basis extension method to use ('trivial', 'gram_schmidt', 'pod')
        pod_modes
            Number of POD modes that shall be appended to the basis.
        pod_orthonormalize
            If `True`, re-orthonormalize the new basis vectors obtained by the POD
            in order to improve numerical accuracy.
        copy_U
            If `copy_U` is `False`, the new basis vectors are removed from `U`.

        Raises
        ------
        ExtensionError
            Raised when all vectors in `U` are already contained in the basis.
        """
        assert method in ('trivial', 'gram_schmidt', 'pod')

        basis_length = len(self.RB)

        if method == 'trivial':
            remove = set()
            for i in range(len(U)):
                if np.any(almost_equal(U[i], self.RB)):
                    remove.add(i)
            self.RB.append(U[[i for i in range(len(U)) if i not in remove]],
                           remove_from_other=(not copy_U))
        elif method == 'gram_schmidt':
            self.RB.append(U, remove_from_other=(not copy_U))
            gram_schmidt(self.RB, offset=basis_length, product=self.product, copy=False)
        elif method == 'pod':
            if self.product is None:
                U_proj_err = U - self.RB.lincomb(U.dot(self.RB))
            else:
                U_proj_err = U - self.RB.lincomb(self.product.apply2(U, self.RB))

            self.RB.append(pod(U_proj_err, modes=pod_modes, product=self.product, orthonormalize=False)[0])

            if pod_orthonormalize:
                gram_schmidt(self.RB, offset=basis_length, product=self.product, copy=False)

        if len(self.RB) <= basis_length:
            raise ExtensionError


def reduce_to_subbasis(d, dim):
    """Further reduce a |Discretization| to the subbasis formed by the first `dim` basis vectors.

    This is achieved by calling :meth:`~pymor.algorithms.projection.project_to_subbasis`
    for each operator of the given |Discretization|. Additionally, if a reconstructor
    for the |Discretization| is provided, its :meth:`restricted_to_subbasis` method is also
    called to obtain a reconstructor for the further reduced |Discretization|. Otherwise
    :class:`SubbasisReconstructor` is used (which will be less efficient).

    Parameters
    ----------
    d
        The |Discretization| to further reduce.
    dim
        The dimension of the subbasis.

    Returns
    -------
    rd
        The further reduced |Discretization|.
    """

    def project_operator(op):
        return project_to_subbasis(op,
                                   dim_range=dim if op.range == d.solution_space else None,
                                   dim_source=dim if op.source == d.solution_space else None)

    projected_operators = {k: project_operator(op) if op else None for k, op in d.operators.items()}

    projected_products = {k: project_operator(op) for k, op in d.products.items()}

    if hasattr(d.estimator, 'restricted_to_subbasis'):
        estimator = d.estimator.restricted_to_subbasis(dim, discretization=d)
    else:
        estimator = None

    rd = d.with_(operators=projected_operators, products=projected_products,
                 visualizer=None, estimator=estimator,
                 name=d.name + '_reduced_to_subbasis')
    rd.disable_logging()

    return rd
