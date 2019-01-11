# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.core.exceptions import ExtensionError
from pymor.core.interfaces import BasicInterface
from pymor.operators.constructions import IdentityOperator, Concatenation, InverseOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class GenericRBReductor(BasicInterface):
    """Generic reduced basis reductor.

    Replaces each |Operator| of the given |Discretization| with the Galerkin
    projection onto the span of the given reduced basis.

    Parameters
    ----------
    d
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    basis_is_orthonormal
        If `RB` is specified, indicate whether or not the basis is orthonormal
        w.r.t. `product`.
    vector_ranged_operators
        List of keys in `d.operators` for which the corresponding |Operator|
        should be orthogonally projected (i.e. operators which map to vectors in
        contrast to bilinear forms which map to functionals).
    product
        Inner product for the orthonormalization of `RB` and the projection of the
        |Operators| given by `vector_ranged_operators`.
    """

    def __init__(self, d, RB=None, basis_is_orthonormal=None,
                 vector_ranged_operators=('initial_data',), product=None):
        if RB is not None and basis_is_orthonormal is None:
            raise ValueError('Please specify is given basis is orthonormal using basis_is_orthonormal.')
        if RB is None and basis_is_orthonormal is None:
            basis_is_orthonormal = True
        self.d = d
        self.RB = d.solution_space.empty() if RB is None else RB
        assert self.RB in d.solution_space
        self.basis_is_orthonormal = basis_is_orthonormal
        self.vector_ranged_operators = vector_ranged_operators
        self.product = product
        self._last_rd = None

    def reduce(self, dim=None):
        """Perform the reduced basis projection.

        Parameters
        ----------
        dim
            If specified, the desired reduced state dimension. Must not be larger than the
            current reduced basis dimension.

        Returns
        -------
        The reduced |Discretization|.
        """
        if dim is None:
            dim = len(self.RB)
        if dim > len(self.RB):
            raise ValueError('Specified reduced state dimension larger than reduced basis')
        if self._last_rd is None or dim > self._last_rd.solution_space.dim:
            self._last_rd = self._reduce()
        if dim == self._last_rd.solution_space.dim:
            return self._last_rd
        else:
            return self._reduce_to_subbasis(dim)

    def _reduce(self):

        d = self.d
        RB = self.RB

        if any(k in self.vector_ranged_operators for k in d.operators) and not self.basis_is_orthonormal:
            projection_matrix = RB.inner(RB, self.product)
            projection_op = NumpyMatrixOperator(projection_matrix, source_id=RB.space.id, range_id=RB.space.id)
            inverse_projection_op = InverseOperator(projection_op, 'inverse_projection_op')

        def project_operator(k, op):
            if op is self.product and self.basis_is_orthonormal:
                return IdentityOperator(NumpyVectorSpace(len(RB), RB.space.id), name=op.name)
            elif k in self.vector_ranged_operators:
                assert RB in op.range
                pop = project(op, range_basis=RB, source_basis=RB if RB in op.source else None, product=self.product)
                if not self.basis_is_orthonormal:
                    return Concatenation([inverse_projection_op, pop])
                else:
                    return pop
            else:
                return project(op,
                               range_basis=RB if RB in op.range else None,
                               source_basis=RB if RB in op.source else None)

        projected_operators = {k: project_operator(k, op) if op else None for k, op in d.operators.items()}

        projected_products = {k: project_operator(k, p) for k, p in d.products.items()}

        rd = d.with_(operators=projected_operators, products=projected_products,
                     visualizer=None, estimator=None,
                     cache_region=None, name=d.name + '_reduced')
        rd.disable_logging()

        return rd

    def _reduce_to_subbasis(self, dim):
        rd = self._last_rd

        def project_operator(k, op):
            if k in self.vector_ranged_operators and not self.basis_is_orthonormal:
                assert (isinstance(op, Concatenation) and len(op.operators) == 2 and
                        op.operators[0].name == 'inverse_projection_op')
                pop = project_to_subbasis(op.operators[1],
                                          dim_range=dim,
                                          dim_source=dim if op.source == rd.solution_space else None)
                inverse_projection_op = InverseOperator(
                    project_to_subbasis(op.operators[0].operator, dim_range=dim, dim_source=dim),
                    name='inverse_projection_op'
                )
                return Concatenation([inverse_projection_op, pop])
            else:
                return project_to_subbasis(op,
                                           dim_range=dim if op.range == rd.solution_space else None,
                                           dim_source=dim if op.source == rd.solution_space else None)

        projected_operators = {k: project_operator(k, op) if op else None for k, op in rd.operators.items()}

        projected_products = {k: project_operator(k, op) for k, op in rd.products.items()}

        if rd.estimator:
            estimator = rd.estimator.restricted_to_subbasis(dim, d=rd)
        else:
            estimator = None

        rrd = rd.with_(operators=projected_operators, products=projected_products, estimator=estimator,
                       visualizer=None, name=rd.name + '_reduced_to_subbasis')

        return rrd

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.RB[:u.dim].lincomb(u.to_numpy())

    def extend_basis(self, U, method='gram_schmidt', pod_modes=1, pod_orthonormalize=True, orthonormal=None,
                     copy_U=True):
        """Extend basis by new vectors.

        Parameters
        ----------
        U
            |VectorArray| containing the new basis vectors.
        method
            Basis extension method to use. The following methods are available:

                :trivial:      Vectors in `U` are appended to the basis. Duplicate vectors
                               in the sense of :func:`~pymor.algorithms.basic.almost_equal`
                               are removed.
                :gram_schmidt: New basis vectors are orthonormalized w.r.t. to the old
                               basis using the :func:`~pymor.algorithms.gram_schmidt.gram_schmidt`
                               algorithm.
                :pod:          Append the first POD modes of the defects of the projections
                               of the vectors in U onto the existing basis
                               (e.g. for use in POD-Greedy algorithm).

            .. warning::
                In case of the `'gram_schmidt'` and `'pod'` extension methods, the existing reduced
                basis is assumed to be orthonormal w.r.t. the given inner product.

        pod_modes
            In case `method == 'pod'`, the number of POD modes that shall be appended to
            the basis.
        pod_orthonormalize
            If `True` and `method == 'pod'`, re-orthonormalize the new basis vectors obtained
            by the POD in order to improve numerical accuracy.
        orthonormal
            If `method == 'trivial'`, set this to `True` to indicate that the basis will remain
            orthonormal after extending.
        copy_U
            If `copy_U` is `False`, the new basis vectors might be removed from `U`.

        Raises
        ------
        ExtensionError
            Raised when the selected extension method does not yield a basis of increased
            dimension.
        """
        assert method == 'trivial' or orthonormal is None
        extend_basis(self.RB, U, self.product, method=method, pod_modes=pod_modes,
                     pod_orthonormalize=pod_orthonormalize, copy_U=copy_U)
        if method == 'trivial' and not orthonormal:
            self.basis_is_orthonormal = False


class GenericPGReductor(BasicInterface):
    """Generic Petrov-Galerkin reductor.

    Replaces each |Operator| of the given |Discretization| with the projection
    onto the span of the given projection matrices.

    Parameters
    ----------
    d
        The |Discretization| which is to be reduced.
    W
        |VectorArray| containing the left projection matrix.
    V
        |VectorArray| containing the right projection matrix.
    bases_are_biorthonormal
        Indicate whether or not V and W are biorthonormal w.r.t. `product`.
    vector_ranged_operators
        List of keys in `d.operators` for which the corresponding |Operator|
        should be biorthogonally projected (i.e. operators which map to vectors in
        contrast to bilinear forms which map to functionals).
    product
        Inner product for the projection of the |Operators| given by
        `vector_ranged_operators`.
    """

    def __init__(self, d, W, V, bases_are_biorthonormal, vector_ranged_operators=('initial_data',), product=None):
        assert V in d.solution_space
        assert product is None or (W in product.range and V in product.source)
        self.d = d
        self.V = V
        self.W = W
        self.bases_are_biorthonormal = bases_are_biorthonormal
        self.vector_ranged_operators = vector_ranged_operators
        self.product = product

    def reduce(self):
        """Perform the Petrov-Galerkin projection.

        Returns
        -------
        The reduced |Discretization|.
        """
        d, V, W = self.d, self.V, self.W
        product = self.product

        if any(k in self.vector_ranged_operators for k in d.operators) and not self.bases_are_biorthonormal:
            projection_matrix = W.inner(V, product)
            projection_op = NumpyMatrixOperator(projection_matrix, source_id=V.space.id, range_id=W.space.id)
            inverse_projection_op = InverseOperator(projection_op, 'inverse_projection_op')

        def project_operator(k, op):
            if not op:
                return None
            if k is product and self.bases_are_biorthonormal:
                if V.space.id != W.space.id:
                    raise NotImplementedError
                return IdentityOperator(NumpyVectorSpace(len(V), V.space.id))
            elif k in self.vector_ranged_operators:
                assert W in op.range
                pop = project(op, range_basis=W, source_basis=V if V in op.source else None, product=product)
                if not self.bases_are_biorthonormal:
                    return Concatenation([inverse_projection_op, pop])
                else:
                    return pop
            else:
                return project(op,
                               range_basis=W if W in op.range else None,
                               source_basis=V if V in op.source else None)

        projected_ops = {k: project_operator(k, op) for k, op in d.operators.items()}

        rd = d.with_(operators=projected_ops,
                     visualizer=None, estimator=None,
                     cache_region=None, name=d.name + '_reduced')
        rd.disable_logging()

        return rd

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.V[:u.dim].lincomb(u.to_numpy())


def extend_basis(basis, U, product=None, method='gram_schmidt', pod_modes=1, pod_orthonormalize=True, copy_U=True):
    assert method in ('trivial', 'gram_schmidt', 'pod')

    basis_length = len(basis)

    if method == 'trivial':
        remove = set()
        for i in range(len(U)):
            if np.any(almost_equal(U[i], basis)):
                remove.add(i)
        basis.append(U[[i for i in range(len(U)) if i not in remove]],
                     remove_from_other=(not copy_U))
    elif method == 'gram_schmidt':
        basis.append(U, remove_from_other=(not copy_U))
        gram_schmidt(basis, offset=basis_length, product=product, copy=False)
    elif method == 'pod':
        U_proj_err = U - basis.lincomb(U.inner(basis, product))

        basis.append(pod(U_proj_err, modes=pod_modes, product=product, orthonormalize=False)[0])

        if pod_orthonormalize:
            gram_schmidt(basis, offset=basis_length, product=product, copy=False)

    if len(basis) <= basis_length:
        raise ExtensionError
