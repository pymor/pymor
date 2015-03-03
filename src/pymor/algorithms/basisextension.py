# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains algorithms for extending a reduced given basis by a new vector.

The methods are mainly designed to be used in conjunction with
the :func:`~pymor.algorithms.greedy.greedy` algorithm. Each method is of the form ::

    extension_algorithm(basis, U, ...)

where `basis` and `U` are |VectorArrays| containing the old basis and new vectors
to add. The methods return a tuple `new_basis, data` where new_basis holds the
extend basis and data is a dict containing additional information about the extension
process. The `data` dict at least has the key `hierarchic` whose value signifies
if the new basis contains the old basis as its first vectors.

If the basis extension fails, e.g. because the new vector is not linearly
independent from the basis, an :class:`~pymor.core.exceptions.ExtensionError`
exception is raised.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.core.exceptions import ExtensionError


def trivial_basis_extension(basis, U, copy_basis=True, copy_U=True):
    """Trivially extend basis by simply appending the new vectors.

    We check if the new vectors are already contained in the basis, but we do
    not check for linear independence.

    Parameters
    ----------
    basis
        |VectorArray| containing the basis to extend.
    U
        |VectorArray| containing the new basis vectors.
    copy_basis
        If `copy_basis` is `False`, the old basis is extended in-place.
    copy_U
        If `copy_U` is `False`, the new basis vectors are removed from `U`.

    Returns
    -------
    new_basis
        The extended basis.
    extension_data
        Dict containing the following fields:

            :hierarchic: `True` if `new_basis` contains `basis` as its first vectors.

    Raises
    ------
    ExtensionError
        Is raised if all vectors in `U` are already contained in the basis.
    """
    if basis is None:
        basis = U.empty(reserve=len(U))

    old_basis_length = len(basis)
    remove = set()
    for i in xrange(len(U)):
        if np.any(U.almost_equal(basis, ind=i)):
            remove.add(i)

    new_basis = basis.copy() if copy_basis else basis
    new_basis.append(U, o_ind=[i for i in range(len(U)) if i not in remove],
                     remove_from_other=(not copy_U))

    if len(new_basis) == old_basis_length:
        raise ExtensionError

    return new_basis, {'hierarchic': True}


def gram_schmidt_basis_extension(basis, U, product=None, copy_basis=True, copy_U=True):
    """Extend basis using Gram-Schmidt orthonormalization.

    Parameters
    ----------
    basis
        |VectorArray| containing the basis to extend.
    U
        |VectorArray| containing the new basis vectors.
    product
        The scalar product w.r.t. which to orthonormalize; if `None`, the Euclidean
        product is used.
    copy_basis
        If `copy_basis` is `False`, the old basis is extended in-place.
    copy_U
        If `copy_U` is `False`, the new basis vectors are removed from `U`.

    Returns
    -------
    new_basis
        The extended basis.
    extension_data
        Dict containing the following fields:

            :hierarchic: `True` if `new_basis` contains `basis` as its first vectors.

    Raises
    ------
    ExtensionError
        Gram-Schmidt orthonormalization fails. This is the case when no
        vector in `U` is linearly independent from the basis.
    """
    if basis is None:
        basis = U.empty(reserve=len(U))

    basis_length = len(basis)

    new_basis = basis.copy() if copy_basis else basis
    new_basis.append(U, remove_from_other=(not copy_U))
    gram_schmidt(new_basis, offset=basis_length, product=product, copy=False)

    if len(new_basis) <= basis_length:
        raise ExtensionError

    return new_basis, {'hierarchic': True}


def pod_basis_extension(basis, U, count=1, copy_basis=True, product=None, orthonormalize=True):
    """Extend basis with the first `count` POD modes of the projection of `U` onto the
    orthogonal complement of the basis.

    Note that the provided basis is assumed to be orthonormal w.r.t. the provided
    scalar product!

    Parameters
    ----------
    basis
        |VectorArray| containing the basis to extend. The basis is expected to be
        orthonormal w.r.t. `product`.
    U
        |VectorArray| containing the vectors to which the POD is applied.
    count
        Number of POD modes that are to be appended to the basis.
    product
        The scalar product w.r.t. which to orthonormalize; if `None`, the Euclidean
        product is used.
    copy_basis
        If `copy_basis` is `False`, the old basis is extended in-place.
    orthonormalize
        If `True`, re-orthonormalize the new basis vectors obtained by the POD
        in order to improve numerical accuracy.

    Returns
    -------
    new_basis
        The extended basis.
    extension_data
        Dict containing the following fields:

            :hierarchic: `True` if `new_basis` contains `basis` as its first vectors.

    Raises
    ------
    ExtensionError
        POD produces no new vectors. This is the case when no vector in `U`
        is linearly independent from the basis.
    """
    if basis is None:
        return pod(U, modes=count, product=product)[0], {'hierarchic': True}

    basis_length = len(basis)

    new_basis = basis.copy() if copy_basis else basis

    if product is None:
        U_proj_err = U - basis.lincomb(U.dot(basis))
    else:
        U_proj_err = U - basis.lincomb(product.apply2(U, basis))

    new_basis.append(pod(U_proj_err, modes=count, product=product, orthonormalize=False)[0])

    if orthonormalize:
        gram_schmidt(new_basis, offset=len(basis), product=product, copy=False)

    if len(new_basis) <= basis_length:
        raise ExtensionError

    return new_basis, {'hierarchic': True}
