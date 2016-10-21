# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains algorithms for extending a given reduced basis by a new vector.

The methods are mainly designed to be used in conjunction with
the :func:`~pymor.algorithms.greedy.greedy` algorithm. Each method is of the form ::

    extension_algorithm(basis, U, ...)

where `basis` and `U` are |VectorArrays| containing the basis to extend and new vectors
to be added. The methods return a dict containing information about the extension
process. It at least has the key `hierarchic` whose value signifies
if the new basis contains the old basis as its first vectors.

If the basis extension fails, e.g. because the new vector is not linearly
independent from the basis, an :class:`~pymor.core.exceptions.ExtensionError`
exception is raised.
"""

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.core.exceptions import ExtensionError


def trivial_basis_extension(basis, U, copy_U=True):
    """Extend basis by simply appending the new vectors.

    We check if the new vectors are already contained in the basis, but we do
    not check for linear independence.

    Parameters
    ----------
    basis
        |VectorArray| containing the basis to extend.
    U
        |VectorArray| containing the new basis vectors.
    copy_U
        If `copy_U` is `False`, the new basis vectors are removed from `U`.

    Returns
    -------
    extension_data
        Dict containing the following fields:

            :hierarchic: `True` if the extended basis contains the old basis
                         as its first vectors.

    Raises
    ------
    ExtensionError
        Raised when all vectors in `U` are already contained in the basis.
    """
    old_basis_length = len(basis)
    remove = set()
    for i in range(len(U)):
        if np.any(almost_equal(U[i], basis)):
            remove.add(i)

    basis.append(U[[i for i in range(len(U)) if i not in remove]],
                 remove_from_other=(not copy_U))

    if len(basis) == old_basis_length:
        raise ExtensionError

    return {'hierarchic': True}


def gram_schmidt_basis_extension(basis, U, product=None, copy_U=True):
    """Extend basis using Gram-Schmidt orthonormalization.

    Parameters
    ----------
    basis
        |VectorArray| containing the basis to extend.
    U
        |VectorArray| containing the new basis vectors.
    product
        The inner product w.r.t. which to orthonormalize; if `None`, the Euclidean
        product is used.
    copy_U
        If `copy_U` is `False`, the new basis vectors are removed from `U`.

    Returns
    -------
    extension_data
        Dict containing the following fields:

            :hierarchic: `True` if the extended basis contains the old basis
                         as its first vectors.

    Raises
    ------
    ExtensionError
        Gram-Schmidt orthonormalization has failed. This is the case when no
        vector in `U` is linearly independent of the basis.
    """
    basis_length = len(basis)

    basis.append(U, remove_from_other=(not copy_U))
    gram_schmidt(basis, offset=basis_length, product=product, copy=False)

    if len(basis) <= basis_length:
        raise ExtensionError

    return {'hierarchic': True}


def pod_basis_extension(basis, U, count=1, product=None, orthonormalize=True):
    """Extend basis with the first `count` POD modes of the defect of the projection of
    `U` onto the current basis.

    .. warning::
        The provided basis is assumed to be orthonormal w.r.t. the given
        inner product!

    Parameters
    ----------
    basis
        |VectorArray| containing the basis to extend. The basis is expected to be
        orthonormal w.r.t. `product`.
    U
        |VectorArray| containing the vectors to which the POD is applied.
    count
        Number of POD modes that shall be appended to the basis.
    product
        The inner product w.r.t. which to orthonormalize; if `None`, the Euclidean
        product is used.
    orthonormalize
        If `True`, re-orthonormalize the new basis vectors obtained by the POD
        in order to improve numerical accuracy.

    Returns
    -------
    extension_data
        Dict containing the following fields:

            :hierarchic: `True` if the extended basis contains the old basis
                         as its first vectors.

    Raises
    ------
    ExtensionError
        POD has produced no new vectors. This is the case when no vector in `U`
        is linearly independent of the basis.
    """
    basis_length = len(basis)

    if product is None:
        U_proj_err = U - basis.lincomb(U.dot(basis))
    else:
        U_proj_err = U - basis.lincomb(product.apply2(U, basis))

    basis.append(pod(U_proj_err, modes=count, product=product, orthonormalize=False)[0])

    if orthonormalize:
        gram_schmidt(basis, offset=len(basis), product=product, copy=False)

    if len(basis) <= basis_length:
        raise ExtensionError

    return {'hierarchic': True}
