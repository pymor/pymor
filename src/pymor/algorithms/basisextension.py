# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.exceptions import ExtensionError
from pymor.tools import float_cmp_all
from pymor.la import gram_schmidt


def trivial_basis_extension(basis, U):
    '''Trivially extend basis by just adding the new vector.

    We check that the new vector is not already contained in the basis, but we do
    not check for linear independence.

    Parameters
    ----------
    basis
        The basis to extend.
    U
        The new basis vector.

    Returns
    -------
    The new basis.

    Raises
    ------
    ExtensionError
        Is raised if U is already contained in basis.
    '''
    if basis is None:
        return np.reshape(U, (1, -1))

    assert isinstance(basis, np.ndarray)
    if not all(not float_cmp_all(B, U) for B in basis):
        raise ExtensionError

    new_basis = np.empty((basis.shape[0] + 1, basis.shape[1]))
    new_basis[:-1, :] = basis
    new_basis[-1, :] = U

    return new_basis


def gram_schmidt_basis_extension(basis, U, product=None):
    '''Extend basis using Gram-Schmidt orthonormalization.

    Parameters
    ----------
    basis
        The basis to extend.
    U
        The new basis vector.
    product
        The scalar product w.r.t. which to orthonormalize; if None, the l2-scalar
        product on the coefficient vector is used.

    Returns
    -------
    The new basis.

    Raises
    ------
    ExtensionError
        Gram-Schmidt orthonormalization fails. Usually this is the case when U
        is not linearily independent from the basis. However this can also happen
        due to rounding errors ...
    '''
    if basis is None:
        basis = np.zeros((0, len(U)))

    assert isinstance(basis, np.ndarray)
    new_basis = np.empty((basis.shape[0] + 1, basis.shape[1]))
    new_basis[:-1, :] = basis
    new_basis[-1, :] = U
    new_basis = gram_schmidt(new_basis, row_offset=basis.shape[0], product=product)

    if new_basis.size <= basis.size:
        raise ExtensionError

    return new_basis
