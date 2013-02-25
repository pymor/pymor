from __future__ import absolute_import, division, print_function

from itertools import izip

import numpy as np

from pymor.core.exceptions import ExtensionError
from pymor.tools import float_cmp_all
from pymor.la import gram_schmidt


def trivial_basis_extension(basis, U):
    if basis is None: return np.reshape(U, (1, -1))

    # check if snapshot is already contained in basis; we do not check for linear independence
    assert isinstance(basis, np.ndarray)
    if not all(not float_cmp_all(B, U) for B in basis): raise ExtensionError

    new_basis = np.empty((basis.shape[0] + 1, basis.shape[1]))
    new_basis[:-1, :] = basis
    new_basis[-1, :] = U

    return new_basis


def gram_schmidt_basis_extension(basis, U, product=None):
    if basis is None: basis = np.zeros((0, len(U)))

    assert isinstance(basis, np.ndarray)
    new_basis = np.empty((basis.shape[0] + 1, basis.shape[1]))
    new_basis[:-1, :] = basis
    new_basis[-1, :] = U
    new_basis = gram_schmidt(new_basis, row_offset=basis.shape[0], product=product)

    if new_basis.size <= basis.size: raise ExtensionError

    return new_basis
