# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Module containing some basic but generic linear algebra algorithms."""

import numpy as np
import scipy

from pymor.core.defaults import defaults
from pymor.tools.floatcmp import float_cmp


@defaults('rtol', 'atol')
def almost_equal(U, V, product=None, sup_norm=False, rtol=1e-14, atol=1e-14):
    """Compare U and V for almost equality.

    The vectors of `U` and `V` are compared in pairs for almost equality.
    Two vectors `u` and `v` are considered almost equal iff

       ||u - v|| <= atol + ||v|| * rtol.

    The norm to be used can be specified via the `product` or `sup_norm`
    parameters.

    If the length of `U`  resp. `V`  is 1, the single specified
    vector is compared to all vectors of the other array.
    Otherwise, the lengths of both indexed arrays have to agree.

    Parameters
    ----------
    U, V
        |VectorArrays| to be compared.
    product
        If specified, use this inner product |Operator| to compute the norm.
        Otherwise, the Euclidean norm is used.
    sup_norm
        If `True`, use the sup norm to compute the error.
    rtol
        The relative tolerance.
    atol
        The absolute tolerance.
    """
    assert product is None or not sup_norm

    V_norm = V.sup_norm() if sup_norm else V.norm(product)
    X = V.copy()

    # broadcast if necessary
    if len(X) == 1:
        if len(U) > 1:
            X.append(X[np.zeros(len(U) - 1, dtype=int)])

    X -= U
    ERR_norm = X.sup_norm() if sup_norm else X.norm(product)

    return ERR_norm <= atol + V_norm * rtol


def relative_error(U, V, product=None):
    """Compute error between U and V relative to norm of U."""
    return (U - V).norm(product) / U.norm(product)


def project_array(U, basis, product=None, orthonormal=True):
    """Orthogonal projection of |VectorArray| onto subspace.

    Parameters
    ----------
    U
        The |VectorArray| to project.
    basis
        |VectorArray| of basis vectors for the subspace onto which
        to project.
    product
        Inner product |Operator| w.r.t. which to project.
    orthonormal
        If `True`, the vectors in `basis` are assumed to be orthonormal
        w.r.t. `product`.

    Returns
    -------
    The projected |VectorArray|.
    """
    if orthonormal:
        return basis.lincomb(basis.inner(U, product).T)
    else:
        gramian = basis.gramian(product)
        rhs = basis.inner(U, product)
        coeffs = scipy.linalg.solve(gramian, rhs, sym_pos=True, overwrite_a=True, overwrite_b=True).T
        return basis.lincomb(coeffs)


def contains_zero_vector(vector_array, rtol=None, atol=None):
    """Return `True` iff any vector in the array float_compares to 0s of the same dim.

    Parameters
    ----------
    vector_array
        a |VectorArray| implementation
    rtol
        relative tolerance for float_cmp
    atol
        absolute tolerance for float_cmp
    """
    for i in range(len(vector_array)):
        sup = vector_array[i].sup_norm()
        if float_cmp(sup, 0.0, rtol, atol):
            return True
    return False
