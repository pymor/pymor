# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.pod import pod
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


def pca(A, product=None, modes=None, rtol=None, atol=None, l2_err=None,
        method=None, orth_tol=None,
        return_reduced_coefficients=False, copy=True):
    """Principal component analysis (PCA) of `A` using :func:`~pymor.algorithms.pod.pod`.

    Viewing the |VectorArray| `A` as an `A.dim` x `len(A)` matrix, the
    return values of this method are the |VectorArray| of left singular
    vectors and a |NumPy array| of singular values of the singular value
    decomposition of `A` centered around the `mean`, where the inner
    product on R^(`dim(A)`) is given by `product` and the inner product
    on R^(`len(A)`) is the Euclidean inner product. If desired, also
    the right singular vectors, which correspond to the reduced
    coefficients of `A` w.r.t. the left singular vectors and singular
    values, are returned. To approximately reconstruct the original `A`,
    add the mean back, e.g.:

        reconstructed = principal_components.lincomb(coeffs) + mean

    Parameters
    ----------
    A
        See :func:`~pymor.algorithms.pod.pod`.
    product
        See :func:`~pymor.algorithms.pod.pod`.
    modes
        See :func:`~pymor.algorithms.pod.pod`.
    rtol
        See :func:`~pymor.algorithms.pod.pod`.
    atol
        See :func:`~pymor.algorithms.pod.pod`.
    l2_err
        See :func:`~pymor.algorithms.pod.pod`.
    method
        See :func:`~pymor.algorithms.pod.pod`.
    orth_tol
        See :func:`~pymor.algorithms.pod.pod`.
    return_reduced_coefficients
        See :func:`~pymor.algorithms.pod.pod`.
    copy
        If `True` (default) create a centered copy of `A`. If `False` subtract the mean
        from `A` in-place.

    Returns
    -------
    principal_components
        |VectorArray| of principal components.
    svals
        One-dimensional |NumPy array| of singular values.
    coeffs
        If `return_reduced_coefficients` is `True`, a |NumPy array|
        of right singular vectors as conjugated rows.
    mean
        |VectorArray| containing the empirical mean of the input `A`.
    """
    assert isinstance(A, VectorArray)
    assert product is None or isinstance(product, Operator)

    if copy:
        A = A.copy()

    logger = getLogger('pymor.algorithms.pca.pca')

    logger.info('Centering data around the mean ... ')
    weights = np.full(len(A), 1.0 / len(A))
    mean = A.lincomb(weights)
    A.axpy(-1, mean)

    with logger.block('Applying POD to centered data ...'):
        principal_coponents, svals, coeffs = pod(A, product=product, modes=modes, rtol=rtol,
                                                    atol=atol, l2_err=l2_err, method=method,
                                                    orth_tol=orth_tol, return_reduced_coefficients=True)

    if return_reduced_coefficients:
        return mean, principal_coponents, svals, coeffs
    return mean, principal_coponents, svals
