# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.pod import pod
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


def pca(A, product=None, modes=None, rtol=None, atol=None, l2_err=None,
        method=None, orth_tol=None, return_reduced_coefficients=False, copy=True):
    """Principal component analysis (PCA) using :func:`~pymor.algorithms.pod.pod`.

    The principal components of `A` are the :func:`~pymor.algorithms.pod.pod`
    modes of `A` centered around the `mean`.

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
        If `True` (default) do not modify `A`. If `False` subtract the mean
        from `A` in-place.

    Returns
    -------
    mean
        |VectorArray| containing the empirical mean of the input `A`.
    principal_components
        |VectorArray| of principal components.
    svals
        One-dimensional |NumPy array| of singular values.
    coeffs
        If `return_reduced_coefficients` is `True`, a |NumPy array|
        of right singular vectors as conjugated rows.
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
        pod_result = pod(A, product=product, modes=modes, rtol=rtol,
                         atol=atol, l2_err=l2_err, method=method,
                         orth_tol=orth_tol, return_reduced_coefficients=return_reduced_coefficients)

    return (mean,) + pod_result
