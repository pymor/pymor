# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.pod import pod
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


def pca(A, product=None, modes=None, rtol=1e-7, atol=0., l2_err=0.,
        method='method_of_snapshots', orth_tol=1e-10,
        return_scores=False, copy=True):
    """Principal component analysis (PCA) wrapper that centers `A` and applies 'pod'.

    Viewing the |VectorArray| `A` as a `A.dim` x `len(A)` matrix, the
    return values of this method are the |VectorArray| of left singular
    vectors and a |NumPy array| of singular values of the singular value
    decomposition of `A` centered around the `mean`, where the inner
    product on R^(`dim(A)`) is given by `product` and the inner product
    on R^(`len(A)`) is the Euclidean inner product. If desired, also
    the right singular vectors, which correspond to the reduced
    coefficients of `A` w.r.t. the left singular vectors and singular
    values, are returned. To approximately reconstruct the original `A`,
    add the mean back, e.g.:
        ``reconstructed = principal_components.lincomb(coeffs) + mean``

    Parameters
    ----------
    A
        See :class:`~pymor.algorithms.pod`.
    product
        See :class:`~pymor.algorithms.pod`.
    modes
        See :class:`~pymor.algorithms.pod`.
    rtol
        See :class:`~pymor.algorithms.pod`.
    atol
        See :class:`~pymor.algorithms.pod`.
    l2_err
        See :class:`~pymor.algorithms.pod`.
    method
        See :class:`~pymor.algorithms.pod`.
    orth_tol
        See :class:`~pymor.algorithms.pod`.
    return_scores
        See :class:`~pymor.algorithms.pod`.
    copy
        If `True` (default) create a centered copy of `A`. If `False` subtract the mean
        from `A` in-place and return the modified `A` as the
        centered data.

    Returns
    -------
    principal_components
        |VectorArray| of PCA coordinates.
    svals
        One-dimensional |NumPy array| of singular values.
    scores
        If `return_scores` is `True`, a |NumPy array|
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
        principal_coponents, svals, scores = pod(A, product=product, modes=modes, rtol=rtol,
                                                    atol=atol, l2_err=l2_err, method=method,
                                                    orth_tol=orth_tol, return_reduced_coefficients=True)

    if return_scores:
        return mean, principal_coponents, svals, scores
    return mean, principal_coponents, svals
