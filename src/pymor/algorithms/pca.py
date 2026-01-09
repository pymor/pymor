# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.pod import pod


@defaults('rtol', 'atol', 'l2_err', 'method', 'orth_tol')
def pca(A, product=None, modes=None, rtol=1e-7, atol=0., l2_err=0.,
        method='method_of_snapshots', orth_tol=1e-10,
        return_reduced_coefficients=False):

    """Principal component analysis (PCA) wrapper that centers `A` around the mean and applies 'pod'.

    Viewing the |VectorArray| `A` as a `A.dim` x `len(A)` matrix, the
    return values of this method are the |VectorArray| of left singular
    vectors and a |NumPy array| of singular values of the singular value
    decomposition of `A` centered around the mean, where the inner
    product on R^(`dim(A)`) is given by `product` and the inner product
    on R^(`len(A)`) is the Euclidean inner product. If desired, also
    the right singular vectors, which correspond to the reduced
    coefficients of `A` w.r.t. the left singular vectors and singular
    values, are returned.

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
    return_reduced_coefficients
        See :class:`~pymor.algorithms.pod`.

    Returns
    -------
    pod_results
        a tuple `(PRINCIPAL_COMPONENTS, SVALS)` or
        `(PRINCIPAL_COMPONENTS, SVALS, COEFFS)` depending on the value of
        `return_reduced_coefficients`:

        PRINCIPAL_COMPONENTS
            |VectorArray| of PCA coordinates.
        SVALS
            One-dimensional |NumPy array| of singular values.
        COEFFS
            If `return_reduced_coefficients` is `True`, a |NumPy array|
            of right singular vectors as conjugated rows.
    mean
        |VectorArray| containing the empirical mean of the input `A`.
        The input |VectorArray| is centered by subtracting this mean
        before applying 'pod'. To approximately reconstruct original snapshots add
        the mean back, e.g.:
        ``reconstructed = POD.lincomb(COEFFS) + mean``
    """
    assert isinstance(A, VectorArray)
    assert product is None or isinstance(product, Operator)
    assert method in ('method_of_snapshots', 'qr_svd')

    logger = getLogger('pymor.algorithms.pca.pca')

    logger.info('Computing empirical mean and centering data ... ')
    weights = np.full(len(A), 1.0 / len(A))
    mean = A.lincomb(weights)
    A_mean = A - mean

    with logger.block('Applying POD to centered data ...'):
        pod_results = pod(A_mean, product=product, modes=modes, rtol=rtol,
                                 atol=atol, l2_err=l2_err, method=method,
                                 orth_tol=orth_tol, return_reduced_coefficients=True)

    if return_reduced_coefficients:
        return pod_results, mean
    return pod_results, mean
