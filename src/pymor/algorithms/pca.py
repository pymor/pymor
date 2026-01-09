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
    """Principal component analysis (PCA) wrapper that centers `A`
    around the mean and then applies 'pod'.

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
        The |VectorArray| for which the POD is to be computed.
    product
        Inner product |Operator| w.r.t. which the POD is computed.
    modes
        If not `None`, at most the first `modes` POD modes (singular
        vectors) are returned.
    rtol
        Singular values smaller than this value multiplied by the
        largest singular value are ignored.
    atol
        Singular values smaller than this value are ignored.
    l2_err
        Do not return more modes than needed to bound the
        l2-approximation error by this value. I.e. the number of
        returned modes is at most ::

            argmin_N { sum_{n=N+1}^{infty} s_n^2 <= l2_err^2 }

        where `s_n` denotes the n-th singular value.
    method
        Which SVD method from :mod:`~pymor.algorithms.svd_va` to use
        (`'method_of_snapshots'` or `'qr_svd'`).
    orth_tol
        POD modes are reorthogonalized if the orthogonality error is
        above this value.
    return_reduced_coefficients
        Determines whether or not to also return the right singular
        vectors, which determine the reduced coefficients.

    Returns
    -------
    POD
        |VectorArray| of POD modes.
    SVALS
        One-dimensional |NumPy array| of singular values.
    COEFFS
        If `return_reduced_coefficients` is `True`, a |NumPy array|
        of right singular vectors as conjugated rows.
    mean
        |VectorArray| containing the empirical mean of the input `A`.
        The input |VectorArray| is centered by subtracting this mean
        before applying 'pod'. To reconstruct original snapshots add
        the mean back, e.g.:
        ``reconstructed = POD.lincomb(COEFFS) + mean``
    """
    assert isinstance(A, VectorArray)
    assert product is None or isinstance(product, Operator)
    assert method in ('method_of_snapshots', 'qr_svd')

    logger = getLogger('pymor.algorithms.pca.pca')

    # compute empirical mean and center A around the mean
    weights = np.full(len(A), 1.0 / len(A))
    mean = A.lincomb(weights)
    A_mean = A - mean

    # apply pod to centered data A_mean
    POD, SVALS, COEFFS = pod(A_mean, product=product, modes=modes, rtol=rtol,
                             atol=atol, l2_err=l2_err, method=method,
                             orth_tol=orth_tol, return_reduced_coefficients=True)

    if return_reduced_coefficients:
        return POD, SVALS, COEFFS, mean
    return POD, SVALS, mean
