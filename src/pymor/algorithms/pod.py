# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.svd_va import method_of_snapshots, qr_svd
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


@defaults('rtol', 'atol', 'l2_err', 'method', 'orth_tol')
def pod(A, product=None, modes=None, rtol=1e-7, atol=0., l2_err=0.,
        method='method_of_snapshots', orth_tol=1e-10):
    """Proper orthogonal decomposition of `A`.

    Viewing the |VectorArray| `A` as a `A.dim` x `len(A)` matrix, the
    return values of this method are the |VectorArray| of left singular
    vectors and a |NumPy array| of singular values of the singular value
    decomposition of `A`, where the inner product on R^(`dim(A)`) is
    given by `product` and the inner product on R^(`len(A)`) is the
    Euclidean inner product.

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

    Returns
    -------
    POD
        |VectorArray| of POD modes.
    SVALS
        One-dimensional |NumPy array| of singular values.
    """
    assert isinstance(A, VectorArray)
    assert product is None or isinstance(product, Operator)
    assert method in ('method_of_snapshots', 'qr_svd')

    logger = getLogger('pymor.algorithms.pod.pod')

    svd_va = method_of_snapshots if method == 'method_of_snapshots' else qr_svd
    with logger.block('Computing SVD ...'):
        POD, SVALS, _ = svd_va(A, product=product, modes=modes, rtol=rtol, atol=atol, l2_err=l2_err)

    if POD.dim > 0 and len(POD) > 0 and np.isfinite(orth_tol):
        logger.info('Checking orthonormality ...')
        err = np.max(np.abs(POD.inner(POD, product) - np.eye(len(POD))))
        if err >= orth_tol:
            logger.info('Reorthogonalizing POD modes ...')
            gram_schmidt(POD, product=product, atol=0., rtol=0., copy=False)

    return POD, SVALS
