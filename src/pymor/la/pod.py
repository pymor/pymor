# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.linalg import eigh

from pymor import defaults
from pymor.core.exceptions import AccuracyError
from pymor.la import VectorArrayInterface
from pymor.la.gram_schmidt import gram_schmidt
from pymor.operators import OperatorInterface
from pymor.tools import float_cmp_all


def pod(A, modes=None, product=None, tol=None, symmetrize=None, orthonormalize=None,
        check=None, check_tol=None):
    '''Proper orthogonal decomposition of `A`.

    If the |VectorArray| `A` is viewed as a map ::

        A: R^(len(A)) ---> R^(dim(A))

    then the return value of this method is simply the array of left-singular
    vectors of the singular value decomposition of `A`, where the scalar product
    on R^(dim(A) is given by `product` and the scalar product on R^(len(A)) is
    the Euclidean product.

    Parameters
    ----------
    A
        The |VectorArray| for which the POD is to be computed.
    modes
        If not `None` only the first `modes` POD modes (singular vectors) are
        returned.
    products
        Scalar product given as a linear |Operator| w.r.t. compute the POD.
    tol
        Squared singular values smaller than this value are ignored. If `None`,
        the `pod_tol` |default| value is used.
    symmetrize
        If `True` symmetrize the gramian again before proceeding. If `None`,
        the `pod_symmetrize` |default| value is chosen.
    orthonormalize
        If `True`, orthonormalize the computed POD modes again using
        :func:`la.gram_schmidt.gram_schmidt`. If `None`, the `pod_orthonormalize`
        |default| value is used.
    check
        If `True`, check the computed POD modes for orthonormality. If `None`,
        the `pod_check` |default| value is used.
    check_tol
        Tolerance for the orthonormality check. If `None`, the `pod_check_tol`
        |default| value is used.

    Returns
    -------
    |VectorArray| of POD modes.

    '''

    assert isinstance(A, VectorArrayInterface)
    assert len(A) > 0
    assert modes is None or modes <= len(A)
    assert product is None or isinstance(product, OperatorInterface)

    tol = defaults.pod_tol if tol is None else tol
    symmetrize = defaults.pod_symmetrize if symmetrize is None else symmetrize
    orthonormalize = defaults.pod_orthonormalize if orthonormalize is None else orthonormalize
    check = defaults.pod_check if check is None else check
    check_tol = defaults.pod_check_tol if check_tol is None else check_tol

    B = A.gramian() if product is None else product.apply2(A, A, pairwise=False)

    if symmetrize:     # according to rbmatlab this is necessary due to rounding
        B = B + B.T
        B *= 0.5

    eigvals = None if modes is None else (len(B) - modes, len(B) - 1)

    EVALS, EVECS = eigh(B, overwrite_a=True, turbo=True, eigvals=eigvals)
    EVALS = EVALS[::-1]
    EVECS = EVECS.T[::-1, :]  # is this a view? yes it is!

    above_tol = np.where(EVALS >= tol)[0]
    if len(above_tol) == 0:
        return type(A).empty(A.dim)
    last_above_tol = above_tol[-1]

    EVALS = EVALS[:last_above_tol + 1]
    EVECS = EVECS[:last_above_tol + 1]

    POD = A.lincomb(EVECS / np.sqrt(EVALS[:, np.newaxis]))

    if orthonormalize:
        POD = gram_schmidt(POD, product=product, copy=False)

    if check:
        if not product and not float_cmp_all(POD.dot(POD, pairwise=False), np.eye(len(POD)), check_tol):
            err = np.max(np.abs(POD.dot(POD, pairwise=False) - np.eye(len(POD))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))
        elif product and not float_cmp_all(product.apply2(POD, POD, pairwise=False), np.eye(len(POD)), check_tol):
            err = np.max(np.abs(product.apply2(POD, POD, pairwise=False) - np.eye(len(POD))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))

    return POD
