# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.linalg import eigh

from pymor.core import defaults
from pymor.core.exceptions import AccuracyError
from pymor.tools import float_cmp_all
from pymor.operators import OperatorInterface
from pymor.la import VectorArray
from pymor.la.gram_schmidt import gram_schmidt


def pod(A, modes=None, product=None, tol=None, symmetrize=None, orthonormalize=False,
        check=None, check_tol=None):

    assert isinstance(A, VectorArray)
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
    EVECS = EVECS.T[::-1, :] # is this a view?

    last_above_tol = np.where(EVALS >= tol)[0][-1]
    EVALS = EVALS[:last_above_tol + 1]
    EVECS = EVECS[:last_above_tol + 1]

    if len(EVALS) == 0:
        return type(A).empty(A.dim)

    POD = A.lincomb(EVECS / np.sqrt(EVALS[:, np.newaxis]))

    if orthonormalize:
        POD = gram_schmidt(POD, product=product)

    if check:
        if not product and not float_cmp_all(POD.prod(POD, pairwise=False), np.eye(len(POD)), check_tol):
            err = np.max(np.abs(POD.prod(POD, pairwise=False) - np.eye(len(POD))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))
        elif product and not float_cmp_all(product.apply2(POD, POD, pairwise=False), np.eye(len(POD)), check_tol):
            err = np.max(np.abs(product.apply2(POD, POD, pairwise=False) - np.eye(len(POD))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))

    return POD
