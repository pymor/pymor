# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import getLogger
logger = getLogger(__name__)


def inverse_relation(R, size_rhs=None, with_indices=False):
    '''Computes the inverse relation of a relation.

    If `r` is a relation, then the inverse relation `ri` is defined by

        x ri y  <=>  y r x

    Parameters
    ----------
    R
        2D |NumPy array| of integers representing a relation r on the
        natural numbers via ::

            x r y <=> (x < R.size[0] and y in R[x]).

        Rows of `R` which are to short are padded with -1.
    size_rhs
        Can be provided for speedup. Has to be greater than `R.max()`.
    with_indices
        If `True`, also return the matrix `RINVI`.

    Returns
    -------
    RINV
        2D |NumPy array| representation of the inverse relation.
    RINVI
        |NumPy array| such that for `RINV[i, j] != -1`::

            R[RINV[i, j], RINVI[i, j]] = i.

        Only returned if `with_indices` is `True`.
    '''

    assert R.ndim == 2
    logger.warn('Call to unoptimized function inverse_relation')

    num_columns_RINV = np.bincount(R.ravel()).max()
    size_rhs = size_rhs or (R.max() + 1)
    RINV = np.empty((size_rhs, num_columns_RINV), dtype=R.dtype)
    RINV.fill(-1)
    if with_indices:
        RINVI = np.empty_like(RINV)
        RINVI.fill(-1)

    RINV_COL_COUNTS = np.zeros(size_rhs, dtype=np.int32)

    if not with_indices:
        for index, x in np.ndenumerate(R):
            if x >= 0:
                RINV[x, RINV_COL_COUNTS[x]] = index[0]
                RINV_COL_COUNTS[x] += 1
        return RINV
    else:
        for index, x in np.ndenumerate(R):
            if x >= 0:
                RINV[x, RINV_COL_COUNTS[x]] = index[0]
                RINVI[x, RINV_COL_COUNTS[x]] = index[1]
                RINV_COL_COUNTS[x] += 1
        return RINV, RINVI
