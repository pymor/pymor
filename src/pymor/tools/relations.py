from __future__ import absolute_import, division, print_function

import numpy as np
import logging

def inverse_relation(R, size_rhs=None, with_indices=False):
    assert R.ndim == 2
    logging.warn('Call to unoptimized function inverse_relation')

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
