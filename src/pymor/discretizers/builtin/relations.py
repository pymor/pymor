# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np


DTYPE = np.int32


def inverse_relation(SE, size_rhs=None, with_indices=False):
    S, I = np.unravel_index(np.arange(SE.size), SE.shape)
    SE = SE.ravel()
    if size_rhs is None:
        size_rhs = np.max(SE) + 1
    E, indices, counts = np.unique(SE, return_index=True, return_counts=True)
    SUE = np.full((size_rhs, np.max(counts)), -1, dtype=DTYPE)
    SUI = np.full((size_rhs, np.max(counts)), -1, dtype=DTYPE)
    SUE[E, 0] = S[indices]
    SUI[E, 0] = I[indices]
    for i in range(1, SUE.shape[1]):
        SE = np.delete(SE, indices)
        S = np.delete(S, indices)
        I = np.delete(I, indices)
        E, indices = np.unique(SE, return_index=True)
        SUE[E, i] = S[indices]
        SUI[E, i] = I[indices]

    if with_indices:
        return SUE, SUI
    else:
        return SUI
