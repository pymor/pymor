# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip

from pymor.core.logger import getLogger
logger = getLogger(__name__)


def iadd_masked(U, V, U_ind):
    """Indexed, masked in-place addition.

    This is the same as ::

        U[U_ind] += V

    with two exceptions:
        1. Negative indices are skipped.
        2. If the same index is repeated, all additions are performed,
           not only the last one.
    """
    logger.warn('Call to unoptimized function iadd_masked')
    assert len(U_ind) == len(V), 'Lengths of U_ind and V must match'
    assert U.shape[1:] == V.shape[1:], 'U.shape[1:] != V.shape[1:]'

    for ind, v in izip(U_ind, V):
        if ind < 0:
            continue
        U[ind] += v


def isub_masked(U, V, U_ind):
    """Indexed, masked in-place subtraction.

    This is the same as ::

        U[U_ind] -= V

    with two exceptions:
        1. Negative indices are skipped.
        2. If the same index is repeated, all subtractions are performed,
           not only the last one.
    """
    logger.warn('Call to unoptimized function iadd_masked')
    assert len(U_ind) == len(V), 'Lengths of U_ind and V must match'
    assert U.shape[1:] == V.shape[1:], 'U.shape[1:] != V.shape[1:]'

    for ind, v in izip(U_ind, V):
        if ind < 0:
            continue
        U[ind] -= v
