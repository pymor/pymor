# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip

from pymor.core import getLogger
logger = getLogger(__name__)

def iadd_masked(U, V, U_ind):
    logger.warn('Call to unoptimized function iadd_masked')
    assert len(U_ind) == len(V), 'Lengths of U_ind and V must match'
    assert U.shape[1:] == V.shape[1:], 'U.shape[1:] != V.shape[1:]'

    for ind, v in izip(U_ind):
        if ind < 0:
            continue
        U[ind] += v


def isub_masked(U, V, U_ind):
    logger.warn('Call to unoptimized function iadd_masked')
    assert len(U_ind) == len(V), 'Lengths of U_ind and V must match'
    assert U.shape[1:] == V.shape[1:], 'U.shape[1:] != V.shape[1:]'

    for ind, v in izip(U_ind):
        if ind < 0:
            continue
        U[ind] -= v
