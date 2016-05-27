# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Module containing some basic but generic linear algebra algorithms."""

import numpy as np

from pymor.core.defaults import defaults
from pymor.operators.constructions import induced_norm


@defaults('rtol', 'atol')
def almost_equal(U, V, U_ind=None, V_ind=None, product=None, norm=None, rtol=1e-14, atol=1e-14):
    """Compare U and V for almost equality.

    The vectors of `U` and `V` are compared in pairs for almost equality.
    Two vectors `u` and `v` are considered almost equal iff

       ||u - v|| <= atol + ||v|| * rtol.

    The norm to be used can be specified via the `norm` or `product`
    parameter.

    If the length of `U` (`U_ind`) resp. `V` (`V_ind`) is 1, the one
    specified vector is compared to all vectors of the other array.
    Otherwise, the lengths of both indexed arrays have to agree.

    Parameters
    ----------
    U, V
        |VectorArrays| to be compared.
    U_ind, V_ind
        Indices of the vectors that are to be compared (see |VectorArray|).
    product
        If specified, use this scalar product |Operator| to compute the norm.
        `product` and `norm` are mutually exclusive.
    norm
        If specified, must be a callable, which is used to compute the norm
        or one of the string 'l1', 'l2', 'sup', in which case the respective
        |VectorArray| norm methods are used. `product` and `norm` are mutually
        exclusive. If neither is specified, `norm='l2'` is assumed.
    rtol
        The relative tolerance.
    atol
        The absolute tolerance.
    """

    assert product is None or norm is None
    assert not isinstance(norm, str) or norm in ('l1', 'l2', 'sup')
    norm = induced_norm(product) if product is not None else norm
    if norm is None:
        norm = 'l2'
    if isinstance(norm, str):
        norm_str = norm
        norm = lambda U: getattr(U, norm_str + '_norm')()

    X = V.copy(V_ind)
    V_norm = norm(X)

    # broadcast if necessary
    if len(X) == 1:
        len_U = U.len_ind(U_ind)
        if len_U > 1:
            X.append(X, o_ind=np.zeros(len_U - 1, dtype=np.int))

    X.axpy(-1, U, x_ind=U_ind)
    ERR_norm = norm(X)

    return ERR_norm <= atol + V_norm * rtol
