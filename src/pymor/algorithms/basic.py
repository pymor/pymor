# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Module containing some basic but generic linear algebra algorithms."""

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.interfaces import ImmutableInterface
from pymor.parameters.base import Parametric


@defaults('rtol', 'atol')
def almost_equal(U, V, product=None, norm=None, rtol=1e-14, atol=1e-14):
    """Compare U and V for almost equality.

    The vectors of `U` and `V` are compared in pairs for almost equality.
    Two vectors `u` and `v` are considered almost equal iff

       ||u - v|| <= atol + ||v|| * rtol.

    The norm to be used can be specified via the `norm` or `product`
    parameter.

    If the length of `U`  resp. `V`  is 1, the single specified
    vector is compared to all vectors of the other array.
    Otherwise, the lengths of both indexed arrays have to agree.

    Parameters
    ----------
    U, V
        |VectorArrays| to be compared.
    product
        If specified, use this inner product |Operator| to compute the norm.
        `product` and `norm` are mutually exclusive.
    norm
        If specified, must be a callable, which is used to compute the norm
        or, alternatively, one of the string 'l1', 'l2', 'sup', in which case the
        respective |VectorArray| norm methods are used.
        `product` and `norm` are mutually exclusive. If neither is specified,
        `norm='l2'` is assumed.
    rtol
        The relative tolerance.
    atol
        The absolute tolerance.
    """

    assert product is None or norm is None
    assert not isinstance(norm, str) or norm in ('l1', 'l2', 'sup')
    norm = Norm(product) if product is not None else norm
    if norm is None:
        norm = 'l2'
    if isinstance(norm, str):
        norm_str = norm
        norm = lambda U: getattr(U, norm_str + '_norm')()

    X = V.copy()
    V_norm = norm(X)

    # broadcast if necessary
    if len(X) == 1:
        if len(U) > 1:
            X.append(X[np.zeros(len(U) - 1, dtype=np.int)])

    X -= U
    ERR_norm = norm(X)

    return ERR_norm <= atol + V_norm * rtol


def inner(V, U, product=None):
    if product:
        return product.apply2(V, U)
    else:
        return V.dot(U)


def pairwise_inner(V, U, product=None):
    if product:
        return product.pairwise_apply2(V, U)
    else:
        return V.pairwise_dot(U)


@defaults('raise_negative', 'tol')
def norm(U, product=None, raise_negative=True, tol=1e-10):
    if product:
        norm_squared = product.pairwise_apply2(U, U)
        if tol > 0:
            norm_squared = np.where(np.logical_and(0 > norm_squared, norm_squared > - tol),
                                    0, norm_squared)
        if raise_negative and np.any(norm_squared < 0):
            raise ValueError('norm is negative (square = {})'.format(norm_squared))
        return np.sqrt(norm_squared)
    else:
        return U.l2_norm()


class Norm(ImmutableInterface):
    """The induced norm of a scalar product.

    The norm of a the vectors in a |VectorArray| U is calculated by
    calling ::

        product.pairwise_apply2(U, U, mu=mu)

    In addition, negative norm squares of absolute value smaller
    than `tol` are clipped to `0`.
    If `raise_negative` is `True`, a :exc:`ValueError` exception
    is raised if there are still negative norm squares afterwards.

    Parameters
    ----------
    product
        The scalar product |Operator| for which the norm is to be
        calculated.
    raise_negative
        If `True`, raise an exception if calcuated norm is negative.
    tol
        See above.

    Returns
    -------
    norm
        A function `norm(U, mu=None)` taking a |VectorArray| `U`
        as input together with the |Parameter| `mu` which is
        passed to the product.
    """

    def __init__(self, product, raise_negative=None, tol=None, name=None):
        self.product = product
        self.raise_negative = raise_negative
        self.tol = tol
        self.name = name or product.name

    def __call__(self, U, mu=None):
        return norm(U, self.product, raise_negative=self.raise_negative, tol=self.tol)
