
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.projection import project
from pymortests.fixtures.operator import operator_with_arrays, operator_with_arrays_and_products


def test_project(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    op_UV = project(op, V, U)
    np.random.seed(4711 + U.dim + len(V))
    coeffs = np.random.random(len(U))
    X = op_UV.apply(op_UV.source.make_array(coeffs), mu=mu)
    Y = op_UV.range.make_array(V.inner(op.apply(U.lincomb(coeffs), mu=mu)).T)
    assert np.all(almost_equal(X, Y))


def test_project_2(operator_with_arrays):
    op, mu, U, V = operator_with_arrays
    op_U = project(op, None, U)
    op_V = project(op, V, None)
    op_U_V = project(op_U, V, None)
    op_V_U = project(op_V, None, U)
    op_UV = project(op, V, U)
    np.random.seed(4711 + U.dim + len(V))
    W = op_UV.source.make_array(np.random.random(len(U)))
    Y0 = op_UV.apply(W, mu=mu)
    Y1 = op_U_V.apply(W, mu=mu)
    Y2 = op_V_U.apply(W, mu=mu)
    assert np.all(almost_equal(Y0, Y1))
    assert np.all(almost_equal(Y0, Y2))


def test_project_with_product(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    op_UV = project(op, V, U, product=rp)
    np.random.seed(4711 + U.dim + len(V))
    coeffs = np.random.random(len(U))
    X = op_UV.apply(op_UV.source.make_array(coeffs), mu=mu)
    Y = op_UV.range.make_array(rp.apply2(op.apply(U.lincomb(coeffs), mu=mu), V))
    assert np.all(almost_equal(X, Y))


def test_project_with_product_2(operator_with_arrays_and_products):
    op, mu, U, V, sp, rp = operator_with_arrays_and_products
    op_U = project(op, None, U)
    op_V = project(op, V, None, product=rp)
    op_U_V = project(op_U, V, None, product=rp)
    op_V_U = project(op_V, None, U)
    op_UV = project(op, V, U, product=rp)
    np.random.seed(4711 + U.dim + len(V))
    W = op_UV.source.make_array(np.random.random(len(U)))
    Y0 = op_UV.apply(W, mu=mu)
    Y1 = op_U_V.apply(W, mu=mu)
    Y2 = op_V_U.apply(W, mu=mu)
    assert np.all(almost_equal(Y0, Y1))
    assert np.all(almost_equal(Y0, Y2))
