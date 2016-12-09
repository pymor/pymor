# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import pytest
import numpy as np

from pymor.algorithms.basic import almost_equal, Norm
from pymortests.fixtures.vectorarray import \
    (vector_array_without_reserve, vector_array, compatible_vector_array_pair_without_reserve,
     compatible_vector_array_pair, incompatible_vector_array_pair)
from pymortests.fixtures.operator import operator_with_arrays_and_products
from pymortests.vectorarray import valid_inds, valid_inds_of_same_length, invalid_ind_pairs, indexed


def test_almost_equal(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if hasattr(v1, 'data'):
        dv1 = v1.data
        dv2 = v2.data
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        for rtol, atol in ((1e-5, 1e-8), (1e-10, 1e-12), (0., 1e-8), (1e-5, 1e-8)):
            for n, o in [('sup', np.inf), ('l1', 1), ('l2', 2)]:
                r = almost_equal(v1[ind1], v2[ind2], norm=n)
                assert isinstance(r, np.ndarray)
                assert r.shape == (v1.len_ind(ind1),)
                if hasattr(v1, 'data'):
                    if dv2.shape[1] == 0:
                        continue
                    assert np.all(r == (np.linalg.norm(indexed(dv1, ind1) - indexed(dv2, ind2), ord=o, axis=1)
                                        <= atol + rtol * np.linalg.norm(indexed(dv2, ind2), ord=o, axis=1)))


def test_almost_equal_product(operator_with_arrays_and_products):
    _, _, v1, _, product, _ = operator_with_arrays_and_products
    if len(v1) < 2:
        return
    v2 = v1.empty()
    v2.append(v1[:len(v1) // 2])
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        for rtol, atol in ((1e-5, 1e-8), (1e-10, 1e-12), (0., 1e-8), (1e-5, 1e-8)):
            norm = Norm(product)

            r = almost_equal(v1[ind1], v2[ind2], norm=norm)
            assert isinstance(r, np.ndarray)
            assert r.shape == (v1.len_ind(ind1),)
            assert np.all(r == (norm(v1[ind1] - v2[ind2])
                                <= atol + rtol * norm(v2[ind2])))

            r = almost_equal(v1[ind1], v2[ind2], product=product)
            assert isinstance(r, np.ndarray)
            assert r.shape == (v1.len_ind(ind1),)
            assert np.all(r == (norm(v1[ind1] - v2[ind2])
                                <= atol + rtol * norm(v2[ind2])))


def test_almost_equal_self(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        for rtol, atol in ((1e-5, 1e-8), (1e-10, 1e-12), (0., 1e-8), (1e-5, 1e-8), (1e-12, 0.)):
            for n in ['sup', 'l1', 'l2']:
                r = almost_equal(v[ind], v[ind], norm=n)
                assert isinstance(r, np.ndarray)
                assert r.shape == (v.len_ind(ind),)
                assert np.all(r)
                if v.len_ind(ind) == 0 or np.max(v[ind].sup_norm() == 0):
                    continue

                c = v.copy()
                c.scal(atol * (1 - 1e-10) / (np.max(getattr(v[ind], n + '_norm')())))
                assert np.all(almost_equal(c[ind], c.zeros(v.len_ind(ind)), atol=atol, rtol=rtol, norm=n))

                if atol > 0:
                    c = v.copy()
                    c.scal(2. * atol / (np.max(getattr(v[ind], n + '_norm')())))
                    assert not np.all(almost_equal(c[ind], c.zeros(v.len_ind(ind)), atol=atol, rtol=rtol, norm=n))

                c = v.copy()
                c.scal(1. + rtol * 0.9)
                assert np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, norm=n))

                if rtol > 0:
                    c = v.copy()
                    c.scal(2. + rtol * 1.1)
                    assert not np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, norm=n))

                c = v.copy()
                c.scal(1. + atol * 0.9 / np.max(getattr(v[ind], n + '_norm')()))
                assert np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, norm=n))

                if atol > 0 or rtol > 0:
                    c = v.copy()
                    c.scal(1 + rtol * 1.1 + atol * 1.1 / np.max(getattr(v[ind], n + '_norm')()))
                    assert not np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, norm=n))


def test_almost_equal_self_product(operator_with_arrays_and_products):
    _, _, v, _, product, _ = operator_with_arrays_and_products
    norm = Norm(product)
    for ind in valid_inds(v):
        for rtol, atol in ((1e-5, 1e-8), (1e-10, 1e-12), (0., 1e-8), (1e-5, 1e-8), (1e-12, 0.)):
            r = almost_equal(v[ind], v[ind], norm=norm)
            assert isinstance(r, np.ndarray)
            assert r.shape == (v.len_ind(ind),)
            assert np.all(r)
            if v.len_ind(ind) == 0 or np.max(v[ind].sup_norm() == 0):
                continue

            r = almost_equal(v[ind], v[ind], product=product)
            assert isinstance(r, np.ndarray)
            assert r.shape == (v.len_ind(ind),)
            assert np.all(r)
            if v.len_ind(ind) == 0 or np.max(v[ind].sup_norm() == 0):
                continue

            c = v.copy()
            c.scal(atol * (1 - 1e-10) / (np.max(norm(v[ind]))))
            assert np.all(almost_equal(c[ind], c.zeros(v.len_ind(ind)), atol=atol, rtol=rtol, norm=norm))
            assert np.all(almost_equal(c[ind], c.zeros(v.len_ind(ind)), atol=atol, rtol=rtol, product=product))

            if atol > 0:
                c = v.copy()
                c.scal(2. * atol / (np.max(norm(v[ind]))))
                assert not np.all(almost_equal(c[ind], c.zeros(v.len_ind(ind)), atol=atol, rtol=rtol, norm=norm))
                assert not np.all(almost_equal(c[ind], c.zeros(v.len_ind(ind)), atol=atol, rtol=rtol, product=product))

            c = v.copy()
            c.scal(1. + rtol * 0.9)
            assert np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, norm=norm))
            assert np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, product=product))

            if rtol > 0:
                c = v.copy()
                c.scal(2. + rtol * 1.1)
                assert not np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, norm=norm))
                assert not np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, product=product))

            c = v.copy()
            c.scal(1. + atol * 0.9 / np.max(np.max(norm(v[ind]))))
            assert np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, norm=norm))
            assert np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, product=product))

            if atol > 0 or rtol > 0:
                c = v.copy()
                c.scal(1 + rtol * 1.1 + atol * 1.1 / np.max(np.max(norm(v[ind]))))
                assert not np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, norm=norm))
                assert not np.all(almost_equal(c[ind], v[ind], atol=atol, rtol=rtol, product=product))


def test_almost_equal_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        for n in ['sup', 'l1', 'l2']:
            c1, c2 = v1.copy(), v2.copy()
            with pytest.raises(Exception):
                almost_equal(c1[ind1], c2[ind2], norm=n)


def test_almost_equal_wrong_ind(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        for n in ['sup', 'l1', 'l2']:
            if (ind1 is None and len(v1) == 1 or isinstance(ind1, Number) or hasattr(ind1, '__len__') and len(ind1) == 1 or
                ind2 is None and len(v2) == 1 or isinstance(ind2, Number) or hasattr(ind2, '__len__') and len(ind2) == 1):  # NOQA
                continue
            c1, c2 = v1.copy(), v2.copy()
            with pytest.raises(Exception):
                almost_equal(c1[ind], c2[ind], norm=n)
