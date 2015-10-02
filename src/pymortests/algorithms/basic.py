# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import pytest
import numpy as np

from pymor.algorithms.basic import almost_equal
from pymortests.fixtures.vectorarray import \
    (vector_array_without_reserve, vector_array, compatible_vector_array_pair_without_reserve,
     compatible_vector_array_pair, incompatible_vector_array_pair)
from pymortests.vectorarray import valid_inds, valid_inds_of_same_length, invalid_ind_pairs, indexed


def test_almost_equal(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if hasattr(v1, 'data'):
        dv1 = v1.data
        dv2 = v2.data
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        for rtol, atol in ((1e-5, 1e-8), (1e-10, 1e-12), (0., 1e-8), (1e-5, 1e-8)):
            r = almost_equal(v1, v2, U_ind=ind1, V_ind=ind2, norm='sup')
            assert isinstance(r, np.ndarray)
            assert r.shape == (v1.len_ind(ind1),)
            if hasattr(v1, 'data'):
                if dv2.shape[1] == 0:
                    continue
                assert np.all(r == (np.max(np.abs(indexed(dv1, ind1) - indexed(dv2, ind2)), axis=1)
                                    <= atol + rtol * np.max(np.abs(indexed(dv2, ind2)), axis=1)))


def test_almost_equal_self(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        for rtol, atol in ((1e-5, 1e-8), (1e-10, 1e-12), (0., 1e-8), (1e-5, 1e-8), (1e-12, 0.)):
            r = almost_equal(v, v, U_ind=ind, V_ind=ind)
            assert isinstance(r, np.ndarray)
            assert r.shape == (v.len_ind(ind),)
            assert np.all(r)
            if v.len_ind(ind) == 0 or np.max(v.sup_norm(ind) == 0):
                continue

            c = v.copy()
            c.scal(atol / (np.max(v.sup_norm(ind))))
            assert np.all(almost_equal(c, c.zeros(v.len_ind(ind)), U_ind=ind, atol=atol, rtol=rtol, norm='sup'))

            if atol > 0:
                c = v.copy()
                c.scal(2. * atol / (np.max(v.sup_norm(ind))))
                assert not np.all(almost_equal(c, c.zeros(v.len_ind(ind)), U_ind=ind, atol=atol, rtol=rtol, norm='sup'))

            c = v.copy()
            c.scal(1. + rtol * 0.9)
            assert np.all(almost_equal(c, v, U_ind=ind, V_ind=ind, atol=atol, rtol=rtol, norm='sup'))

            if rtol > 0:
                c = v.copy()
                c.scal(2. + rtol * 1.1)
                assert not np.all(almost_equal(c, v, U_ind=ind, V_ind=ind, atol=atol, rtol=rtol, norm='sup'))

            c = v.copy()
            c.scal(1. + atol * 0.9 / np.max(v.sup_norm(ind)))
            assert np.all(almost_equal(c, v, U_ind=ind, V_ind=ind, atol=atol, rtol=rtol, norm='sup'))

            if atol > 0 or rtol > 0:
                c = v.copy()
                c.scal(1 + rtol * 1.1 + atol * 1.1 / np.max(v.sup_norm(ind)))
                assert not np.all(almost_equal(c, v, U_ind=ind, V_ind=ind, atol=atol, rtol=rtol, norm='sup'))


def test_almost_equal_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            almost_equal(c1, c2, U_ind=ind1, V_ind=ind2)


def test_almost_equal_wrong_ind(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        if (ind1 is None and len(v1) == 1 or isinstance(ind1, Number) or hasattr(ind1, '__len__') and len(ind1) == 1 or
            ind2 is None and len(v2) == 1 or isinstance(ind2, Number) or hasattr(ind2, '__len__') and len(ind2) == 1):  # NOQA
            continue
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            almost_equal(c1, c2, U_ind=ind1, V_ind=ind2)
