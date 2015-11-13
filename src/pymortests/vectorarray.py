# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import product, chain, izip
from numbers import Number

import pytest
import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.core import NUMPY_INDEX_QUIRK
from pymor.vectorarrays.interfaces import VectorSpace
from pymortests.fixtures.vectorarray import \
    (vector_array_without_reserve, vector_array, compatible_vector_array_pair_without_reserve,
     compatible_vector_array_pair, incompatible_vector_array_pair,
     picklable_vector_array_without_reserve, picklable_vector_array)
from pymortests.pickle import assert_picklable_without_dumps_function

pytestmark = pytest.mark.slow

def ind_complement(v, ind):
    if ind is None:
        return []
    if isinstance(ind, Number):
        ind = [ind]
    return sorted(set(xrange(len(v))) - set(ind))


def indexed(v, ind):
    if ind is None:
        return v
    elif isinstance(ind, Number):
        return v[[ind]]
    elif len(ind) == 0:
        return np.empty((0, v.shape[1]), dtype=v.dtype)
    else:
        return v[ind]


def invalid_inds(v, length=None):
    if length is None:
        yield len(v)
        yield [len(v)]
        yield -1
        yield [-1]
        yield [0, len(v)]
        length = 42
    if length > 0:
        yield [-1] + [0, ] * (length - 1)
        yield range(length - 1) + [len(v)]


def valid_inds(v, length=None):
    if length is None:
        for ind in [None, [], range(len(v)), range(int(len(v)/2)), range(len(v)) * 2]:
            yield ind
        length = 32
    if len(v) > 0:
        for ind in [0, len(v) - 1]:
            yield ind
        if len(v) == length:
            yield None
        np.random.seed(len(v) * length)
        yield list(np.random.randint(0, len(v), size=length))
        yield list(np.random.randint(0, len(v), size=length))
    else:
        if len(v) == 0:
            yield None
        yield []


def valid_inds_of_same_length(v1, v2):
    if len(v1) == len(v2):
        yield None, None
        yield range(len(v1)), range(len(v1))
    yield [], []
    if len(v1) > 0 and len(v2) > 0:
        yield 0, 0
        yield len(v1) - 1, len(v2) - 1
        yield [0], 0
        yield (range(int(min(len(v1), len(v2))/2)),) * 2
        np.random.seed(len(v1) * len(v2))
        for count in np.linspace(0, min(len(v1), len(v2)), 3):
            yield (list(np.random.randint(0, len(v1), size=count)),
                   list(np.random.randint(0, len(v2), size=count)))
        yield None, np.random.randint(0, len(v2), size=len(v1))
        yield np.random.randint(0, len(v1), size=len(v2)), None


def valid_inds_of_different_length(v1, v2):
    if len(v1) != len(v2):
        yield None, None
        yield range(len(v1)), range(len(v2))
    if len(v1) > 0 and len(v2) > 0:
        if len(v1) > 1:
            yield [0, 1], 0
            yield [0, 1], [0]
        if len(v2) > 1:
            yield 0, [0, 1]
            yield [0], [0, 1]
        np.random.seed(len(v1) * len(v2))
        for count1 in np.linspace(0, len(v1), 3).astype(int):
            count2 = np.random.randint(0, len(v2))
            if count2 == count1:
                count2 += 1
                if count2 == len(v2):
                    count2 -= 2
            if count2 >= 0:
                yield (list(np.random.randint(0, len(v1), size=count1)),
                       list(np.random.randint(0, len(v2), size=count2)))


def invalid_ind_pairs(v1, v2):
    for inds in valid_inds_of_different_length(v1, v2):
        yield inds
    for ind1 in valid_inds(v1):
        for ind2 in invalid_inds(v2, length=v1.len_ind(ind1)):
            yield ind1, ind2
    for ind2 in valid_inds(v2):
        for ind1 in invalid_inds(v1, length=v2.len_ind(ind2)):
            yield ind1, ind2


def test_empty(vector_array):
    with pytest.raises(Exception):
        vector_array.empty(-1)
    for r in (0, 1, 100):
        v = vector_array.empty(reserve=r)
        assert v.dim == vector_array.dim
        assert v.subtype == vector_array.subtype
        assert v.space == vector_array.space
        assert len(v) == 0
        if hasattr(v, 'data'):
            d = v.data
            assert d.shape == (0, v.dim)


def test_zeros(vector_array):
    with pytest.raises(Exception):
        vector_array.zeros(-1)
    for c in (0, 1, 2, 30):
        v = vector_array.zeros(count=c)
        assert v.dim == vector_array.dim
        assert v.subtype == vector_array.subtype
        assert v.space == vector_array.space
        assert len(v) == c
        if min(v.dim, c) > 0:
            assert max(v.sup_norm()) == 0
            assert max(v.l2_norm()) == 0
        if hasattr(v, 'data'):
            d = v.data
            assert d.shape == (c, v.dim)
            assert np.allclose(d, np.zeros((c, v.dim)))


def test_shape(vector_array):
    v = vector_array
    assert len(vector_array) >= 0
    assert vector_array.dim >= 0
    if hasattr(v, 'data'):
        d = v.data
        assert d.shape == (len(v), v.dim)


def test_space(vector_array):
    v = vector_array
    assert isinstance(v.space, VectorSpace)
    assert v.dim == v.space.dim
    assert v.subtype == v.space.subtype
    assert type(v) == v.space.type
    assert v in v.space


def test_copy(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        c = v.copy(ind=ind)
        assert len(c) == v.len_ind(ind)
        assert c.dim == v.dim
        assert c.subtype == v.subtype
        assert np.all(almost_equal(c, v, V_ind=ind))
        if hasattr(v, 'data'):
            dv = v.data
            dc = c.data
            assert np.allclose(dc, indexed(dv, ind))


def test_copy_repeated_index(vector_array):
    v = vector_array
    if len(v) == 0:
        return
    ind = [int(len(vector_array) * 3 / 4)] * 2
    c = v.copy(ind)
    assert almost_equal(c, v, U_ind=0, V_ind=ind[0])
    assert almost_equal(c, v, U_ind=1, V_ind=ind[0])
    if hasattr(v, 'data'):
        dv = indexed(v.data, ind)
        dc = c.data
        assert dv.shape == dc.shape
    c.scal(2., ind=0)
    assert almost_equal(c, v, U_ind=1, V_ind=ind[0])
    assert c.l2_norm(ind=0) == 2 * v.l2_norm(ind=ind[0])
    if hasattr(v, 'data'):
        dv = indexed(v.data, ind)
        dc = c.data
        assert dv.shape == dc.shape


def test_append(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if hasattr(v1, 'data'):
        dv1 = v1.data
        dv2 = v2.data
    len_v1, len_v2 = len(v1), len(v2)
    for ind in valid_inds(v2):
        c1, c2 = v1.copy(), v2.copy()
        c1.append(c2, o_ind=ind)
        len_ind = v2.len_ind(ind)
        ind_complement_ = ind_complement(v2, ind)
        assert len(c1) == len_v1 + len_ind
        assert np.all(almost_equal(c1, c2, U_ind=range(len_v1, len(c1)), V_ind=ind))
        if hasattr(v1, 'data'):
            assert np.allclose(c1.data, np.vstack((dv1, indexed(dv2, ind))))
        c1.append(c2, o_ind=ind, remove_from_other=True)
        assert len(c2) == len(ind_complement_)
        assert c2.dim == c1.dim
        assert c2.subtype == c1.subtype
        assert len(c1) == len_v1 + 2 * len_ind
        assert np.all(almost_equal(c1, c1, U_ind=range(len_v1, len_v1 + len_ind), V_ind=range(len_v1 + len_ind, len(c1))))
        assert np.all(almost_equal(c2, v2, V_ind=ind_complement_))
        if hasattr(v1, 'data'):
            assert np.allclose(c2.data, indexed(dv2, ind_complement_))


def test_append_self(vector_array):
    v = vector_array
    c = v.copy()
    len_v = len(v)
    c.append(c)
    assert len(c) == 2 * len_v
    assert np.all(almost_equal(c, c, U_ind=range(len_v), V_ind=range(len_v, len(c))))
    if hasattr(v, 'data'):
        assert np.allclose(c.data, np.vstack((v.data, v.data)))
    c = v.copy()
    with pytest.raises(Exception):
        v.append(v, remove_from_other=True)


def test_remove(vector_array):
    v = vector_array
    if hasattr(v, 'data'):
        dv = v.data
    for ind in valid_inds(v):
        ind_complement_ = ind_complement(v, ind)
        c = v.copy()
        c.remove(ind)
        assert c.dim == v.dim
        assert c.subtype == v.subtype
        assert len(c) == len(ind_complement_)
        assert np.all(almost_equal(v, c, U_ind=ind_complement_))
        if hasattr(v, 'data'):
            assert np.allclose(c.data, indexed(dv, ind_complement_))
        c.remove()
        assert len(c) == 0


def test_replace(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    len_v1, len_v2 = len(v1), len(v2)
    if hasattr(v1, 'data'):
        dv1 = v1.data
        dv2 = v2.data
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        if v1.len_ind(ind1) != v1.len_ind_unique(ind1):
            with pytest.raises(Exception):
                c1, c2 = v1.copy(), v2.copy()
                c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=False)
            with pytest.raises(Exception):
                c1, c2 = v1.copy(), v2.copy()
                c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=True)
            continue
        c1, c2 = v1.copy(), v2.copy()
        c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=False)
        assert len(c1) == len(v1)
        assert c1.dim == v1.dim
        assert c1.subtype == v1.subtype
        assert np.all(almost_equal(c1, v2, U_ind=ind1, V_ind=ind2))
        assert np.all(almost_equal(c2, v2))
        if hasattr(v1, 'data'):
            x = dv1.copy()
            if NUMPY_INDEX_QUIRK and len(x) == 0 and hasattr(ind1, '__len__') and len(ind1) == 0:
                pass
            else:
                x[ind1] = indexed(dv2, ind2)
            assert np.allclose(c1.data, x)

        c1, c2 = v1.copy(), v2.copy()
        c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=True)
        assert len(c1) == len(v1)
        assert c1.dim == v1.dim
        assert c1.subtype == v1.subtype
        ind2_complement = ind_complement(v2, ind2)
        assert np.all(almost_equal(c1, v2, U_ind=ind1, V_ind=ind2))
        assert len(c2) == len(ind2_complement)
        assert np.all(almost_equal(c2, v2, V_ind=ind2_complement))
        if hasattr(v1, 'data'):
            x = dv1.copy()
            if NUMPY_INDEX_QUIRK and len(x) == 0 and hasattr(ind1, '__len__') and len(ind1) == 0:
                pass
            else:
                x[ind1] = indexed(dv2, ind2)
            assert np.allclose(c1.data, x)
            assert np.allclose(c2.data, indexed(dv2, ind2_complement))


def test_replace_self(vector_array):
    v = vector_array
    if hasattr(v, 'data'):
        dv = v.data
    for ind1, ind2 in valid_inds_of_same_length(v, v):
        if v.len_ind(ind1) != v.len_ind_unique(ind1):
            c = v.copy()
            with pytest.raises(Exception):
                c.replace(c, ind=ind1, o_ind=ind2, remove_from_other=False)
            c = v.copy()
            with pytest.raises(Exception):
                c.replace(c, ind=ind1, o_ind=ind2, remove_from_other=True)
            continue

        c = v.copy()
        with pytest.raises(Exception):
            c.replace(c, ind=ind1, o_ind=ind2, remove_from_other=True)

        c = v.copy()
        c.replace(c, ind=ind1, o_ind=ind2, remove_from_other=False)
        assert len(c) == len(v)
        assert c.dim == v.dim
        assert c.subtype == v.subtype
        assert np.all(almost_equal(c, v, U_ind=ind1, V_ind=ind2))
        if hasattr(v, 'data'):
            x = dv.copy()
            if NUMPY_INDEX_QUIRK and len(x) == 0 and hasattr(ind1, '__len__') and len(ind1) == 0:
                pass
            else:
                x[ind1] = indexed(dv, ind2)
            assert np.allclose(c.data, x)


def test_scal(vector_array):
    v = vector_array
    if hasattr(v, 'data'):
        dv = v.data
    for ind in valid_inds(v):
        if v.len_ind(ind) != v.len_ind_unique(ind):
            with pytest.raises(Exception):
                c = v.copy()
                c.scal(1., ind=ind)
            continue
        ind_complement_ = ind_complement(v, ind)
        c = v.copy()
        c.scal(1., ind=ind)
        assert len(c) == len(v)
        assert np.all(almost_equal(c, v))

        c = v.copy()
        c.scal(0., ind=ind)
        assert np.all(almost_equal(c, v.zeros(v.len_ind(ind)), U_ind=ind))
        assert np.all(almost_equal(c, v, U_ind=ind_complement_, V_ind=ind_complement_))

        for x in (1., 1.4, np.random.random(v.len_ind(ind))):
            c = v.copy()
            c.scal(x, ind=ind)
            assert np.all(almost_equal(c, v, U_ind=ind_complement_, V_ind=ind_complement_))
            assert np.allclose(c.sup_norm(ind), v.sup_norm(ind) * abs(x))
            assert np.allclose(c.l2_norm(ind), v.l2_norm(ind) * abs(x))
            if hasattr(v, 'data'):
                y = dv.copy()
                if NUMPY_INDEX_QUIRK and len(y) == 0:
                    pass
                else:
                    if isinstance(x, np.ndarray) and not isinstance(ind, Number):
                        x = x[:, np.newaxis]
                    y[ind] *= x
                assert np.allclose(c.data, y)


def test_axpy(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if hasattr(v1, 'data'):
        dv1 = v1.data
        dv2 = v2.data

    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        if v1.len_ind(ind1) != v1.len_ind_unique(ind1):
            with pytest.raises(Exception):
                c1, c2 = v1.copy(), v2.copy()
                c1.axpy(0., c2, ind=ind1, x_ind=ind2)
            continue

        ind1_complement = ind_complement(v1, ind1)
        c1, c2 = v1.copy(), v2.copy()
        c1.axpy(0., c2, ind=ind1, x_ind=ind2)
        assert len(c1) == len(v1)
        assert np.all(almost_equal(c1, v1))
        assert np.all(almost_equal(c2, v2))

        np.random.seed(len(v1) + 39)
        for a in (1., 1.4, np.random.random(v1.len_ind(ind1))):
            c1, c2 = v1.copy(), v2.copy()
            c1.axpy(a, c2, ind=ind1, x_ind=ind2)
            assert len(c1) == len(v1)
            assert np.all(almost_equal(c1, v1, U_ind=ind1_complement, V_ind=ind1_complement))
            assert np.all(almost_equal(c2, v2))
            assert np.all(c1.sup_norm(ind1) <= v1.sup_norm(ind1) + abs(a) * v2.sup_norm(ind2) * (1. + 1e-10))
            assert np.all(c1.l1_norm(ind1) <= (v1.l1_norm(ind1) + abs(a) * v2.l1_norm(ind2)) * (1. + 1e-10))
            assert np.all(c1.l2_norm(ind1) <= (v1.l2_norm(ind1) + abs(a) * v2.l2_norm(ind2)) * (1. + 1e-10))
            if hasattr(v1, 'data'):
                x = dv1.copy()
                if isinstance(ind1, Number):
                    x[[ind1]] += indexed(dv2, ind2) * a
                else:
                    if NUMPY_INDEX_QUIRK and len(x) == 0:
                        pass
                    else:
                        if isinstance(a, np.ndarray):
                            aa = a[:, np.newaxis]
                        else:
                            aa = a
                        x[ind1] += indexed(dv2, ind2) * aa
                assert np.allclose(c1.data, x)
            c1.axpy(-a, c2, ind=ind1, x_ind=ind2)
            assert len(c1) == len(v1)
            assert np.all(almost_equal(c1, v1))


def test_axpy_one_x(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if hasattr(v1, 'data'):
        dv1 = v1.data
        dv2 = v2.data

    for ind1, ind2 in product(valid_inds(v1), valid_inds(v2, 1)):
        if v1.len_ind(ind1) != v1.len_ind_unique(ind1):
            with pytest.raises(Exception):
                c1, c2 = v1.copy(), v2.copy()
                c1.axpy(0., c2, ind=ind1, x_ind=ind2)
            continue

        ind1_complement = ind_complement(v1, ind1)
        c1, c2 = v1.copy(), v2.copy()
        c1.axpy(0., c2, ind=ind1, x_ind=ind2)
        assert len(c1) == len(v1)
        assert np.all(almost_equal(c1, v1))
        assert np.all(almost_equal(c2, v2))

        np.random.seed(len(v1) + 39)
        for a in (1., 1.4, np.random.random(v1.len_ind(ind1))):
            c1, c2 = v1.copy(), v2.copy()
            c1.axpy(a, c2, ind=ind1, x_ind=ind2)
            assert len(c1) == len(v1)
            assert np.all(almost_equal(c1, v1, U_ind=ind1_complement, V_ind=ind1_complement))
            assert np.all(almost_equal(c2, v2))
            assert np.all(c1.sup_norm(ind1) <= v1.sup_norm(ind1) + abs(a) * v2.sup_norm(ind2) * (1. + 1e-10))
            assert np.all(c1.l1_norm(ind1) <= (v1.l1_norm(ind1) + abs(a) * v2.l1_norm(ind2)) * (1. + 1e-10))
            assert np.all(c1.l2_norm(ind1) <= (v1.l2_norm(ind1) + abs(a) * v2.l2_norm(ind2)) * (1. + 1e-10))
            if hasattr(v1, 'data'):
                x = dv1.copy()
                if isinstance(ind1, Number):
                    x[[ind1]] += indexed(dv2, ind2) * a
                else:
                    if NUMPY_INDEX_QUIRK and len(x) == 0:
                        pass
                    else:
                        if isinstance(a, np.ndarray):
                            aa = a[:, np.newaxis]
                        else:
                            aa = a
                        x[ind1] += indexed(dv2, ind2) * aa
                assert np.allclose(c1.data, x)
            c1.axpy(-a, c2, ind=ind1, x_ind=ind2)
            assert len(c1) == len(v1)
            assert np.all(almost_equal(c1, v1))


def test_axpy_self(vector_array):
    v = vector_array
    if hasattr(v, 'data'):
        dv = v.data

    for ind1, ind2 in valid_inds_of_same_length(v, v):
        if v.len_ind(ind1) != v.len_ind_unique(ind1):
            with pytest.raises(Exception):
                c, = v.copy()
                c.axpy(0., c, ind=ind1, x_ind=ind2)
            continue

        ind1_complement = ind_complement(v, ind1)
        c = v.copy()
        c.axpy(0., c, ind=ind1, x_ind=ind2)
        assert len(c) == len(v)
        assert np.all(almost_equal(c, v))
        assert np.all(almost_equal(c, v))

        np.random.seed(len(v) + 8)
        for a in (1., 1.4, np.random.random(v.len_ind(ind1))):
            c = v.copy()
            c.axpy(a, c, ind=ind1, x_ind=ind2)
            assert len(c) == len(v)
            assert np.all(almost_equal(c, v, U_ind=ind1_complement, V_ind=ind1_complement))
            assert np.all(c.sup_norm(ind1) <= v.sup_norm(ind1) + abs(a) * v.sup_norm(ind2) * (1. + 1e-10))
            assert np.all(c.l1_norm(ind1) <= (v.l1_norm(ind1) + abs(a) * v.l1_norm(ind2)) * (1. + 1e-10))
            if hasattr(v, 'data'):
                x = dv.copy()
                if isinstance(ind1, Number):
                    x[[ind1]] += indexed(dv, ind2) * a
                else:
                    if NUMPY_INDEX_QUIRK and len(x) == 0:
                        pass
                    else:
                        if isinstance(a, np.ndarray):
                            aa = a[:, np.newaxis]
                        else:
                            aa = a
                        x[ind1] += indexed(dv, ind2) * aa
                assert np.allclose(c.data, x)
            c.axpy(-a, v, ind=ind1, x_ind=ind2)
            assert len(c) == len(v)
            assert np.all(almost_equal(c, v))

    for ind in valid_inds(v):
        if v.len_ind(ind) != v.len_ind_unique(ind):
            continue

        for x in (1., 23., -4):
            c = v.copy()
            cc = v.copy()
            c.axpy(x, c, ind=ind, x_ind=ind)
            cc.scal(1 + x, ind=ind)
            assert np.all(almost_equal(c, cc))


def test_pairwise_dot(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if hasattr(v1, 'data'):
        dv1, dv2 = v1.data, v2.data
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        r = v1.pairwise_dot(v2, ind=ind1, o_ind=ind2)
        assert isinstance(r, np.ndarray)
        assert r.shape == (v1.len_ind(ind1),)
        r2 = v2.pairwise_dot(v1, ind=ind2, o_ind=ind1)
        assert np.all(r == r2)
        assert np.all(r <= v1.l2_norm(ind1) * v2.l2_norm(ind2) * (1. + 1e-10))
        if hasattr(v1, 'data'):
            assert np.allclose(r, np.sum(indexed(dv1, ind1) * indexed(dv2, ind2), axis=1))


def test_pairwise_dot_self(vector_array):
    v = vector_array
    if hasattr(v, 'data'):
        dv = v.data
    for ind1, ind2 in valid_inds_of_same_length(v, v):
        r = v.pairwise_dot(v, ind=ind1, o_ind=ind2)
        assert isinstance(r, np.ndarray)
        assert r.shape == (v.len_ind(ind1),)
        r2 = v.pairwise_dot(v, ind=ind2, o_ind=ind1)
        assert np.all(r == r2)
        assert np.all(r <= v.l2_norm(ind1) * v.l2_norm(ind2) * (1. + 1e-10))
        if hasattr(v, 'data'):
            assert np.allclose(r, np.sum(indexed(dv, ind1) * indexed(dv, ind2), axis=1))
    for ind in valid_inds(v):
        r = v.pairwise_dot(v, ind=ind, o_ind=ind)
        assert np.allclose(r, v.l2_norm(ind) ** 2)


def test_dot(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if hasattr(v1, 'data'):
        dv1, dv2 = v1.data, v2.data
    for ind1, ind2 in chain(valid_inds_of_different_length(v1, v2), valid_inds_of_same_length(v1, v2)):
        r = v1.dot(v2, ind=ind1, o_ind=ind2)
        assert isinstance(r, np.ndarray)
        assert r.shape == (v1.len_ind(ind1), v2.len_ind(ind2))
        r2 = v2.dot(v1, ind=ind2, o_ind=ind1)
        assert np.all(r == r2.T)
        assert np.all(r <= v1.l2_norm(ind1)[:, np.newaxis] * v2.l2_norm(ind2)[np.newaxis, :] * (1. + 1e-10))
        if hasattr(v1, 'data'):
            assert np.allclose(r, indexed(dv1, ind1).dot(indexed(dv2, ind2).T))


def test_dot_self(vector_array):
    v = vector_array
    if hasattr(v, 'data'):
        dv = v.data
    for ind1, ind2 in chain(valid_inds_of_different_length(v, v), valid_inds_of_same_length(v, v)):
        r = v.dot(v, ind=ind1, o_ind=ind2)
        assert isinstance(r, np.ndarray)
        assert r.shape == (v.len_ind(ind1), v.len_ind(ind2))
        r2 = v.dot(v, ind=ind2, o_ind=ind1)
        assert np.all(r == r2.T)
        assert np.all(r <= v.l2_norm(ind1)[:, np.newaxis] * v.l2_norm(ind2)[np.newaxis, :] * (1. + 1e-10))
        if hasattr(v, 'data'):
            assert np.allclose(r, indexed(dv, ind1).dot(indexed(dv, ind2).T))
    for ind in valid_inds(v):
        r = v.dot(v, ind=ind, o_ind=ind)
        assert np.all(r == r.T)


def test_lincomb_1d(vector_array):
    v = vector_array
    np.random.seed(len(v) + 42 + v.dim)
    for ind in valid_inds(v):
        coeffs = np.random.random(v.len_ind(ind))
        lc = v.lincomb(coeffs, ind=ind)
        assert lc.dim == v.dim
        assert lc.subtype == v.subtype
        assert len(lc) == 1
        lc2 = v.zeros()
        ind = range(len(v)) if ind is None else [ind] if isinstance(ind, Number) else ind
        for coeff, i in zip(coeffs, ind):
            lc2.axpy(coeff, v, x_ind=i)
        assert np.all(almost_equal(lc, lc2))


def test_lincomb_2d(vector_array):
    v = vector_array
    np.random.seed(len(v) + 42 + v.dim)
    for ind in valid_inds(v):
        for count in (0, 1, 5):
            coeffs = np.random.random((count, v.len_ind(ind)))
            lc = v.lincomb(coeffs, ind=ind)
            assert lc.dim == v.dim
            assert lc.subtype == v.subtype
            assert len(lc) == count
            lc2 = v.empty(reserve=count)
            for coeffs_1d in coeffs:
                lc2.append(v.lincomb(coeffs_1d, ind=ind))
            assert np.all(almost_equal(lc, lc2))


def test_lincomb_wrong_coefficients(vector_array):
    v = vector_array
    np.random.seed(len(v) + 42 + v.dim)
    for ind in valid_inds(v):
        coeffs = np.random.random(v.len_ind(ind) + 1)
        with pytest.raises(Exception):
            v.lincomb(coeffs, ind=ind)
        coeffs = np.random.random(v.len_ind(ind)).reshape((1, 1, -1))
        with pytest.raises(Exception):
            v.lincomb(coeffs, ind=ind)
        if v.len_ind(ind) > 0:
            coeffs = np.random.random(v.len_ind(ind) - 1)
            with pytest.raises(Exception):
                v.lincomb(coeffs, ind=ind)
            coeffs = np.array([])
            with pytest.raises(Exception):
                v.lincomb(coeffs, ind=ind)


def test_l1_norm(vector_array):
    v = vector_array
    if hasattr(v, 'data'):
        dv = v.data
    for ind in valid_inds(v):
        c = v.copy()
        norm = c.l1_norm(ind)
        assert isinstance(norm, np.ndarray)
        assert norm.shape == (v.len_ind(ind),)
        assert np.all(norm >= 0)
        if v.dim == 0:
            assert np.all(norm == 0)
        if hasattr(v, 'data'):
            assert np.allclose(norm, np.sum(np.abs(indexed(dv, ind)), axis=1))
        c.scal(4.)
        assert np.allclose(c.l1_norm(ind), norm * 4)
        c.scal(-4.)
        assert np.allclose(c.l1_norm(ind), norm * 16)
        c.scal(0.)
        assert np.allclose(c.l1_norm(ind), 0)


def test_l2_norm(vector_array):
    v = vector_array
    if hasattr(v, 'data'):
        dv = v.data
    for ind in valid_inds(v):
        c = v.copy()
        norm = c.l2_norm(ind)
        assert isinstance(norm, np.ndarray)
        assert norm.shape == (v.len_ind(ind),)
        assert np.all(norm >= 0)
        if v.dim == 0:
            assert np.all(norm == 0)
        if hasattr(v, 'data'):
            assert np.allclose(norm, np.sqrt(np.sum(np.power(indexed(dv, ind), 2), axis=1)))
        c.scal(4.)
        assert np.allclose(c.l2_norm(ind), norm * 4)
        c.scal(-4.)
        assert np.allclose(c.l2_norm(ind), norm * 16)
        c.scal(0.)
        assert np.allclose(c.l2_norm(ind), 0)


def test_sup_norm(vector_array):
    v = vector_array
    if hasattr(v, 'data'):
        dv = v.data
    for ind in valid_inds(v):
        c = v.copy()
        norm = c.sup_norm(ind)
        assert isinstance(norm, np.ndarray)
        assert norm.shape == (v.len_ind(ind),)
        assert np.all(norm >= 0)
        if v.dim == 0:
            assert np.all(norm == 0)
        if hasattr(v, 'data') and v.dim > 0:
            assert np.allclose(norm, np.max(np.abs(indexed(dv, ind)), axis=1))
        c.scal(4.)
        assert np.allclose(c.sup_norm(ind), norm * 4)
        c.scal(-4.)
        assert np.allclose(c.sup_norm(ind), norm * 16)
        c.scal(0.)
        assert np.allclose(c.sup_norm(ind), 0)


def test_components(vector_array):
    v = vector_array
    np.random.seed(len(v) + 24 + v.dim)
    if hasattr(v, 'data'):
        dv = v.data
    for ind in valid_inds(v):
        c = v.copy()
        comp = c.components(np.array([], dtype=np.int), ind=ind)
        assert isinstance(comp, np.ndarray)
        assert comp.shape == (v.len_ind(ind), 0)

        c = v.copy()
        comp = c.components([], ind=ind)
        assert isinstance(comp, np.ndarray)
        assert comp.shape == (v.len_ind(ind), 0)

        if v.dim > 0:
            for count in (1, 5, 10):
                c_ind = np.random.randint(0, v.dim, count)
                c = v.copy()
                comp = c.components(c_ind, ind=ind)
                assert comp.shape == (v.len_ind(ind), count)
                c = v.copy()
                comp2 = c.components(list(c_ind), ind=ind)
                assert np.all(comp == comp2)
                c = v.copy()
                c.scal(3.)
                comp2 = c.components(c_ind, ind=ind)
                assert np.allclose(comp * 3, comp2)
                c = v.copy()
                comp2 = c.components(np.hstack((c_ind, c_ind)), ind=ind)
                assert np.all(comp2 == np.hstack((comp, comp)))
                if hasattr(v, 'data'):
                    assert np.all(comp == indexed(dv, ind)[:, c_ind])


def test_components_wrong_component_indices(vector_array):
    v = vector_array
    np.random.seed(len(v) + 24 + v.dim)
    for ind in valid_inds(v):
        with pytest.raises(Exception):
            v.components(None, ind=ind)
        with pytest.raises(Exception):
            v.components(1, ind=ind)
        with pytest.raises(Exception):
            v.components(np.array([-1]), ind=ind)
        with pytest.raises(Exception):
            v.components(np.array([v.dim]), ind=ind)


def test_amax(vector_array):
    v = vector_array
    if v.dim == 0:
        return
    for ind in valid_inds(v):
        max_inds, max_vals = v.amax(ind)
        assert np.allclose(np.abs(max_vals), v.sup_norm(ind))
        if ind is None:
            ind = xrange(len(v))
        elif isinstance(ind, Number):
            ind = [ind]
        for i, max_ind, max_val in zip(ind, max_inds, max_vals):
            assert np.allclose(max_val, v.components([max_ind], ind=[i]))


# def test_amax_zero_dim(zero_dimensional_vector_space):
#     for count in (0, 10):
#         v = zero_dimensional_vector_space.zeros(count=count)
#         for ind in valid_inds(v):
#             with pytest.raises(Exception):
#                 v.amax(ind)


def test_gramian(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        assert np.allclose(v.gramian(ind), v.dot(v, ind=ind, o_ind=ind))


def test_add(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if len(v2) < len(v1):
        v2.append(v2, o_ind=np.zeros(len(v1) - len(v2), dtype=np.int))
    elif len(v2) > len(v1):
        v2.remove(range(len(v2)-len(v1)))
    c1 = v1.copy()
    cc1 = v1.copy()
    c1.axpy(1, v2)
    assert np.all(almost_equal(v1 + v2, c1))
    assert np.all(almost_equal(v1, cc1))


def test_iadd(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if len(v2) < len(v1):
        v2.append(v2, o_ind=np.zeros(len(v1) - len(v2), dtype=np.int))
    elif len(v2) > len(v1):
        v2.remove(range(len(v2)-len(v1)))
    c1 = v1.copy()
    c1.axpy(1, v2)
    v1 += v2
    assert np.all(almost_equal(v1, c1))


def test_sub(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if len(v2) < len(v1):
        v2.append(v2, o_ind=np.zeros(len(v1) - len(v2), dtype=np.int))
    elif len(v2) > len(v1):
        v2.remove(range(len(v2)-len(v1)))
    c1 = v1.copy()
    cc1 = v1.copy()
    c1.axpy(-1, v2)
    assert np.all(almost_equal((v1 - v2), c1))
    assert np.all(almost_equal(v1, cc1))


def test_isub(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    if len(v2) < len(v1):
        v2.append(v2, o_ind=np.zeros(len(v1) - len(v2), dtype=np.int))
    elif len(v2) > len(v1):
        v2.remove(range(len(v2)-len(v1)))
    c1 = v1.copy()
    c1.axpy(-1, v2)
    v1 -= v2
    assert np.all(almost_equal(v1, c1))


def test_neg(vector_array):
    v = vector_array
    c = v.copy()
    cc = v.copy()
    c.scal(-1)
    assert np.all(almost_equal(c, -v))
    assert np.all(almost_equal(v, cc))


def test_mul(vector_array):
    v = vector_array
    c = v.copy()
    for a in (-1, -3, 0, 1, 23):
        cc = v.copy()
        cc.scal(a)
        assert np.all(almost_equal((v * a), cc))
        assert np.all(almost_equal(v, c))


def test_mul_wrong_factor(vector_array):
    v = vector_array
    with pytest.raises(Exception):
        _ = v * v


def test_imul(vector_array):
    v = vector_array
    for a in (-1, -3, 0, 1, 23):
        c = v.copy()
        cc = v.copy()
        c.scal(a)
        cc *= a
        assert np.all(almost_equal(c, cc))


def test_imul_wrong_factor(vector_array):
    v = vector_array
    with pytest.raises(Exception):
        v *= v


########################################################################################################################


def test_append_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, remove_from_other=False)
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, remove_from_other=True)
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, ind=0)


def test_replace_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=False)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=True)


def test_axpy_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.axpy(0., c2, ind=ind1, x_ind=ind2)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.axpy(1., c2, ind=ind1, x_ind=ind2)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.axpy(-1., c2, ind=ind1, x_ind=ind2)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.axpy(1.42, c2, ind=ind1, x_ind=ind2)


def test_dot_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.dot(c2, ind=ind1, o_ind=ind2)


def test_pairwise_dot_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.pairwise_dot(c2, ind=ind1, o_ind=ind2)


def test_add_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    with pytest.raises(Exception):
        _ = v1 + v2


def test_iadd_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    with pytest.raises(Exception):
        v1 += v2


def test_sub_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    with pytest.raises(Exception):
        _ = v1 - v2


def test_isub_incompatible(incompatible_vector_array_pair):
    v1, v2 = incompatible_vector_array_pair
    with pytest.raises(Exception):
        v1 -= v2


########################################################################################################################


def test_copy_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.copy(ind)


def test_append_wrong_ind(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind in invalid_inds(v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            v1.append(v2, o_ind=ind)


def test_remove_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            c.remove(ind)


def test_replace_wrong_ind(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=False)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=True)


def test_scal_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            c.scal(0., ind=ind)
        c = v.copy()
        with pytest.raises(Exception):
            c.scal(1., ind=ind)
        c = v.copy()
        with pytest.raises(Exception):
            c.scal(-1., ind=ind)
        c = v.copy()
        with pytest.raises(Exception):
            c.scal(1.2, ind=ind)


def test_scal_wrong_coefficients(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        np.random.seed(len(v) + 99)
        for alpha in ([np.array([]), np.eye(v.len_ind(ind)), np.random.random(v.len_ind(ind) + 1)]
                      if v.len_ind(ind) > 0 else
                      [np.random.random(1)]):
            with pytest.raises(Exception):
                v.scal(alpha, ind=ind)


def test_axpy_wrong_ind(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        if v2.len_ind(ind2) == 1:
            continue
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.axpy(0., c2, ind=ind1, x_ind=ind2)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.axpy(1., c2, ind=ind1, x_ind=ind2)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.axpy(-1., c2, ind=ind1, x_ind=ind2)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.axpy(1.456, c2, ind=ind1, x_ind=ind2)


def test_axpy_wrong_coefficients(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        np.random.seed(len(v1) + 99)
        for alpha in ([np.array([]), np.eye(v1.len_ind(ind1)), np.random.random(v1.len_ind(ind1) + 1)]
                      if v1.len_ind(ind1) > 0 else
                      [np.random.random(1)]):
            with pytest.raises(Exception):
                v1.axpy(alpha, v2, ind=ind1, x_ind=ind2)


def test_dot_wrong_ind(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in chain(izip(valid_inds(v1), invalid_inds(v2)),
                            izip(invalid_inds(v1), valid_inds(v2))):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.dot(c2, ind=ind1, x_ind=ind2)


def test_pairwise_dot_wrong_ind(compatible_vector_array_pair):
    v1, v2 = compatible_vector_array_pair
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.pairwise_dot(c2, ind=ind1, x_ind=ind2)


def test_lincomb_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.lincomb(np.array([]), ind=ind)
        with pytest.raises(Exception):
            v.lincomb(np.array([1., 2.]), ind=ind)
        with pytest.raises(Exception):
            v.lincomb(np.ones(len(v)), ind=ind)
        if isinstance(ind, Number):
            with pytest.raises(Exception):
                v.lincomb(np.ones(1), ind=ind)
        if hasattr(ind, '__len__'):
            with pytest.raises(Exception):
                v.lincomb(np.zeros(len(ind)), ind=ind)


def test_l1_norm_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.l1_norm(ind)


def test_l2_norm_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.l2_norm(ind)


def test_sup_norm_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.sup_norm(ind)


def test_components_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.components(np.array([1]), ind=ind)
        with pytest.raises(Exception):
            v.components(np.array([]), ind=ind)
        with pytest.raises(Exception):
            v.components(np.arange(len(v)), ind=ind)


def test_amax_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.amax(ind)


def test_gramian_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.gramian(ind)


def test_pickle(picklable_vector_array):
    assert_picklable_without_dumps_function(picklable_vector_array)
