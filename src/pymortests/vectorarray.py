# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import pytest
import numpy as np
from hypothesis import assume, settings, example
from hypothesis import strategies as hyst

from pymor.algorithms.basic import almost_equal
from pymor.core.config import config
from pymor.vectorarrays.interface import VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.tools.floatcmp import float_cmp, bounded
from pymortests.pickling import assert_picklable_without_dumps_function
import pymortests.strategies as pyst

MAX_RNG_REALIZATIONS = 30


def ind_complement(v, ind):
    if isinstance(ind, Number):
        ind = [ind]
    elif type(ind) is slice:
        ind = range(*ind.indices(len(v)))
    l = len(v)
    return sorted(set(range(l)) - {i if i >= 0 else l+i for i in ind})


def indexed(v, ind):
    if ind is None:
        return v
    elif type(ind) is slice:
        return v[ind]
    elif isinstance(ind, Number):
        return v[[ind]]
    elif len(ind) == 0:
        return np.empty((0, v.shape[1]), dtype=v.dtype)
    else:
        return v[ind]


def ind_to_list(v, ind):
    if type(ind) is slice:
        return list(range(*ind.indices(len(v))))
    elif not hasattr(ind, '__len__'):
        return [ind]
    else:
        return ind


@pyst.given_vector_arrays()
def test_empty(vector_array):
    with pytest.raises(Exception):
        vector_array.empty(-1)
    for r in (0, 1, 100):
        v = vector_array.empty(reserve=r)
        assert v.space == vector_array.space
        assert len(v) == 0
        try:
            assert v.to_numpy().shape == (0, v.dim)
        except NotImplementedError:
            pass


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_print(vectors_and_indices):
    v, ind = vectors_and_indices
    assert len(str(v))
    assert len(repr(v))
    assert len(str(v[ind]))
    assert len(repr(v[ind]))


@pyst.given_vector_arrays()
def test_zeros(vector_array):
    with pytest.raises(Exception):
        vector_array.zeros(-1)
    for c in (0, 1, 2, 30):
        v = vector_array.zeros(count=c)
        assert v.space == vector_array.space
        assert len(v) == c
        if min(v.dim, c) > 0:
            assert max(v.sup_norm()) == 0
            assert max(v.norm()) == 0
        try:
            assert v.to_numpy().shape == (c, v.dim)
            assert np.allclose(v.to_numpy(), np.zeros((c, v.dim)))
        except NotImplementedError:
            pass


@pyst.given_vector_arrays()
def test_ones(vector_array):
    with pytest.raises(Exception):
        vector_array.ones(-1)
    for c in (0, 1, 2, 30):
        v = vector_array.ones(count=c)
        assert v.space == vector_array.space
        assert len(v) == c
        if min(v.dim, c) > 0:
            assert np.allclose(v.sup_norm(), np.ones(c))
            assert np.allclose(v.norm(), np.full(c, np.sqrt(v.dim)))
        try:
            assert v.to_numpy().shape == (c, v.dim)
            assert np.allclose(v.to_numpy(), np.ones((c, v.dim)))
        except NotImplementedError:
            pass


@pyst.given_vector_arrays()
def test_full(vector_array):
    with pytest.raises(Exception):
        vector_array.full(9, -1)
    for c in (0, 1, 2, 30):
        for val in (-1e-3, 0, 7):
            v = vector_array.full(val, count=c)
            assert v.space == vector_array.space
            assert len(v) == c
            if min(v.dim, c) > 0:
                assert np.allclose(v.sup_norm(), np.full(c, abs(val)))
                assert np.allclose(v.norm(), np.full(c, np.sqrt(val**2 * v.dim)))
            try:
                assert v.to_numpy().shape == (c, v.dim)
                assert np.allclose(v.to_numpy(), np.full((c, v.dim), val))
            except NotImplementedError:
                pass


@pyst.given_vector_arrays(realizations=hyst.integers(min_value=0, max_value=MAX_RNG_REALIZATIONS),
                          low=hyst.floats(allow_infinity=False, allow_nan=False),
                          high=hyst.floats(allow_infinity=False, allow_nan=False))
@example(vector_array=NumpyVectorSpace(1).empty(), realizations=2,
         low=-5e-324, high=0.0)
def test_random_uniform_all(vector_array, realizations, low, high):
    if config.HAVE_DUNEGDT:
        # atm needs special casing due to norm implemenation handling of large vector elements
        from pymor.bindings.dunegdt import DuneXTVectorSpace
        assume(not isinstance(vector_array.space, DuneXTVectorSpace))
    _test_random_uniform(vector_array, realizations, low, high)


if config.HAVE_DUNEGDT:
    @pyst.given_vector_arrays(realizations=hyst.integers(min_value=0, max_value=MAX_RNG_REALIZATIONS),
                              low=hyst.floats(allow_infinity=False, allow_nan=False,
                                              max_value=10e100, min_value=-10e100),
                              high=hyst.floats(allow_infinity=False, allow_nan=False,
                                               max_value=10e100, min_value=-10e100),
                              which=('dunegdt',))
    def test_random_uniform_dune(vector_array, realizations, low, high):
        _test_random_uniform(vector_array, realizations, low, high)


def _test_random_uniform(vector_array, realizations, low, high):
    # avoid Overflow in np.random.RandomState.uniform
    assume(np.isfinite(high-low))
    with pytest.raises(Exception):
        vector_array.random(-1)
    c = realizations
    if c > 0 and high <= low:
        with pytest.raises(ValueError):
            vector_array.random(c, low=low, high=high)
        return
    seed = 123
    try:
        v = vector_array.random(c, low=low, high=high, seed=seed)
    except ValueError as e:
        if high <= low:
            return
        raise e
    assert v.space == vector_array.space
    assert len(v) == c
    if min(v.dim, c) > 0:
        assert np.all(v.sup_norm() <= max(abs(low), abs(high)))
    try:
        x = v.to_numpy()
        assert x.shape == (c, v.dim)
        assert np.all(x <= high)
        assert np.all(x >= low)
    except NotImplementedError:
        pass
    vv = vector_array.random(c, distribution='uniform', low=low, high=high, seed=seed)
    assert np.allclose((v - vv).sup_norm(), 0.)


@pyst.given_vector_arrays(realizations=hyst.integers(min_value=0, max_value=30),
                          loc=hyst.floats(allow_infinity=False, allow_nan=False),
                          scale=hyst.floats(allow_infinity=False, allow_nan=False))
def test_random_normal(vector_array, realizations, loc, scale):
    with pytest.raises(Exception):
        vector_array.random(-1)
    c = realizations
    if c > 0 > scale:
        with pytest.raises(ValueError):
            vector_array.random(c, 'normal', loc=loc, scale=scale)
        return
    seed = 123
    try:
        v = vector_array.random(c, 'normal', loc=loc, scale=scale, seed=seed)
    except ValueError as e:
        if scale <= 0:
            return
        raise e
    assert v.space == vector_array.space
    assert len(v) == c
    try:
        x = v.to_numpy()
        assert x.shape == (c, v.dim)
        import scipy.stats
        n = x.size
        if n == 0:
            return
        # test for expected value
        norm = scipy.stats.norm()
        gamma = 1 - 1e-7
        alpha = 1 - gamma
        lower = np.sum(x)/n - norm.ppf(1 - alpha/2) * scale / np.sqrt(n)
        upper = np.sum(x)/n + norm.ppf(1 - alpha/2) * scale / np.sqrt(n)
        bounded(lower, upper, loc)
    except NotImplementedError:
        pass
    vv = vector_array.random(c, 'normal', loc=loc, scale=scale, seed=seed)
    data = vv.to_numpy()
    # due to scaling data might actually now include nan or inf
    assume(not np.isnan(data).any())
    assume(not np.isinf(data).any())
    assert np.allclose((v - vv).sup_norm(), 0.)


@pyst.given_vector_arrays()
def test_from_numpy(vector_array):
    try:
        d = vector_array.to_numpy()
    except NotImplementedError:
        return
    try:
        v = vector_array.space.from_numpy(d)
        assert np.allclose(d, v.to_numpy())
    except NotImplementedError:
        pass


@pyst.given_vector_arrays()
def test_shape(vector_array):
    assert len(vector_array) >= 0
    assert vector_array.dim >= 0
    try:
        assert vector_array.to_numpy().shape == (len(vector_array), vector_array.dim)
    except NotImplementedError:
        pass


@pyst.given_vector_arrays()
def test_space(vector_array):
    assert isinstance(vector_array.space, VectorSpace)
    assert vector_array in vector_array.space


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_getitem_repeated(vectors_and_indices):
    v, ind = vectors_and_indices
    v_ind = v[ind]
    v_ind_copy = v_ind.copy()
    assert not v_ind_copy.is_view
    for ind_ind in pyst.valid_inds(v_ind, random_module=False):
        v_ind_ind = v_ind[ind_ind]
        assert np.all(almost_equal(v_ind_ind, v_ind_copy[ind_ind]))


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_copy(vectors_and_indices):
    v, ind = vectors_and_indices
    for deep in (True, False):
        if ind is None:
            c = v.copy(deep)
            assert len(c) == len(v)
        else:
            c = v[ind].copy(deep)
            assert len(c) == v.len_ind(ind)
        assert c.space == v.space
        if ind is None:
            assert np.all(almost_equal(c, v))
        else:
            assert np.all(almost_equal(c, v[ind]))
        try:
            assert np.allclose(c.to_numpy(), indexed(v.to_numpy(), ind))
        except NotImplementedError:
            pass


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
@example(vectors_and_indices=(NumpyVectorSpace(1).full(2.22044605e-16, 1), [0]))
def test_COW(vectors_and_indices):
    v, ind = vectors_and_indices
    for deep in (True, False):
        if ind is None:
            c = v.copy(deep)
            assert len(c) == len(v)
        else:
            c = v[ind].copy(deep)
            assert len(c) == v.len_ind(ind)
        assert c.space == v.space
        if len(c) > 0 and not np.all(c.norm() == 0):
            c *= 2
            vi = v[ind] if ind else v
            assert not np.all(almost_equal(c, vi, atol=0, rtol=0))
            try:
                assert np.allclose(c.to_numpy(), 2*indexed(v.to_numpy(), ind))
            except NotImplementedError:
                pass


@pyst.given_vector_arrays()
def test_copy_repeated_index(vector_array):
    v = vector_array
    if len(v) == 0:
        return
    ind = [int(len(v) * 3 / 4)] * 2
    for deep in (True, False):
        c = v[ind].copy(deep)
        assert almost_equal(c[0], v[ind[0]])
        assert almost_equal(c[1], v[ind[0]])
        try:
            assert indexed(v.to_numpy(), ind).shape == c.to_numpy().shape
        except NotImplementedError:
            pass
        c[0].scal(2.)
        assume(c[0].norm() != np.inf)
        assert almost_equal(c[1], v[ind[0]])
        assert float_cmp(c[0].norm(), 2 * v[ind[0]].norm())
        try:
            assert indexed(v.to_numpy(), ind).shape == c.to_numpy().shape
        except NotImplementedError:
            pass


@pyst.given_vector_arrays(count=2, index_strategy=pyst.pairs_both_lengths)
def test_append(vectors_and_indices):
    (v1, v2), (_, ind) = vectors_and_indices
    len_v1 = len(v1)
    c1, c2 = v1.copy(), v2.copy()
    c1.append(c2[ind])
    len_ind = v2.len_ind(ind)
    ind_complement_ = ind_complement(v2, ind)
    assert len(c1) == len_v1 + len_ind
    assert np.all(almost_equal(c1[len_v1:len(c1)], c2[ind]))
    try:
        assert np.allclose(c1.to_numpy(), np.vstack((v1.to_numpy(), indexed(v2.to_numpy(), ind))))
    except NotImplementedError:
        pass
    c1.append(c2[ind], remove_from_other=True)
    assert len(c2) == len(ind_complement_)
    assert c2.space == c1.space
    assert len(c1) == len_v1 + 2 * len_ind
    assert np.all(almost_equal(c1[len_v1:len_v1 + len_ind], c1[len_v1 + len_ind:len(c1)]))
    assert np.all(almost_equal(c2, v2[ind_complement_]))
    try:
        assert np.allclose(c2.to_numpy(), indexed(v2.to_numpy(), ind_complement_))
    except NotImplementedError:
        pass


@pyst.given_vector_arrays()
def test_append_self(vector_array):
    c = vector_array.copy()
    len_v = len(vector_array)
    c.append(c)
    assert len(c) == 2 * len_v
    assert np.all(almost_equal(c[:len_v], c[len_v:len(c)]))
    try:
        assert np.allclose(c.to_numpy(), np.vstack((vector_array.to_numpy(), vector_array.to_numpy())))
    except NotImplementedError:
        pass
    c = vector_array.copy()
    with pytest.raises(Exception):
        vector_array.append(vector_array, remove_from_other=True)


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_del(vectors_and_indices):
    v, ind = vectors_and_indices
    ind_complement_ = ind_complement(v, ind)
    c = v.copy()
    del c[ind]
    assert c.space == v.space
    assert len(c) == len(ind_complement_)
    assert np.all(almost_equal(v[ind_complement_], c))
    try:
        assert np.allclose(c.to_numpy(), indexed(v.to_numpy(), ind_complement_))
    except NotImplementedError:
        pass
    del c[:]
    assert len(c) == 0


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_scal(vectors_and_indices):
    v, ind = vectors_and_indices
    if v.len_ind(ind) != v.len_ind_unique(ind):
        with pytest.raises(Exception):
            c = v.copy()
            c[ind].scal(1.)
        return
    ind_complement_ = ind_complement(v, ind)
    c = v.copy()
    c[ind].scal(1.)
    assert len(c) == len(v)
    assert np.all(almost_equal(c, v))

    c = v.copy()
    c[ind].scal(0.)
    assert np.all(almost_equal(c[ind], v.zeros(v.len_ind(ind))))
    assert np.all(almost_equal(c[ind_complement_], v[ind_complement_]))

    for x in (1., 1.4, np.random.random(v.len_ind(ind))):
        c = v.copy()
        c[ind].scal(x)
        assert np.all(almost_equal(c[ind_complement_], v[ind_complement_]))
        assert np.allclose(c[ind].sup_norm(), v[ind].sup_norm() * abs(x))
        assert np.allclose(c[ind].norm(), v[ind].norm() * abs(x))
        try:
            y = v.to_numpy(True)
            if isinstance(x, np.ndarray) and not isinstance(ind, Number):
                x = x[:, np.newaxis]
            y[ind] *= x
            assert np.allclose(c.to_numpy(), y)
        except NotImplementedError:
            pass


@pyst.given_vector_arrays()
def test_scal_imaginary(vector_array):
    v = vector_array
    w = v.copy()
    w.scal(1j)
    assert np.allclose(v.norm(), w.norm())


@pyst.given_vector_arrays(count=2, index_strategy=pyst.pairs_same_length,
                          scalar=hyst.floats(min_value=1, max_value=pyst.MAX_VECTORARRAY_LENGTH))
def test_axpy(vectors_and_indices, scalar):
    (v1, v2), (ind1, ind2) = vectors_and_indices
    if v1.len_ind(ind1) != v1.len_ind_unique(ind1):
        with pytest.raises(Exception):
            c1, c2 = v1.copy(), v2.copy()
            c1[ind1].axpy(0., c2[ind2])
        return
    # ind2 is used for axpy args
    len_ind2 = v2.len_ind(ind2)
    assume(len_ind2 == 1 or len_ind2 == v1.len_ind(ind1))
    ind1_complement = ind_complement(v1, ind1)
    c1, c2 = v1.copy(), v2.copy()
    c1[ind1].axpy(0., c2[ind2])
    assert len(c1) == len(v1)
    assert np.all(almost_equal(c1, v1))
    assert np.all(almost_equal(c2, v2))

    a = scalar
    c1, c2 = v1.copy(), v2.copy()
    c1[ind1].axpy(a, c2[ind2])
    assert len(c1) == len(v1)
    assert np.all(almost_equal(c1[ind1_complement], v1[ind1_complement]))
    assert np.all(almost_equal(c2, v2))
    assert np.all(c1[ind1].sup_norm() <= v1[ind1].sup_norm() + abs(a) * v2[ind2].sup_norm() * (1. + 1e-10))
    assert np.all(c1[ind1].norm() <= (v1[ind1].norm() + abs(a) * v2[ind2].norm()) * (1. + 1e-10))
    try:
        x = v1.to_numpy(True).astype(complex)  # ensure that inplace addition works
        if isinstance(ind1, Number):
            x[[ind1]] += indexed(v2.to_numpy(), ind2) * a
        else:
            if isinstance(a, np.ndarray):
                aa = a[:, np.newaxis]
            else:
                aa = a
            x[ind1] += indexed(v2.to_numpy(), ind2) * aa
        assert np.allclose(c1.to_numpy(), x)
    except NotImplementedError:
        pass
    c1[ind1].axpy(-a, c2[ind2])
    assert len(c1) == len(v1)
    assert np.all(almost_equal(c1, v1, atol=1e-13, rtol=1e-13))


@pyst.given_vector_arrays(count=2, index_strategy=pyst.pairs_same_length,
                          scalar=hyst.floats(min_value=1, max_value=pyst.MAX_VECTORARRAY_LENGTH))
def test_axpy_one_x(vectors_and_indices, scalar):
    (v1, v2), (ind1, _) = vectors_and_indices
    for ind2 in pyst.valid_inds(v2, 1, random_module=False):
        assert v1.check_ind(ind1)
        assert v2.check_ind(ind2)
        if v1.len_ind(ind1) != v1.len_ind_unique(ind1):
            with pytest.raises(Exception):
                c1, c2 = v1.copy(), v2.copy()
                c1[ind1].axpy(0., c2[ind2])
            continue

        ind1_complement = ind_complement(v1, ind1)
        c1, c2 = v1.copy(), v2.copy()

        gc = c1[ind1]
        gv = c2[ind2]
        gc.axpy(0., gv)
        assert len(c1) == len(v1)
        assert np.all(almost_equal(c1, v1))
        assert np.all(almost_equal(c2, v2))

        a = scalar
        c1, c2 = v1.copy(), v2.copy()
        c1[ind1].axpy(a, c2[ind2])
        assert len(c1) == len(v1)
        assert np.all(almost_equal(c1[ind1_complement], v1[ind1_complement]))
        assert np.all(almost_equal(c2, v2))
        # for the openstack CI machines this could be 1 + 1e-10
        rtol_factor = 1. + 147e-9
        assert np.all(c1[ind1].sup_norm() <= v1[ind1].sup_norm() + abs(a) * v2[ind2].sup_norm() * rtol_factor)
        assert np.all(c1[ind1].norm() <= (v1[ind1].norm() + abs(a) * v2[ind2].norm()) * (1. + 1e-10))
        try:
            x = v1.to_numpy(True).astype(complex)  # ensure that inplace addition works
            if isinstance(ind1, Number):
                x[[ind1]] += indexed(v2.to_numpy(), ind2) * a
            else:
                if isinstance(a, np.ndarray):
                    aa = a[:, np.newaxis]
                else:
                    aa = a
                x[ind1] += indexed(v2.to_numpy(), ind2) * aa
            assert np.allclose(c1.to_numpy(), x)
        except NotImplementedError:
            pass
        c1[ind1].axpy(-a, c2[ind2])
        assert len(c1) == len(v1)
        assert np.all(almost_equal(c1, v1, atol=1e-13, rtol=1e-13))


@pyst.given_vector_arrays(index_strategy=pyst.pairs_same_length,
                          scalar=hyst.floats(min_value=1, max_value=pyst.MAX_VECTORARRAY_LENGTH))
def test_axpy_self(vectors_and_indices, scalar):
    v, (ind1, ind2) = vectors_and_indices
    if v.len_ind(ind1) != v.len_ind_unique(ind1):
        with pytest.raises(Exception):
            c, = v.copy()
            c[ind1].axpy(0., c[ind2])
        return

    ind1_complement = ind_complement(v, ind1)
    c = v.copy()
    rr = c[ind2]
    lp = c[ind1]
    lp.axpy(0., rr)
    assert len(c) == len(v)
    assert np.all(almost_equal(c, v))
    a = scalar
    c = v.copy()
    c[ind1].axpy(a, c[ind2])
    assert len(c) == len(v)
    assert np.all(almost_equal(c[ind1_complement], v[ind1_complement]))
    assert np.all(c[ind1].sup_norm() <= v[ind1].sup_norm() + abs(a) * v[ind2].sup_norm() * (1. + 1e-10))
    try:
        x = v.to_numpy(True).astype(complex)  # ensure that inplace addition works
        if isinstance(ind1, Number):
            x[[ind1]] += indexed(v.to_numpy(), ind2) * a
        else:
            if isinstance(a, np.ndarray):
                aa = a[:, np.newaxis]
            else:
                aa = a
            x[ind1] += indexed(v.to_numpy(), ind2) * aa
        assert np.allclose(c.to_numpy(), x)
    except NotImplementedError:
        pass
    c[ind1].axpy(-a, v[ind2])
    assert len(c) == len(v)
    assert np.all(almost_equal(c, v))

    ind = ind1
    if v.len_ind(ind) != v.len_ind_unique(ind):
        return

    for x in (1., 23., -4):
        c = v.copy()
        cc = v.copy()
        c[ind].axpy(x, c[ind])
        cc[ind].scal(1 + x)
        assert np.all(almost_equal(c, cc))


@pyst.given_vector_arrays(count=2)
def test_pairwise_inner(vector_arrays):
    v1, v2 = vector_arrays
    for ind1, ind2 in pyst.valid_inds_of_same_length(v1, v2):
        r = v1[ind1].pairwise_inner(v2[ind2])
        assert isinstance(r, np.ndarray)
        assert r.shape == (v1.len_ind(ind1),)
        r2 = v2[ind2].pairwise_inner(v1[ind1])
        assert np.allclose, (r, r2)
        assert np.all(r <= (v1[ind1].norm() * v2[ind2].norm() * (1. + 1e-10) + 1e-15))
        try:
            assert np.allclose(r, np.sum(indexed(v1.to_numpy(), ind1).conj() * indexed(v2.to_numpy(), ind2), axis=1))
        except NotImplementedError:
            pass


@pyst.given_vector_arrays(index_strategy=pyst.pairs_same_length)
def test_pairwise_inner_self(vectors_and_indices):
    v, (ind1, ind2) = vectors_and_indices
    r = v[ind1].pairwise_inner(v[ind2])
    assert isinstance(r, np.ndarray)
    assert r.shape == (v.len_ind(ind1),)
    r2 = v[ind2].pairwise_inner(v[ind1])
    assert np.allclose(r, r2.T.conj())
    assert np.all(r <= (v[ind1].norm() * v[ind2].norm() * (1. + 1e-10) + 1e-15))
    try:
        assert np.allclose(r, np.sum(indexed(v.to_numpy(), ind1).conj() * indexed(v.to_numpy(), ind2), axis=1))
    except NotImplementedError:
        pass
    ind = ind1
    r = v[ind].pairwise_inner(v[ind])
    assert np.allclose(r, v[ind].norm() ** 2)


@settings(deadline=None, print_blob=True)
@pyst.given_vector_arrays(count=2, index_strategy=pyst.pairs_both_lengths)
def test_inner(vectors_and_indices):
    (v1, v2), (ind1, ind2) = vectors_and_indices
    r = v1[ind1].inner(v2[ind2])
    assert isinstance(r, np.ndarray)
    assert r.shape == (v1.len_ind(ind1), v2.len_ind(ind2))
    r2 = v2[ind2].inner(v1[ind1])
    assert np.allclose(r, r2.T.conj())
    assert np.all(r <= (v1[ind1].norm()[:, np.newaxis] * v2[ind2].norm()[np.newaxis, :] * (1. + 1e-10) + 1e-15))
    try:
        assert np.allclose(r, indexed(v1.to_numpy(), ind1).conj().dot(indexed(v2.to_numpy(), ind2).T))
    except NotImplementedError:
        pass


@settings(deadline=None)
@pyst.given_vector_arrays(index_strategy=pyst.pairs_both_lengths)
def test_inner_self(vectors_and_indices):
    v, (ind1, ind2) = vectors_and_indices
    r = v[ind1].inner(v[ind2])
    assert isinstance(r, np.ndarray)
    assert r.shape == (v.len_ind(ind1), v.len_ind(ind2))
    r2 = v[ind2].inner(v[ind1])
    assert np.allclose(r, r2.T.conj())
    assert np.all(r <= (v[ind1].norm()[:, np.newaxis] * v[ind2].norm()[np.newaxis, :] * (1. + 1e-10) + 1e-15))
    try:
        assert np.allclose(r, indexed(v.to_numpy(), ind1).conj().dot(indexed(v.to_numpy(), ind2).T))
    except NotImplementedError:
        pass
    r = v[ind1].inner(v[ind1])
    assert np.allclose(r, r.T.conj())


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices, random=hyst.random_module())
def test_lincomb_1d(vectors_and_indices, random):
    v, ind = vectors_and_indices
    coeffs = np.random.random(v.len_ind(ind))
    lc = v[ind].lincomb(coeffs)
    assert lc.space == v.space
    assert len(lc) == 1
    lc2 = v.zeros()
    for coeff, i in zip(coeffs, ind_to_list(v, ind)):
        lc2.axpy(coeff, v[i])
    assert np.all(almost_equal(lc, lc2))


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices, random=hyst.random_module())
def test_lincomb_2d(vectors_and_indices, random):
    v, ind = vectors_and_indices
    for count in (0, 1, 5):
        coeffs = np.random.random((count, v.len_ind(ind)))
        lc = v[ind].lincomb(coeffs)
        assert lc.space == v.space
        assert len(lc) == count
        lc2 = v.empty(reserve=count)
        for coeffs_1d in coeffs:
            lc2.append(v[ind].lincomb(coeffs_1d))
        assert np.all(almost_equal(lc, lc2))


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices, random=hyst.random_module())
def test_lincomb_wrong_coefficients(vectors_and_indices, random):
    v, ind = vectors_and_indices
    coeffs = np.random.random(v.len_ind(ind) + 1)
    with pytest.raises(Exception):
        v[ind].lincomb(coeffs)
    coeffs = np.random.random(v.len_ind(ind)).reshape((1, 1, -1))
    with pytest.raises(Exception):
        v[ind].lincomb(coeffs)
    if v.len_ind(ind) > 0:
        coeffs = np.random.random(v.len_ind(ind) - 1)
        with pytest.raises(Exception):
            v[ind].lincomb(coeffs)
        coeffs = np.array([])
        with pytest.raises(Exception):
            v[ind].lincomb(coeffs)


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_norm(vectors_and_indices):
    v, ind = vectors_and_indices
    c = v.copy()
    norm = c[ind].norm()
    assert isinstance(norm, np.ndarray)
    assert norm.shape == (v.len_ind(ind),)
    assert np.all(norm >= 0)
    if v.dim == 0:
        assert np.all(norm == 0)
    try:
        assert np.allclose(norm, np.linalg.norm(indexed(v.to_numpy(), ind), axis=1))
    except NotImplementedError:
        pass
    c.scal(4.)
    assert np.allclose(c[ind].norm(), norm * 4)
    c.scal(-4.)
    assert np.allclose(c[ind].norm(), norm * 16)
    c.scal(0.)
    assert np.allclose(c[ind].norm(), 0)


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_norm2(vectors_and_indices):
    v, ind = vectors_and_indices
    c = v.copy()
    norm = c[ind].norm2()
    assert isinstance(norm, np.ndarray)
    assert norm.shape == (v.len_ind(ind),)
    assert np.all(norm >= 0)
    if v.dim == 0:
        assert np.all(norm == 0)
    try:
        assert np.allclose(norm, np.linalg.norm(indexed(v.to_numpy(), ind), axis=1)**2)
    except NotImplementedError:
        pass
    c.scal(4.)
    assert np.allclose(c[ind].norm2(), norm * 16)
    c.scal(-4.)
    assert np.allclose(c[ind].norm2(), norm * 256)
    c.scal(0.)
    assert np.allclose(c[ind].norm2(), 0)


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_sup_norm(vectors_and_indices):
    v, ind = vectors_and_indices
    c = v.copy()
    norm = c[ind].sup_norm()
    assert isinstance(norm, np.ndarray)
    assert norm.shape == (v.len_ind(ind),)
    assert np.all(norm >= 0)
    if v.dim == 0:
        assert np.all(norm == 0)
    if v.dim > 0:
        try:
            assert np.allclose(norm, np.max(np.abs(indexed(v.to_numpy(), ind)), axis=1))
        except NotImplementedError:
            pass
    c.scal(4.)
    assert np.allclose(c[ind].sup_norm(), norm * 4)
    c.scal(-4.)
    assert np.allclose(c[ind].sup_norm(), norm * 16)
    c.scal(0.)
    assert np.allclose(c[ind].sup_norm(), 0)


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices, random_count=hyst.integers(min_value=1, max_value=10))
def test_dofs(vectors_and_indices, random_count):
    v, ind = vectors_and_indices
    c = v.copy()
    dofs = c[ind].dofs(np.array([], dtype=int))
    assert isinstance(dofs, np.ndarray)
    assert dofs.shape == (v.len_ind(ind), 0)

    c = v.copy()
    dofs = c[ind].dofs([])
    assert isinstance(dofs, np.ndarray)
    assert dofs.shape == (v.len_ind(ind), 0)

    assume(v.dim > 0)

    c_ind = np.random.randint(0, v.dim, random_count)
    c = v.copy()
    dofs = c[ind].dofs(c_ind)
    assert dofs.shape == (v.len_ind(ind), random_count)
    c = v.copy()
    dofs2 = c[ind].dofs(list(c_ind))
    assert np.all(dofs == dofs2)
    c = v.copy()
    c.scal(3.)
    dofs2 = c[ind].dofs(c_ind)
    assert np.allclose(dofs * 3, dofs2)
    c = v.copy()
    dofs2 = c[ind].dofs(np.hstack((c_ind, c_ind)))
    assert np.all(dofs2 == np.hstack((dofs, dofs)))
    try:
        assert np.all(dofs == indexed(v.to_numpy(), ind)[:, c_ind])
    except NotImplementedError:
        pass


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_components_wrong_dof_indices(vectors_and_indices):
    v, ind = vectors_and_indices
    with pytest.raises(Exception):
        v[ind].dofs(None)
    with pytest.raises(Exception):
        v[ind].dofs(1)
    with pytest.raises(Exception):
        v[ind].dofs(np.array([-1]))
    with pytest.raises(Exception):
        v[ind].dofs(np.array([v.dim]))


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_amax(vectors_and_indices):
    v, ind = vectors_and_indices
    assume(v.dim > 0)
    max_inds, max_vals = v[ind].amax()
    assert np.allclose(max_vals, v[ind].sup_norm())
    for i, max_ind, max_val in zip(ind_to_list(v, ind), max_inds, max_vals):
        assert np.allclose(max_val, np.abs(v[[i]].dofs([max_ind])))


# def test_amax_zero_dim(zero_dimensional_vector_space):
#     for count in (0, 10):
#         v = zero_dimensional_vector_space.zeros(count=count)
#         for ind in valid_inds(v):
#             with pytest.raises(Exception):
#                 v.amax(ind)


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_gramian(vectors_and_indices):
    v, ind = vectors_and_indices
    assert np.allclose(v[ind].gramian(), v[ind].inner(v[ind]))


@pyst.given_vector_arrays(count=2, length=pyst.equal_tuples(pyst.hy_lengths, count=2))
def test_add(vector_arrays):
    v1, v2 = vector_arrays
    c1 = v1.copy()
    cc1 = v1.copy()
    c1.axpy(1, v2)
    assert np.all(almost_equal(v1 + v2, c1))
    assert np.all(almost_equal(v1, cc1))


@pyst.given_vector_arrays(count=2, length=pyst.equal_tuples(pyst.hy_lengths, count=2))
def test_iadd(vector_arrays):
    v1, v2 = vector_arrays
    c1 = v1.copy()
    c1.axpy(1, v2)
    v1 += v2
    assert np.all(almost_equal(v1, c1))


@pyst.given_vector_arrays(count=2, length=pyst.equal_tuples(pyst.hy_lengths, count=2))
def test_sub(vector_arrays):
    v1, v2 = vector_arrays
    c1 = v1.copy()
    cc1 = v1.copy()
    c1.axpy(-1, v2)
    assert np.all(almost_equal((v1 - v2), c1))
    assert np.all(almost_equal(v1, cc1))


@pyst.given_vector_arrays(count=2, length=pyst.equal_tuples(pyst.hy_lengths, count=2))
def test_isub(vector_arrays):
    v1, v2 = vector_arrays
    c1 = v1.copy()
    c1.axpy(-1, v2)
    v1 -= v2
    assert np.all(almost_equal(v1, c1))


@pyst.given_vector_arrays()
def test_neg(vector_array):
    c = vector_array.copy()
    cc = vector_array.copy()
    c.scal(-1)
    assert np.all(almost_equal(c, -vector_array))
    assert np.all(almost_equal(vector_array, cc))


@pyst.given_vector_arrays(index_strategy=pyst.st_scaling_value)
def test_mul(vectors_and_indices):
    vector_array, a = vectors_and_indices
    c = vector_array.copy()
    cc = vector_array.copy()
    cc.scal(a)
    assert np.all(almost_equal((vector_array * a), cc))
    assert np.all(almost_equal(vector_array, c))


@pyst.given_vector_arrays()
def test_mul_wrong_factor(vector_array):
    with pytest.raises(Exception):
        _ = vector_array * vector_array


@pyst.given_vector_arrays(index_strategy=pyst.st_scaling_value)
def test_rmul(vectors_and_indices):
    vector_array, a = vectors_and_indices
    c = vector_array.copy()
    cc = vector_array.copy()
    cc.scal(a)
    alpha = a * vector_array
    # the scaling_value strategy also draws ndarrays, for which alpha here will be an ndarray,
    # which in turn will fail the axpy hidden in the almost_equal check
    assume(not isinstance(alpha, np.ndarray))
    assert np.all(almost_equal(alpha, cc))
    assert np.all(almost_equal(vector_array, c))


@pyst.given_vector_arrays(index_strategy=pyst.st_scaling_value)
def test_imul(vectors_and_indices):
    vector_array, a = vectors_and_indices
    c = vector_array.copy()
    cc = vector_array.copy()
    c.scal(a)
    cc *= a
    assert np.all(almost_equal(c, cc))


@pyst.given_vector_arrays()
def test_imul_wrong_factor(vector_array):
    with pytest.raises(Exception):
        vector_array *= vector_array


@pyst.given_vector_arrays()
def test_iter(vector_array):
    v = vector_array
    w = v.empty()
    for vv in v:
        w.append(vv)
    assert np.all(almost_equal(w, v))


####################################################################################################


@pyst.given_vector_arrays(count=2, compatible=False)
def test_append_incompatible(vector_arrays):
    v1, v2 = vector_arrays
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, remove_from_other=False)
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, remove_from_other=True)


@pyst.given_vector_arrays(count=2, compatible=False)
def test_axpy_incompatible(vector_arrays):
    v1, v2 = vector_arrays
    for ind1, ind2 in pyst.valid_inds_of_same_length(v1, v2, random_module=False):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(0., c2[ind2])
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(1., c2[ind2])
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(-1., c2[ind2])
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].axpy(1.42, c2[ind2])


@pyst.given_vector_arrays(count=2, compatible=False)
def test_inner_incompatible(vector_arrays):
    v1, v2 = vector_arrays
    for ind1, ind2 in pyst.valid_inds_of_same_length(v1, v2, random_module=False):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].inner(c2[ind2])


@pyst.given_vector_arrays(count=2, compatible=False)
def test_pairwise_inner_incompatible(vector_arrays):
    v1, v2 = vector_arrays
    for ind1, ind2 in pyst.valid_inds_of_same_length(v1, v2, random_module=False):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1[ind1].pairwise_inner(c2[ind2])


@pyst.given_vector_arrays(count=2, compatible=False)
def test_add_incompatible(vector_arrays):
    v1, v2 = vector_arrays
    with pytest.raises(Exception):
        _ = v1 + v2


@pyst.given_vector_arrays(count=2, compatible=False)
def test_iadd_incompatible(vector_arrays):
    v1, v2 = vector_arrays
    with pytest.raises(Exception):
        v1 += v2


@pyst.given_vector_arrays(count=2, compatible=False)
def test_sub_incompatible(vector_arrays):
    v1, v2 = vector_arrays
    with pytest.raises(Exception):
        _ = v1 - v2


@pyst.given_vector_arrays(count=2, compatible=False)
def test_isub_incompatible(vector_arrays):
    v1, v2 = vector_arrays
    with pytest.raises(Exception):
        v1 -= v2


####################################################################################################


@pyst.given_vector_arrays(index_strategy=pyst.invalid_indices)
def test_wrong_ind_raises_exception(vectors_and_indices):
    vector_array, ind = vectors_and_indices
    with pytest.raises(Exception):
        vector_array[ind]


@pyst.given_vector_arrays(index_strategy=pyst.valid_indices)
def test_scal_wrong_coefficients(vectors_and_indices):
    v, ind = vectors_and_indices
    for alpha in ([np.array([]), np.eye(v.len_ind(ind)), np.random.random(v.len_ind(ind) + 1)]
                  if v.len_ind(ind) > 0 else
                  [np.random.random(1)]):
        with pytest.raises(Exception):
            v[ind].scal(alpha)


@pyst.given_vector_arrays(count=2, index_strategy=pyst.pairs_same_length)
def test_axpy_wrong_coefficients(vectors_and_indices):
    (v1, v2), (ind1, ind2) = vectors_and_indices
    for alpha in ([np.array([]), np.eye(v1.len_ind(ind1)), np.random.random(v1.len_ind(ind1) + 1)]
                  if v1.len_ind(ind1) > 0 else
                  [np.random.random(1)]):
        with pytest.raises(Exception):
            v1[ind1].axpy(alpha, v2[ind2])


@pyst.given_vector_arrays(which='picklable')
def test_pickle(vector_array):
    assert_picklable_without_dumps_function(vector_array)
