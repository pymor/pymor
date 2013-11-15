from functools import partial
from itertools import product, chain, izip
from numbers import Number

import pytest
import numpy as np

from pymor.la import NumpyVectorArray, Communicable
from pymor.la.listvectorarray import NumpyListVectorArray, NumpyVector


def random_integers(count, seed):
    np.random.seed(seed)
    return list(np.random.randint(0, 3200, count))


def numpy_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return NumpyVectorArray(np.random.random((length, dim)), copy=False)


def numpy_list_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return NumpyListVectorArray([NumpyVector(v, copy=False) for v in np.random.random((length, dim))], copy=False)


def vector_array_from_empty_reserve(v, reserve):
    if reserve == 0:
        return v
    if reserve == 1:
        r = 0
    elif reserve == 2:
        r = len(v) + 10
    elif reserve == 3:
        r = int(len(v) / 2)
    c = type(v).empty(v.dim, reserve=r)
    c.append(v)
    return c


numpy_vector_array_factory_arguments = \
    zip([0,  0,  1, 43, 102],      # len
        [0, 10, 34, 32,   0],      # dim
        random_integers(5, 123))   # seed

numpy_vector_array_factory_arguments_pairs_with_same_dim = \
    [[2, 2, 13, 3, 3]] + \
    zip([0,  0,  1, 43, 102],      # len1
        [0,  1, 37,  9, 104],      # len2
        [0, 10, 34, 32,   3],      # dim
        random_integers(5, 1234),  # seed1
        random_integers(5, 1235))  # seed2

numpy_vector_array_factory_arguments_pairs_with_different_dim = \
    zip([0,  0,  1, 43, 102],      # len1
        [0,  1,  1,  9,  10],      # len2
        [0, 10, 34, 32,   3],      # dim1
        [1, 11,  0, 33,   2],      # dim2
        random_integers(5, 1234),  # seed1
        random_integers(5, 1235))  # seed2

numpy_vector_array_generators = \
    [lambda: numpy_vector_array_factory(*args) for args in numpy_vector_array_factory_arguments]

numpy_list_vector_array_generators = \
    [lambda: numpy_list_vector_array_factory(*args) for args in numpy_vector_array_factory_arguments]

numpy_vector_array_pair_with_same_dim_generators = \
    [lambda: (numpy_vector_array_factory(l, d, s1), numpy_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_same_dim]

numpy_list_vector_array_pair_with_same_dim_generators = \
    [lambda: (numpy_list_vector_array_factory(l, d, s1), numpy_list_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_same_dim]

numpy_vector_array_pair_with_different_dim_generators = \
    [lambda: (numpy_vector_array_factory(l, d1, s1), numpy_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_different_dim]

numpy_list_vector_array_pair_with_different_dim_generators = \
    [lambda: (numpy_list_vector_array_factory(l, d1, s1), numpy_list_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_different_dim]


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
        yield [-1] + [0,] * (length - 1)
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
        yield len(v1) - 1 , len(v2) - 1
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
        for count1 in np.linspace(0, len(v1), 3):
            count2 = np.random.randint(-count1, len(v2) - count1) + count1
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



@pytest.fixture(params = numpy_vector_array_generators + numpy_list_vector_array_generators)
def vector_array_without_reserve(request):
    return request.param()


@pytest.fixture(params = range(3))
def vector_array(vector_array_without_reserve, request):
    return vector_array_from_empty_reserve(vector_array_without_reserve, request.param)


@pytest.fixture(params = numpy_vector_array_pair_with_same_dim_generators +
                         numpy_list_vector_array_pair_with_same_dim_generators)
def vector_array_pair_with_same_dim_without_reserve(request):
    return request.param()


@pytest.fixture(params = list(product(range(3), range(3))))
def vector_array_pair_with_same_dim(vector_array_pair_with_same_dim_without_reserve, request):
    v1, v2 = vector_array_pair_with_same_dim_without_reserve
    return vector_array_from_empty_reserve(v1, request.param[0]), vector_array_from_empty_reserve(v2, request.param[1])


@pytest.fixture(params = numpy_vector_array_pair_with_different_dim_generators +
                         numpy_list_vector_array_pair_with_different_dim_generators)
def vector_array_pair_with_different_dim(request):
    return request.param()


@pytest.fixture(params = [NumpyVectorArray, NumpyListVectorArray])
def VectorArray(request):
    return request.param


def test_empty(VectorArray):
    with pytest.raises(Exception):
        VectorArray.empty(-1)
    with pytest.raises(Exception):
        VectorArray.empty(1, reserve=-1)
    for dim, r in product((0, 1, 2, 10, 100), (0, 1, 100)):
        v = VectorArray.empty(dim, reserve=r)
        assert v.dim == dim
        assert len(v) == 0
        if isinstance(v, Communicable):
            d = v.data
            assert d.shape == (0, dim)


def test_zeros(VectorArray):
    with pytest.raises(Exception):
        VectorArray.zeros(-1)
    with pytest.raises(Exception):
        VectorArray.zeros(1, count=-1)
    for dim, c in product((0, 1, 10, 100), (0, 1, 2, 30)):
        v = VectorArray.zeros(dim, count=c)
        assert v.dim == dim
        assert len(v) == c
        if min(dim, c) > 0:
            assert max(v.sup_norm()) == 0
            assert max(v.l2_norm()) == 0
        if isinstance(v, Communicable):
            d = v.data
            assert d.shape == (c, dim)
            assert np.allclose(d, np.zeros((c, dim)))


def test_shape(vector_array):
    v = vector_array
    assert len(vector_array) >= 0
    assert vector_array.dim >= 0
    if isinstance(v, Communicable):
        d = v.data
        assert d.shape == (len(v), v.dim)


def test_copy(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        c = v.copy(ind=ind)
        assert len(c) == v.len_ind(ind)
        assert c.dim == v.dim
        assert np.all(c.almost_equal(v, o_ind=ind))
        if isinstance(v, Communicable):
            dv = v.data
            dc = c.data
            assert np.allclose(dc, indexed(dv, ind))


def test_copy_repeated_index(vector_array):
    v = vector_array
    if len(v) == 0:
        return
    ind = [int(len(vector_array) * 3 / 4)] * 2
    c = v.copy(ind)
    assert c.almost_equal(v, ind=0, o_ind=ind[0])
    assert c.almost_equal(v, ind=1, o_ind=ind[0])
    if isinstance(v, Communicable):
        dv = indexed(v.data, ind)
        dc = c.data
        assert dv.shape == dc.shape
    c.scal(2., ind=0)
    assert c.almost_equal(v, ind=1, o_ind=ind[0])
    assert c.l2_norm(ind=0) == 2 * v.l2_norm(ind=ind[0])
    if isinstance(v, Communicable):
        dv = indexed(v.data, ind)
        dc = c.data
        assert dv.shape == dc.shape


def test_append(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
    if isinstance(v1, Communicable):
        dv1 = v1.data
        dv2 = v2.data
    len_v1, len_v2 = len(v1), len(v2)
    for ind in valid_inds(v2):
        c1, c2 = v1.copy(), v2.copy()
        c1.append(c2, o_ind=ind)
        len_ind = v2.len_ind(ind)
        ind_complement_ = ind_complement(v2, ind)
        assert len(c1) == len_v1 + len_ind
        assert np.all(c1.almost_equal(c2, ind=range(len_v1, len(c1)), o_ind=ind))
        if isinstance(v1, Communicable):
            assert np.allclose(c1.data, np.vstack((dv1, indexed(dv2, ind))))
        c1.append(c2, o_ind=ind, remove_from_other=True)
        assert len(c2) == len(ind_complement_)
        assert c2.dim == c1.dim
        assert len(c1) == len_v1 + 2 * len_ind
        assert np.all(c1.almost_equal(c1, ind=range(len_v1, len_v1 + len_ind), o_ind=range(len_v1 + len_ind, len(c1))))
        assert np.all(c2.almost_equal(v2, o_ind=ind_complement_))
        if isinstance(v1, Communicable):
            assert np.allclose(c2.data, indexed(dv2, ind_complement_))


def test_append_self(vector_array):
    v = vector_array
    c = v.copy()
    len_v = len(v)
    c.append(c)
    assert len(c) == 2 * len_v
    assert np.all(c.almost_equal(c, ind=range(len_v), o_ind=range(len_v, len(c))))
    if isinstance(v, Communicable):
        assert np.allclose(c.data, np.vstack((v.data, v.data)))
    c = v.copy()
    with pytest.raises(Exception):
        v.append(v, remove_from_other=True)


def test_remove(vector_array):
    v = vector_array
    if isinstance(v, Communicable):
        dv = v.data
    for ind in valid_inds(v):
        ind_complement_ = ind_complement(v, ind)
        c = v.copy()
        c.remove(ind)
        assert c.dim == v.dim
        assert len(c) == len(ind_complement_)
        assert np.all(v.almost_equal(c, ind=ind_complement_))
        if isinstance(v, Communicable):
            assert np.allclose(c.data, indexed(dv, ind_complement_))
        c.remove()
        assert len(c) == 0


def test_replace(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
    len_v1, len_v2 = len(v1), len(v2)
    if isinstance(v1, Communicable):
        dv1 = v1.data
        dv2 = v2.data
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=False)
        assert len(c1) == len(v1)
        assert c1.dim == v1.dim
        # if the same index is repeated in ind1, the corresponding vector
        # will be the last one assigned to it
        if hasattr(ind1, '__len__'):
            ind2 = ind2 if hasattr(ind2, '__len__') else [ind2] if isinstance(ind2, Number) else range(len(v2))
            last_inds = {}
            for i1, i2 in zip(ind1, ind2):
                last_inds[i1] = i2
            ind2 = [last_inds[i] for i in ind1]
        assert np.all(c1.almost_equal(v2, ind=ind1, o_ind=ind2))
        assert np.all(c2.almost_equal(v2))
        if isinstance(v1, Communicable):
            x = dv1.copy()
            x[ind1] = indexed(dv2, ind2)
            assert np.allclose(c1.data, x)

        c1, c2 = v1.copy(), v2.copy()
        c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=True)
        assert len(c1) == len(v1)
        assert c1.dim == v1.dim
        ind2_complement = ind_complement(v2, ind2)
        # if the same index is repeated in ind1, the corresponding vector
        # will be the last one assigned to it
        if hasattr(ind1, '__len__'):
            ind2 = ind2 if hasattr(ind2, '__len__') else [ind2] if isinstance(ind2, Number) else range(len(v2))
            last_inds = {}
            for i1, i2 in zip(ind1, ind2):
                last_inds[i1] = i2
            ind2 = [last_inds[i] for i in ind1]
        assert np.all(c1.almost_equal(v2, ind=ind1, o_ind=ind2))
        assert len(c2) == len(ind2_complement)
        assert np.all(c2.almost_equal(v2, o_ind=ind2_complement))
        if isinstance(v1, Communicable):
            x = dv1.copy()
            x[ind1] = indexed(dv2, ind2)
            assert np.allclose(c1.data, x)
            assert np.allclose(c2.data, indexed(dv2, ind2_complement))


def test_replace_self(vector_array):
    v = vector_array
    if isinstance(v, Communicable):
        dv = v.data
    for ind1, ind2 in valid_inds_of_same_length(v, v):
        c = v.copy()
        with pytest.raises(Exception):
            c.replace(c, ind=ind1, o_ind=ind2, remove_from_other=True)
    for ind1, ind2 in valid_inds_of_same_length(v, v):
        c = v.copy()
        c.replace(c, ind=ind1, o_ind=ind2, remove_from_other=False)
        assert len(c) == len(v)
        assert c.dim == v.dim
        # if the same index is repeated in ind1, the corresponding vector
        # will be the last one assigned to it
        if hasattr(ind1, '__len__'):
            last_ind2 = ind2 if hasattr(ind2, '__len__') else [ind2] if isinstance(ind2, Number) else range(len(v))
            last_inds = {}
            for i1, i2 in zip(ind1, last_ind2):
                last_inds[i1] = i2
            last_ind2 = [last_inds[i] for i in ind1]
        else:
            last_ind2 = ind2
        assert np.all(c.almost_equal(v, ind=ind1, o_ind=last_ind2))
        if isinstance(v, Communicable):
            x = dv.copy()
            x[ind1] = indexed(dv, ind2)
            assert np.allclose(c.data, x)


def test_almost_equal(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
    if isinstance(v1, Communicable):
        dv1 = v1.data
        dv2 = v2.data
    for ind1, ind2 in valid_inds_of_same_length(v1, v2):
        for rtol, atol in ((1e-5, 1e-8), (1e-10, 1e-12), (0., 1e-8), (1e-5, 1e-8)):
            r = v1.almost_equal(v2, ind=ind1, o_ind=ind2)
            assert isinstance(r, np.ndarray)
            assert r.shape == (v1.len_ind(ind1),)
            if isinstance(v1, Communicable):
                assert np.all(r == np.all(np.abs(indexed(dv1, ind1) - indexed(dv2, ind2))
                                          <= atol + rtol * np.abs(indexed(dv2, ind2)), axis=1))


def test_almost_equal2(vector_array):
    v = vector_array
    for ind in valid_inds(v):
        for rtol, atol in ((1e-5, 1e-8), (1e-10, 1e-12), (0., 1e-8), (1e-5, 1e-8)):
            r = v.almost_equal(v, ind=ind, o_ind=ind)
            assert isinstance(r, np.ndarray)
            assert r.shape == (v.len_ind(ind),)
            assert np.all(r)
            if v.dim == 0:
                continue

            c = v.copy()
            c.scal(1. / (np.max(v.sup_norm(ind)) * atol))
            assert np.all(c.almost_equal(c.zeros(v.dim, v.len_ind(ind)), ind=ind))

            c = v.copy()
            c.scal(2. / (np.max(v.sup_norm(ind)) * atol))
            assert not np.all(c.almost_equal(c.zeros(v.dim, v.len_ind(ind)), ind=ind))

            c = v.copy()
            c.scal(1. + atol / np.max(v.sup_norm(ind)))
            assert np.all(c.almost_equal(v, ind=ind, o_ind=ind))

            c = v.copy()
            c.scal(2. + atol / np.max(v.sup_norm(ind)))
            assert not np.all(c.almost_equal(v, ind=ind, o_ind=ind))

            c = v.copy()
            c.scal(1. + rtol)
            assert np.all(c.almost_equal(v, ind=ind, o_ind=ind))

            c = v.copy()
            c.scal(2. + rtol)
            assert not np.all(c.almost_equal(v, ind=ind, o_ind=ind))



########################################################################################################################


def test_append_wrong_dim(vector_array_pair_with_different_dim):
    v1, v2 = vector_array_pair_with_different_dim
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, remove_from_other=False)
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, remove_from_other=True)
    c1, c2 = v1.copy(), v2.copy()
    with pytest.raises(Exception):
        c1.append(c2, ind=0)

def test_replace_wrong_dim(vector_array_pair_with_different_dim):
    v1, v2 = vector_array_pair_with_different_dim
    for ind1, ind2 in chain(valid_inds_of_same_length(v1, v2), invalid_ind_pairs(v1, v2)):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=False)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=True)


def test_almost_equal_wrong_dim(vector_array_pair_with_different_dim):
    v1, v2 = vector_array_pair_with_different_dim
    for ind1, ind2 in chain(valid_inds_of_same_length(v1, v2), invalid_ind_pairs(v1, v2)):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.almost_equal(c2, ind=ind1, o_ind=ind2)


def test_axpy_wrong_dim(vector_array_pair_with_different_dim):
    v1, v2 = vector_array_pair_with_different_dim
    for ind1, ind2 in chain(valid_inds_of_same_length(v1, v2), invalid_ind_pairs(v1, v2)):
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


def test_dot_wrong_dim(vector_array_pair_with_different_dim):
    v1, v2 = vector_array_pair_with_different_dim
    for ind1, ind2 in chain(valid_inds_of_same_length(v1, v2), invalid_ind_pairs(v1, v2)):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.dot(c2, pairwise=True, ind=ind1, o_ind=ind2)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.dot(c2, pairwise=False, ind=ind1, o_ind=ind2)


########################################################################################################################


def test_copy_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.copy(ind)


def test_append_wrong_ind(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
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


def test_replace_wrong_ind(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=False)
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.replace(c2, ind=ind1, o_ind=ind2, remove_from_other=True)


def test_almost_equal_wrong_ind(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        if (ind1 is None and len(v1) == 1 or isinstance(ind1, Number) or hasattr(ind1, '__len__') and len(ind1) == 1 or
            ind2 is None and len(v2) == 1 or isinstance(ind2, Number) or hasattr(ind2, '__len__') and len(ind2) == 1) :
            continue
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.almost_equal(c2, ind=ind1, o_ind=ind2)


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


def test_axpy_wrong_ind(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
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


def test_dot_wrong_ind(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
    for ind1, ind2 in invalid_ind_pairs(v1, v2):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.dot(c2, pairwise=True, ind=ind1, x_ind=ind2)
    for ind1, ind2 in chain(izip(valid_inds(v1), invalid_inds(v2)),
                            izip(invalid_inds(v1), valid_inds(v2))):
        c1, c2 = v1.copy(), v2.copy()
        with pytest.raises(Exception):
            c1.dot(c2, pairwise=False, ind=ind1, x_ind=ind2)


def test_lincomb_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            c.lincomb(np.array([]), ind=ind)
        c = v.copy()
        with pytest.raises(Exception):
            c.lincomb(np.array([1., 2.]), ind=ind)
        c = v.copy()
        with pytest.raises(Exception):
            c.lincomb(np.ones(len(v)), ind=ind)
        if isinstance(ind, Number):
            c = v.copy()
            with pytest.raises(Exception):
                c.lincomb(np.ones(1), ind=ind)
        if hasattr(ind, '__len__'):
            c = v.copy()
            with pytest.raises(Exception):
                c.lincomb(np.zeros(len(ind)), ind=ind)


def test_l1_norm_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            c.l1_norm(ind)


def test_l2_norm_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            c.l2_norm(ind)


def test_sup_norm_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            c.sup_norm(ind)


def test_components_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            c.components(np.array([1]), ind=ind)
        c = v.copy()
        with pytest.raises(Exception):
            c.components(np.array([]), ind=ind)
        c = v.copy()
        with pytest.raises(Exception):
            c.components(np.arange(len(v)), ind=ind)


def test_amax_wrong_ind(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        c = v.copy()
        with pytest.raises(Exception):
            c.amax(ind)
