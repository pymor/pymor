from functools import partial
from itertools import product
from numbers import Number

import pytest
import numpy as np

from pymor.la import NumpyVectorArray
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

numpy_vector_array_factory_arguments = \
    zip([0, 0, 1, 43, 1024],
        [0, 10, 34, 32, 0],
        random_integers(4, 123))

numpy_vector_array_factory_arguments_pairs_with_same_dim = \
    zip([0, 0, 1, 43, 1024],
        [0, 1, 37, 9, 104],
        [0, 10, 34, 32, 3],
        random_integers(4, 1234),
        random_integers(4, 1235))

numpy_vector_array_factory_arguments_pairs_with_different_dim = \
    zip([0, 0, 1, 43, 1024],
        [0, 1, 1, 9, 104],
        [0, 10, 34, 32, 3],
        [1, 11, 0, 33, 2],
        random_integers(4, 1234),
        random_integers(4, 1235))

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


def invalid_inds(v):
    yield len(v)
    yield [len(v)]
    yield -1
    yield [-1]
    yield [0, len(v)]


def valid_inds(v):
    for ind in [None, [], range(len(v)), range(int(len(v)/2)), range(len(v)) * 2]:
        yield ind
    if len(v) > 0:
        for ind in [0, len(v) - 1]:
            yield ind


@pytest.fixture(params = numpy_vector_array_generators + numpy_list_vector_array_generators)
def vector_array(request):
    return request.param()


@pytest.fixture(params = numpy_vector_array_pair_with_same_dim_generators +
                         numpy_list_vector_array_pair_with_same_dim_generators)
def vector_array_pair_with_same_dim(request):
    return request.param()


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
    for d, r in product((0, 1, 2, 10, 100), (0, 1, 100)):
        v = VectorArray.empty(d, reserve=r)
        assert v.dim == d
        assert len(v) == 0


def test_zeros(VectorArray):
    with pytest.raises(Exception):
        VectorArray.zeros(-1)
    with pytest.raises(Exception):
        VectorArray.zeros(1, count=-1)
    for d, c in product((0, 1, 10, 100), (0, 1, 2, 30)):
        v = VectorArray.zeros(d, count=c)
        assert v.dim == d
        assert len(v) == c
        if min(d, c) > 0:
            assert max(v.sup_norm()) == 0
            assert max(v.l2_norm()) == 0


def test_shape(vector_array):
    assert len(vector_array) >= 0
    assert vector_array.dim >= 0


def test_copy(vector_array):
    v = vector_array
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.copy(ind)
    for ind in valid_inds(v):
        c = v.copy(ind=ind)
        if ind is None:
            assert len(c) == len(v)
        elif isinstance(ind, Number):
            assert len(c) == 1
        else:
            assert len(c) == len(ind)
        assert c.dim == v.dim
        assert np.all(c.almost_equal(v, o_ind=ind))


def test_copy_repeated_index(vector_array):
    if len(vector_array) == 0:
        return
    ind = [int(len(vector_array) * 3 / 4)] * 2
    c = vector_array.copy(ind)
    assert c.almost_equal(vector_array, ind=0, o_ind=ind[0])
    assert c.almost_equal(vector_array, ind=1, o_ind=ind[0])
    c.scal(2., ind=0)
    assert c.almost_equal(vector_array, ind=1, o_ind=ind[0])
    assert c.l2_norm(ind=0) == 2 * vector_array.l2_norm(ind=ind[0])


def test_append(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
    for ind in invalid_inds(v2):
        with pytest.raises(Exception):
            v1.append(v2, o_ind=ind)
    len_v1 = len(v1)
    len_v2 = len(v2)
    for ind in valid_inds(v2):
        c1 = v1.copy()
        c1.append(v2, o_ind=ind)
        len_ind = len_v2 if ind is None else 1 if isinstance(ind, Number) else len(ind)
        len_ind_unique = len_v2 if ind is None else 1 if isinstance(ind, Number) \
                              else len(set(i % len_v2 for i in ind))
        assert len(c1) == len_v1 + len_ind
        assert np.all(c1.almost_equal(v2, ind=range(len_v1, len(c1)), o_ind=ind))
        c2 = v2.copy()
        c1.append(c2, o_ind=ind, remove_from_other=True)
        assert len(c2) == len_v2 - len_ind_unique
        assert c2.dim == c1.dim
        assert len(c1) == len_v1 + 2 * len_ind
        assert np.all(c1.almost_equal(c1, ind=range(len_v1, len_v1 + len_ind), o_ind=range(len_v1 + len_ind, len(c1))))


def test_append_self(vector_array):
    v = vector_array
    len_v = len(v)
    v.append(v)
    assert len(v) == 2 * len_v
    assert np.all(v.almost_equal(v, ind=range(len_v), o_ind=range(len_v, len(v))))
    c = v.copy()
    with pytest.raises(Exception):
        v.append(v, remove_from_other=True)


def test_append_wrong_dim(vector_array_pair_with_different_dim):
    v1, v2 = vector_array_pair_with_different_dim
    with pytest.raises(Exception):
        v1.append(v2, remove_from_other=False)
    with pytest.raises(Exception):
        v1.append(v2, remove_from_other=True)
    with pytest.raises(Exception):
        v1.append(v2, ind=0)


def test_remove(vector_array):
    v = vector_array
    c = v.copy()
    for ind in invalid_inds(v):
        with pytest.raises(Exception):
            v.remove(ind)
    ind = range(int(len(v) * 3 / 4)) * 2
    v.remove(ind)
    assert v.dim == c.dim
    assert len(v) == len(c) - len(ind) / 2
    assert np.all(c.almost_equal(v, ind=range(len(ind)/2, len(c))))
    v.remove()
    assert len(v) == 0
