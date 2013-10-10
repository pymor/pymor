from functools import partial
from itertools import product

import pytest
import numpy as np

from pymor.la import NumpyVectorArray
from pymor.la.listvectorarray import NumpyListVectorArray


def random_integers(count, seed):
    np.random.seed(seed)
    return list(np.random.randint(0, 3200, count))


def numpy_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return NumpyVectorArray(np.random.random((length, dim)), copy=False)


def numpy_list_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return NumpyListVectorArray(list(np.random.random((length, dim))), copy=False)


numpy_vector_array_generators = \
    [lambda: numpy_vector_array_factory(l, d, s) for l, d, s in zip([0, 0, 1, 43, 1024],
                                                                    [0, 10, 34, 32, 0],
                                                                    random_integers(4, 123))]

numpy_list_vector_array_generators = \
    [lambda: numpy_vector_array_factory(l, d, s) for l, d, s in zip([0, 0, 1, 43, 1024],
                                                                    [0, 10, 34, 32, 0],
                                                                    random_integers(4, 123))]

numpy_vector_array_pair_with_same_dim_generators = \
    [lambda: (numpy_vector_array_factory(l, d, s1), numpy_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in zip([0, 0, 1, 43, 1024],
                                 [0, 1, 37, 9, 104],
                                 [0, 10, 34, 32, 3],
                                 random_integers(4, 1234),
                                 random_integers(4, 1235))]

numpy_list_vector_array_pair_with_same_dim_generators = \
    [lambda: (numpy_vector_array_factory(l, d, s1), numpy_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in zip([0, 0, 1, 43, 1024],
                                 [0, 1, 37, 9, 104],
                                 [0, 10, 34, 32, 3],
                                 random_integers(4, 1234),
                                 random_integers(4, 1235))]


@pytest.fixture(params = numpy_vector_array_generators + numpy_list_vector_array_generators)
def vector_array(request):
    return request.param()


@pytest.fixture(params = numpy_vector_array_pair_with_same_dim_generators +
                         numpy_list_vector_array_pair_with_same_dim_generators)
def vector_array_pair_with_same_dim(request):
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
    with pytest.raises(Exception):
        vector_array.copy(len(vector_array))
    with pytest.raises(Exception):
        vector_array.copy([len(vector_array)])
    for ind in (None, range(len(vector_array)), range(int(len(vector_array)/2)), range(len(vector_array)) * 2):
        c = vector_array.copy(ind=ind)
        assert len(c) == len(vector_array) if ind is None else len(c) == len(ind)
        assert c.dim == vector_array.dim
        assert np.all(c.almost_equal(vector_array, o_ind=ind))


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
    with pytest.raises(Exception):
        v1.append(v2, o_ind=len(v2))
    with pytest.raises(Exception):
        v1.append(v2, o_ind=[len(v2)])
    len_v1 = len(v1)
    len_v2 = len(v2)
    v1.append(v2)
    assert len(v1) == len_v1 + len_v2
    assert np.all(v1.almost_equal(v2, ind=range(len_v1, len(v1))))
    v1.append(v2, remove_from_other=True)
    assert len(v2) == 0
    assert v2.dim == v1.dim
    assert len(v1) == len_v1 + 2 * len_v2
    assert np.all(v1.almost_equal(v1, ind=range(len_v1, len_v1 + len_v2), o_ind=range(len_v1 + len_v2, len(v1))))


def test_append_2(vector_array_pair_with_same_dim):
    v1, v2 = vector_array_pair_with_same_dim
    ind = range(int(len(v2) * 3 / 4)) * 2
    len_v1 = len(v1)
    len_v2 = len(v2)
    v1.append(v2, o_ind=ind)
    assert len(v1) == len_v1 + len(ind)
    assert len(v2) == len_v2
    assert np.all(v1.almost_equal(v2, ind=range(len_v1, len(v1)), o_ind=ind))
    v1.append(v2, o_ind=ind, remove_from_other=True)
    assert len(v1) == len_v1 + 2*len(ind)
    assert len(v2) == len_v2 - len(ind) / 2
    assert np.all(v1.almost_equal(v1, ind=range(len_v1, len_v1 + len(ind)), o_ind=range(len_v1 + len(ind), len(v1))))


def test_append_self(vector_array):
    v = vector_array
    len_v = len(v)
    v.append(v)
    assert len(v) == 2 * len_v
    assert np.all(v.almost_equal(v, ind=range(len_v), o_ind=range(len_v, len(v))))
    c = v.copy()
    with pytest.raises(Exception):
        v.append(v, remove_from_other=True)


def test_remove(vector_array):
    v = vector_array
    c = v.copy()
    ind = range(int(len(v) * 3 / 4)) * 2
    v.remove(ind)
    assert v.dim == c.dim
    assert len(v) == len(c) - len(ind) / 2
    assert np.all(c.almost_equal(v, ind=range(len(ind)/2, len(c))))
    v.remove()
    assert len(v) == 0
