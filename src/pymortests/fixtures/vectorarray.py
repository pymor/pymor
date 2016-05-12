# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import product
import numpy as np
import pytest

from pymor.playground.vectorarrays.disk import DiskVectorArray
from pymor.vectorarrays.block import BlockVectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.vectorarrays.list import NumpyVector, ListVectorArray
from pymor.vectorarrays.fenics import HAVE_FENICS
try:
    from pydealii.pymor.vectorarray import HAVE_DEALII
except ImportError as _:
    HAVE_DEALII = False

import os; TRAVIS = os.getenv('TRAVIS') == 'true'

def random_integers(count, seed):
    np.random.seed(seed)
    return list(np.random.randint(0, 3200, count))


def numpy_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return NumpyVectorArray(np.random.random((length, dim)), copy=False)


def numpy_list_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return ListVectorArray([NumpyVector(v, copy=False) for v in np.random.random((length, dim))],
                           subtype=(NumpyVector, dim), copy=False)


def numpy_disk_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return DiskVectorArray([NumpyVector(v, copy=False) for v in np.random.random((length, dim))],
                           subtype=(NumpyVector, dim))


def block_vector_array_factory(length, dims, seed):
    return BlockVectorArray([numpy_vector_array_factory(length, dim, seed + i) for i, dim in enumerate(dims)],
                            copy=False)

if HAVE_FENICS:
    import dolfin as df
    from pymor.vectorarrays.fenics import FenicsVectorSpace

    fenics_spaces = [df.FunctionSpace(df.UnitSquareMesh(ni, ni), 'Lagrange', 1)
                     for ni in [1, 10, 32, 100]]

    def fenics_vector_array_factory(length, space, seed):
        V = fenics_spaces[space]
        U = FenicsVectorSpace(V).zeros(length)
        dim = U.dim
        np.random.seed(seed)
        for v, a in zip(U._list, np.random.random((length, dim))):
            v.impl[:] = a
        return U

    fenics_vector_array_factory_arguments = \
        list(zip([0,  0,  1, 43, 102],      # len
            [0,  1,  3,  2,  2],      # ni
            random_integers(5, 123)))   # seed

    fenics_vector_array_factory_arguments_pairs_with_same_dim = \
        list(zip([0,  0,   1, 43, 102,  2],         # len1
            [0,  1,  37,  9, 104,  2],         # len2
            [0,  1,   3,  2,   2,  2],         # dim
            random_integers(5, 1234) + [42],  # seed1
            random_integers(5, 1235) + [42]))  # seed2

    fenics_vector_array_factory_arguments_pairs_with_different_dim = \
        list(zip([0,  0,  1, 43, 102],      # len1
            [0,  1,  1,  9,  10],      # len2
            [0,  1,  2,  3,   1],      # dim1
            [1,  2,  1,  2,   3],      # dim2
            random_integers(5, 1234),  # seed1
            random_integers(5, 1235)))  # seed2


if HAVE_DEALII:
    from pydealii.pymor.vectorarray import DealIIVectorSpace

    def dealii_vector_array_factory(length, dim, seed):
        U = DealIIVectorSpace(dim).zeros(length)
        np.random.seed(seed)
        for v, a in zip(U._list, np.random.random((length, dim))):
            v.impl[:] = a
        return U

def vector_array_from_empty_reserve(v, reserve):
    if reserve == 0:
        return v
    if reserve == 1:
        r = 0
    elif reserve == 2:
        r = len(v) + 10
    elif reserve == 3:
        r = int(len(v) / 2)
    c = v.empty(reserve=r)
    c.append(v)
    return c


numpy_vector_array_factory_arguments = \
    list(zip([0,  0,  1, 43, 102],      # len
        [0, 10, 34, 32,   0],      # dim
        random_integers(5, 123)))   # seed

numpy_vector_array_factory_arguments_pairs_with_same_dim = \
    list(zip([0,  0,  1, 43, 102,  2],         # len1
        [0,  1, 37,  9, 104,  2],         # len2
        [0, 10, 34, 32,   3, 13],         # dim
        random_integers(5, 1234) + [42],  # seed1
        random_integers(5, 1235) + [42]))  # seed2

numpy_vector_array_factory_arguments_pairs_with_different_dim = \
    list(zip([0,  0,  1, 43, 102],      # len1
        [0,  1,  1,  9,  10],      # len2
        [0, 10, 34, 32,   3],      # dim1
        [1, 11,  0, 33,   2],      # dim2
        random_integers(5, 1234),  # seed1
        random_integers(5, 1235)))  # seed2

block_vector_array_factory_arguments = \
    list(zip([0, 4, 3, 1, 3, 43, 102],      # len
        [(32, 1), (0, 3), (0, 0), (10,), (34, 1), (32, 3, 1), (1, 1, 1)],      # dim
        random_integers(7, 123)))   # seed

block_vector_array_factory_arguments_pairs_with_same_dim = \
    list(zip([0, 0,  3, 1, 43, 102],         # len1
        [0, 10, 2, 37,  9, 104],         # len2
        [(3, 2), (4, 0, 2), (4,), (34, 1, 1), (32, 3, 3),  (3, 3, 3)],  # dim
        random_integers(6, 1234),  # seed1
        random_integers(6, 1235)))  # seed2

block_vector_array_factory_arguments_pairs_with_different_dim = \
    list(zip([0, 0, 1, 43, 102],      # len1
        [0, 10, 1,  9,  10],      # len2
        [(3, 2), (9,), (34, 1, 1), (32, 3, 3),  (3, 3, 3)],      # dim1
        [(3, 1), (9, 3), (34, 2, 1), (32, 3), (4, 3, 3)],
        random_integers(5, 1234),  # seed1
        random_integers(5, 1235)))  # seed2

numpy_vector_array_generators = \
    [lambda args=args: numpy_vector_array_factory(*args) for args in numpy_vector_array_factory_arguments]

numpy_list_vector_array_generators = \
    [lambda args=args: numpy_list_vector_array_factory(*args) for args in numpy_vector_array_factory_arguments]

numpy_disk_vector_array_generators = [] if TRAVIS else \
    [lambda args=args: numpy_disk_vector_array_factory(*args) for args in numpy_vector_array_factory_arguments]

block_vector_array_generators = \
    [lambda args=args: block_vector_array_factory(*args) for args in block_vector_array_factory_arguments]

fenics_vector_array_generators = \
    [lambda args=args: fenics_vector_array_factory(*args) for args in fenics_vector_array_factory_arguments] \
    if HAVE_FENICS else []

dealii_vector_array_generators = \
    [lambda args=args: dealii_vector_array_factory(*args) for args in numpy_vector_array_factory_arguments] \
    if HAVE_DEALII else []

numpy_vector_array_pair_with_same_dim_generators = \
    [lambda l=l, l2=l2, d=d, s1=s1, s2=s2: (numpy_vector_array_factory(l, d, s1),
                                            numpy_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_same_dim]

numpy_list_vector_array_pair_with_same_dim_generators = \
    [lambda l=l, l2=l2, d=d, s1=s1, s2=s2: (numpy_list_vector_array_factory(l, d, s1),
                                            numpy_list_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_same_dim]

numpy_disk_vector_array_pair_with_same_dim_generators = [] if TRAVIS else \
    [lambda l=l, l2=l2, d=d, s1=s1, s2=s2: (numpy_disk_vector_array_factory(l, d, s1),
                                            numpy_disk_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_same_dim]

block_vector_array_pair_with_same_dim_generators = \
    [lambda l=l, l2=l2, d=d, s1=s1, s2=s2: (block_vector_array_factory(l, d, s1),
                                            block_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in block_vector_array_factory_arguments_pairs_with_same_dim]

fenics_vector_array_pair_with_same_dim_generators = \
    [lambda l=l, l2=l2, d=d, s1=s1, s2=s2: (fenics_vector_array_factory(l, d, s1),
                                            fenics_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in fenics_vector_array_factory_arguments_pairs_with_same_dim] \
    if HAVE_FENICS else []

dealii_vector_array_pair_with_same_dim_generators = \
    [lambda l=l, l2=l2, d=d, s1=s1, s2=s2: (dealii_vector_array_factory(l, d, s1),
                                            dealii_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_same_dim] \
    if HAVE_DEALII else []

numpy_vector_array_pair_with_different_dim_generators = \
    [lambda l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (numpy_vector_array_factory(l, d1, s1),
                                                     numpy_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_different_dim]

numpy_list_vector_array_pair_with_different_dim_generators = \
    [lambda l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (numpy_list_vector_array_factory(l, d1, s1),
                                                     numpy_list_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_different_dim]

numpy_disk_vector_array_pair_with_different_dim_generators = [] if TRAVIS else \
    [lambda l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (numpy_disk_vector_array_factory(l, d1, s1),
                                                     numpy_disk_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_different_dim]

block_vector_array_pair_with_different_dim_generators = \
    [lambda l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (block_vector_array_factory(l, d1, s1),
                                                     block_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in block_vector_array_factory_arguments_pairs_with_different_dim]

fenics_vector_array_pair_with_different_dim_generators = \
    [lambda l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (fenics_vector_array_factory(l, d1, s1),
                                                     fenics_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in fenics_vector_array_factory_arguments_pairs_with_different_dim] \
    if HAVE_FENICS else []

dealii_vector_array_pair_with_different_dim_generators = \
    [lambda l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (dealii_vector_array_factory(l, d1, s1),
                                                     dealii_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_different_dim] \
    if HAVE_DEALII else []



@pytest.fixture(params=numpy_vector_array_generators + numpy_list_vector_array_generators +
                       numpy_disk_vector_array_generators + block_vector_array_generators +
                       fenics_vector_array_generators)
def vector_array_without_reserve(request):
    return request.param()


@pytest.fixture(params=numpy_vector_array_generators + numpy_list_vector_array_generators +
                       numpy_disk_vector_array_generators + block_vector_array_generators)
def picklable_vector_array_without_reserve(request):
    return request.param()


@pytest.fixture(params=range(3))
def vector_array(vector_array_without_reserve, request):
    return vector_array_from_empty_reserve(vector_array_without_reserve, request.param)


@pytest.fixture(params=range(3))
def picklable_vector_array(picklable_vector_array_without_reserve, request):
    return vector_array_from_empty_reserve(picklable_vector_array_without_reserve, request.param)


@pytest.fixture(params=(numpy_vector_array_pair_with_same_dim_generators +
                        numpy_list_vector_array_pair_with_same_dim_generators +
                        numpy_disk_vector_array_pair_with_same_dim_generators +
                        block_vector_array_pair_with_same_dim_generators +
                        fenics_vector_array_pair_with_same_dim_generators))
def compatible_vector_array_pair_without_reserve(request):
    return request.param()


@pytest.fixture(params=product(range(3), range(3)))
def compatible_vector_array_pair(compatible_vector_array_pair_without_reserve, request):
    v1, v2 = compatible_vector_array_pair_without_reserve
    return vector_array_from_empty_reserve(v1, request.param[0]), vector_array_from_empty_reserve(v2, request.param[1])


@pytest.fixture(params=(numpy_vector_array_pair_with_different_dim_generators +
                        numpy_list_vector_array_pair_with_different_dim_generators +
                        numpy_disk_vector_array_pair_with_different_dim_generators +
                        block_vector_array_pair_with_different_dim_generators +
                        fenics_vector_array_pair_with_different_dim_generators +
                        dealii_vector_array_pair_with_different_dim_generators))
def incompatible_vector_array_pair(request):
    return request.param()
