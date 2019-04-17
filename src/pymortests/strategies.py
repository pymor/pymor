from hypothesis import strategies as hyst
from hypothesis import assume, settings, HealthCheck
from hypothesis.extra import numpy as hynp
import numpy as np
import pytest

from pymor.core.config import config
from pymor.vectorarrays.list import NumpyListVectorSpace
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.fixtures.vectorarray import vector_array_from_empty_reserve

if config.HAVE_FENICS:
    import dolfin as df
    from pymor.bindings.fenics import FenicsVectorSpace

if config.HAVE_DEALII:
    from pydealii.pymor.vectorarray import DealIIVectorSpace

if config.HAVE_NGSOLVE:
    import ngsolve as ngs
    import netgen.meshing as ngmsh
    from netgen.geom2d import unit_square
    from pymor.bindings.ngsolve import NGSolveVectorSpace

    NGSOLVE_spaces = {}

    def create_ngsolve_space(dim):
        if dim not in NGSOLVE_spaces:
            mesh = ngmsh.Mesh(dim=1)
            if dim > 0:
                pids = []
                for i in range(dim + 1):
                    pids.append(mesh.Add(ngmsh.MeshPoint(ngmsh.Pnt(i / dim, 0, 0))))
                for i in range(dim):
                    mesh.Add(ngmsh.Element1D([pids[i], pids[i + 1]], index=1))
            NGSOLVE_spaces[dim] = NGSolveVectorSpace(ngs.L2(ngs.Mesh(mesh), order=0))
        return NGSOLVE_spaces[dim]

hy_lengths = hyst.integers(min_value=0, max_value=102)
hy_float_array_elements = hyst.floats(allow_nan=False, allow_infinity=False, min_value=-1, max_value=1)
hy_complex_array_elements = hyst.complex_numbers(allow_nan=False, allow_infinity=False, max_magnitude=2)
hy_dims = hyst.integers(min_value=0, max_value=34)
# TODO non-fixed sampling pool
hy_block_space_dims = hyst.sampled_from([(32, 1), (0, 3), (0, 0), (10,), (34, 1), (32, 3, 1), (1, 1, 1)])
hy_reserves = hyst.integers(min_value=0, max_value=3)
hy_dtypes = hyst.sampled_from([np.float64, np.complex128])


def nothing(*args, **kwargs):
    return hyst.nothing()


def np_arrays(length, dim, dtype=None):
    if dtype is None:
        return hynp.arrays(dtype=np.float64, shape=(length, dim), elements=hy_float_array_elements) | \
               hynp.arrays(dtype=np.complex128, shape=(length, dim), elements=hy_complex_array_elements)
    if dtype is np.complex128:
        return hynp.arrays(dtype=dtype, shape=(length, dim), elements=hy_complex_array_elements)
    if dtype is np.float64:
        return hynp.arrays(dtype=dtype, shape=(length, dim), elements=hy_float_array_elements)
    raise RuntimeError(f'unsupported dtype={dtype}')


@hyst.composite
def numpy_vector_array(draw, count=1, dtype=None, length=None):
    dim = draw(hy_dims) # compatible? draw single : draw tuple
    dtype = dtype or draw(hy_dtypes)
    lngs = length or hyst.tuples(*[hy_lengths for _ in range(count)])
    data = hyst.tuples(*[np_arrays(l, dim, dtype=dtype) for l in draw(lngs)])
    vec = [NumpyVectorSpace.from_numpy(d) for d in draw(data)]
    return [vector_array_from_empty_reserve(v, draw(hy_reserves)) for v in vec]


@hyst.composite
def numpy_list_vector_array(draw, count=1, dtype=None, length=None):
    dim = draw(hy_dims)
    dtype = dtype or draw(hy_dtypes)
    lngs = length or hyst.tuples(*[hy_lengths for _ in range(count)])
    data = hyst.tuples(*[np_arrays(l, dim, dtype=dtype) for l in draw(lngs)])
    vec = [NumpyListVectorSpace.from_numpy(d) for d in draw(data)]
    return [vector_array_from_empty_reserve(v, draw(hy_reserves)) for v in vec]


@hyst.composite
def block_vector_array(draw, count=1, dtype=None, length=None):
    dims = draw(hy_block_space_dims)
    lngs = length or hyst.tuples(*[hy_lengths for _ in range(count)])
    ret = []
    dtype = dtype or draw(hy_dtypes)
    for l in draw(lngs):
        data = draw(np_arrays(l, sum(dims), dtype=dtype))
        V = BlockVectorSpace([NumpyVectorSpace(dim, dtype=data.dtype) for dim in dims]).from_numpy(
            NumpyVectorSpace.from_numpy(data).to_numpy()
        )
        ret.append(V)
    return ret


if config.HAVE_FENICS:
    @hyst.composite
    def fenics_spaces(draw, element_count=hyst.integers(min_value=1, max_value=101)):
        ni = draw(element_count)
        return df.FunctionSpace(df.UnitSquareMesh(ni, ni), 'Lagrange', 1)

    @hyst.composite
    def fenics_vector_array(draw, count=1, dtype=None, length=None):
        if dtype: # complex is not actually supported
            assert dtype == np.float64
        dtype = np.float64
        lngs = draw(length or hyst.tuples(*[hy_lengths for _ in range(count)]))
        V = draw(fenics_spaces())
        Us = [FenicsVectorSpace(V).zeros(l) for l in lngs]
        dims = [U.dim for U in Us]
        for i in range(count):
            # dtype is float here since the petsc vector is not setup for complex
            for v, a in zip(Us[i]._list, draw(np_arrays(lngs[i], dims[i], dtype=dtype))):
            v = v.space.vector_from_numpy(a)
        return Us
else:
    fenics_vector_array = nothing

if config.HAVE_NGSOLVE:
    @hyst.composite
    def ngsolve_vector_array(draw, count=1, dtype=None, length=None):
        if dtype: # complex is not actually supported
            assert dtype == np.float64
        dtype = np.float64
        dim = draw(hy_dims)
        lngs = draw(length or hyst.tuples(*[hy_lengths for _ in range(count)]))
        space = create_ngsolve_space(dim)
        Us = [space.zeros(l) for l in lngs]
        for i in range(count):
            for v, a in zip(Us[i]._list, draw(np_arrays(lngs[i], dim, dtype=dtype))):
                v.to_numpy()[:] = a
        return Us
else:
    ngsolve_vector_array = nothing

if config.HAVE_DEALII:
    @hyst.composite
    def dealii_vector_array(draw, count=1, dtype=None, length=None):
        dim = draw(hy_dims)
        dtype = dtype or draw(hy_dtypes)
        lngs = draw(length or hyst.tuples(*[hy_lengths for _ in range(count)]))
        Us = [DealIIVectorSpace(dim).zeros(l) for l in lngs]
        for i in range(count):
            for v, a in zip(Us[i]._list, draw(np_arrays(lngs[i], dim, dtype=dtype))):
                v.impl[:] = a
        return Us
else:
    dealii_vector_array = nothing


def vector_arrays(count=1, dtype=None, length=None):
    return fenics_vector_array(count, dtype, length) | numpy_vector_array(count, dtype, length) | \
           numpy_list_vector_array(count, dtype, length) | dealii_vector_array(count, dtype, length) | \
           ngsolve_vector_array(count, dtype, length) | block_vector_array(count, dtype, length)


# TODO this needs to be a strategy
def valid_inds(v, length=None):
    if length is None:
        yield []
        yield slice(None)
        yield slice(0, len(v))
        yield slice(0, 0)
        yield slice(-3)
        yield slice(0, len(v), 3)
        yield slice(0, len(v)//2, 2)
        yield list(range(-len(v), len(v)))
        yield list(range(int(len(v)/2)))
        yield list(range(len(v))) * 2
        length = 32
    if len(v) > 0:
        for ind in [-len(v), 0, len(v) - 1]:
            yield ind
        if len(v) == length:
            yield slice(None)
        np.random.seed(len(v) * length)
        yield list(np.random.randint(-len(v), len(v), size=length))
    else:
        if len(v) == 0:
            yield slice(0, 0)
        yield []


# TODO this needs to be a strategy
def valid_inds_of_same_length(v1, v2):
    if len(v1) == len(v2):
        yield slice(None), slice(None)
        yield list(range(len(v1))), list(range(len(v1)))
        yield (slice(0, len(v1)),) * 2
        yield (slice(0, 0),) * 2
        yield (slice(-3),) * 2
        yield (slice(0, len(v1), 3),) * 2
        yield (slice(0, len(v1)//2, 2),) * 2
    yield [], []
    if len(v1) > 0 and len(v2) > 0:
        yield 0, 0
        yield len(v1) - 1, len(v2) - 1
        yield -len(v1), -len(v2)
        yield [0], 0
        yield (list(range(min(len(v1), len(v2))//2)),) * 2
        np.random.seed(len(v1) * len(v2))
        for count in np.linspace(0, min(len(v1), len(v2)), 3).astype(int):
            yield (list(np.random.randint(-len(v1), len(v1), size=count)),
                   list(np.random.randint(-len(v2), len(v2), size=count)))
        yield slice(None), np.random.randint(-len(v2), len(v2), size=len(v1))
        yield np.random.randint(-len(v1), len(v1), size=len(v2)), slice(None)


# TODO this needs to be a strategy
def valid_inds_of_different_length(v1, v2):
    if len(v1) != len(v2):
        yield slice(None), slice(None)
        yield list(range(len(v1))), list(range(len(v2)))
    if len(v1) > 0 and len(v2) > 0:
        if len(v1) > 1:
            yield [0, 1], 0
            yield [0, 1], [0]
            yield [-1, 0, 1], [0]
            yield slice(0, -1), []
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
                yield (list(np.random.randint(-len(v1), len(v1), size=count1)),
                       list(np.random.randint(-len(v2), len(v2), size=count2)))


@hyst.composite
def vector_array_with_ind(draw, ind_length=None, count=1, dtype=None, length=None):
    assert count == 1
    v = draw(vector_arrays(dtype=dtype, length=length), count)
    ind = hyst.sampled_from(list(valid_inds(v[0], ind_length)))
    return (*v, draw(ind))


@hyst.composite
def vector_arrays_with_ind_pairs_same_length(draw, count=1, dtype=None, length=None):
    assert count == 1
    v = draw(vector_arrays(dtype=dtype, length=length), count)
    ind = hyst.sampled_from(list(valid_inds_of_same_length(v[0],v[0])))
    return (*v, draw(ind))


@hyst.composite
def vector_arrays_with_ind_pairs_diff_length(draw, count=1, dtype=None, length=None):
    assert count == 1
    v = draw(vector_arrays(dtype=dtype, length=length), count)
    ind = hyst.sampled_from(list(valid_inds_of_different_length(v[0],v[0])))
    return (*v, draw(ind))


def vector_arrays_with_ind_pairs_both_lengths(count=1, dtype=None, length=None):
    return vector_arrays_with_ind_pairs_same_length(count, dtype, length) | \
           vector_arrays_with_ind_pairs_diff_length(count, dtype, length)


@hyst.composite
def equal_length_tuples(draw, count):
    val = draw(hy_lengths)
    return draw(hyst.tuples(*[hyst.just(val) for _ in range(count)]))