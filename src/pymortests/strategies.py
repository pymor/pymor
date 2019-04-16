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

lengths = hyst.integers(min_value=0, max_value=102)
array_elements = hyst.floats(allow_nan=False, allow_infinity=False, min_value=-1, max_value=1)
dims = hyst.integers(min_value=0, max_value=34)
# TODO non-fixed sampling pool
block_space_dims = hyst.sampled_from([(32, 1), (0, 3), (0, 0), (10,), (34, 1), (32, 3, 1), (1, 1, 1)])
seeders = hyst.random_module()
reserves = hyst.integers(min_value=0, max_value=3)
dtypes = hyst.sampled_from([np.float, np.complex])


@hyst.composite
def length_dim_data(draw):
    length, dim = draw(lengths), draw(dims)
    data = draw(hynp.arrays(dtype=dtypes, shape=(length, dim), elements=array_elements))
    return length, dim, data


@hyst.composite
def numpy_vector_array(draw):
    length, dim, data = draw(length_dim_data())
    v = NumpyVectorSpace.from_numpy(data)
    return vector_array_from_empty_reserve(v, draw(reserves))


@hyst.composite
def numpy_list_vector_array(draw):
    length, dim, data = draw(length_dim_data())
    v = NumpyListVectorSpace.from_numpy(data)
    return vector_array_from_empty_reserve(v, draw(reserves))


@hyst.composite
def block_vector_array(draw):
    length, dims = draw(lengths), draw(block_space_dims)
    return BlockVectorSpace([NumpyVectorSpace(dim) for dim in dims]).from_numpy(
        NumpyVectorSpace.from_numpy(draw(hynp.arrays(dtype=dtypes, shape=(length, sum(dims)),
                                                     elements=array_elements))).to_numpy()
    )


if config.HAVE_FENICS:
    @hyst.composite
    def fenics_spaces(draw, element_count=hyst.integers(min_value=1, max_value=101)):
        ni = draw(element_count)
        return df.FunctionSpace(df.UnitSquareMesh(ni, ni), 'Lagrange', 1)

    @hyst.composite
    def fenics_vector_array(draw):
        length = draw(lengths)
        V = draw(fenics_spaces())
        U = FenicsVectorSpace(V).zeros(length)
        dim = U.dim
        # dtype is float here since the petsc vector is not setup for complex
        for v, a in zip(U._list, draw(hynp.arrays(dtype=np.float, shape=(length, dim), elements=array_elements))):
            v.impl[:] = a
        return U
else:
    fenics_vector_array = hyst.nothing

if config.HAVE_NGSOLVE:
    @hyst.composite
    def ngsolve_vector_array(draw):
        length, dim = draw(lengths), draw(dims)
        space = create_ngsolve_space(dim)
        U = space.zeros(length)
        for v, a in zip(U._list, draw(hynp.arrays(dtype=np.float, shape=(length, dim), elements=array_elements))):
            v.to_numpy()[:] = a
        return U
else:
    ngsolve_vector_array = hyst.nothing

if config.HAVE_DEALII:
    @hyst.composite
    def dealii_vector_array(draw):
        length, dim = draw(lengths), draw(dims)
        U = DealIIVectorSpace(dim).zeros(length)
        for v, a in zip(U._list, draw(hynp.arrays(dtype=np.float, shape=(length, dim), elements=array_elements))):
            v.impl[:] = a
        return U
else:
    dealii_vector_array = hyst.nothing

vector_array = fenics_vector_array() | numpy_vector_array() | numpy_list_vector_array() | \
    dealii_vector_array() | ngsolve_vector_array() | block_vector_array()


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
def vector_array_with_ind(draw, ind_length=None):
    v = draw(vector_array)
    ind = hyst.sampled_from(list(valid_inds(v, ind_length)))
    i = draw(ind)
    return v, i


@hyst.composite
def vector_arrays_with_ind_pairs_same_length(draw):
    v = draw(vector_array)
    ind = hyst.sampled_from(list(valid_inds_of_same_length(v,v)))
    return v, draw(ind)


@hyst.composite
def vector_arrays_with_ind_pairs_diff_length(draw):
    v = draw(vector_array)
    ind = hyst.sampled_from(list(valid_inds_of_same_length(v,v)))
    return v, draw(ind)


vector_arrays_with_ind_pairs_both_lengths = vector_arrays_with_ind_pairs_same_length() | \
                                           vector_arrays_with_ind_pairs_diff_length()


@hyst.composite
def compatible_vector_array_pair(draw):
    v1 = draw(vector_array)
    v2 = draw(vector_array)
    assume(v1.dim == v2.dim)
    assume(v1.space == v2.space)
    return v1, v2