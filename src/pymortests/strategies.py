from hypothesis import strategies as hyst
from hypothesis.extra import numpy as hynp
import numpy as np
import pytest

from pymor.core.config import config
from pymor.vectorarrays.list import NumpyListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.fixtures.vectorarray import vector_array_from_empty_reserve

lengths = hyst.integers(min_value=0, max_value=102)
dims = hyst.integers(min_value=0, max_value=34)
seeders = hyst.random_module()
reserves = hyst.integers(min_value=0, max_value=3)
dtypes = hyst.sampled_from([np.float, np.complex])


@hyst.composite
def length_dim_data(draw):
    length, dim = draw(lengths), draw(dims)
    data = draw(hynp.arrays(dtype=dtypes, shape=(length, dim)))
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


if config.HAVE_FENICS:
    import dolfin as df
    from pymor.bindings.fenics import FenicsVectorSpace

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
        for v, a in zip(U._list, draw(hynp.arrays(dtype=np.float, shape=(length, dim)))):
            v.impl[:] = a
        return U
else:
    fenics_vector_array = hyst.nothing

