from hypothesis import strategies as hyst
from hypothesis.extra import numpy as hynp
import numpy as np
import pytest

from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.fixtures.vectorarray import vector_array_from_empty_reserve

lengths = hyst.integers(min_value=0, max_value=102)
dims = hyst.integers(min_value=0, max_value=34)
seeders = hyst.random_module()
reserves = hyst.integers(min_value=0, max_value=3)


@hyst.composite
def numpy_vector_array(draw):
    length = draw(lengths)
    dim = draw(dims)
    seeder = draw(seeders)
    np.random.seed(seeder.seed)
    v = NumpyVectorSpace.from_numpy(np.random.random((length, dim)))
    v = vector_array_from_empty_reserve(v, draw(reserves))
    return v