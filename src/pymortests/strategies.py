# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import functools

from hypothesis import strategies as hyst
from hypothesis import assume, given
from hypothesis.extra import numpy as hynp
import numpy as np
from scipy.stats._multivariate import random_correlation_gen

from pymor.analyticalproblems.functions import Function, ExpressionFunction, ConstantFunction
from pymor.core.config import config
from pymor.parameters.base import Mu
from pymor.tools.deprecated import Deprecated
from pymor.vectorarrays.list import NumpyListVectorSpace
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

if config.HAVE_FENICS:
    import dolfin as df
    from pymor.bindings.fenics import FenicsVectorSpace

if config.HAVE_DEALII:
    from pymor_dealii.pymor.vectorarray import DealIIVectorSpace

if config.HAVE_DUNEGDT:
    from pymor.bindings.dunegdt import DuneXTVectorSpace

if config.HAVE_NGSOLVE:
    import ngsolve as ngs
    import netgen.meshing as ngmsh
    from pymor.bindings.ngsolve import NGSolveVectorSpace


# hypothesis will gladly fill all our RAM with vector arrays if it's not restricted.
MAX_VECTORARRAY_LENGTH = 102
hy_lengths = hyst.integers(min_value=0, max_value=MAX_VECTORARRAY_LENGTH)
# this is a legacy restriction, some tests will not work as expected when this is changed/unset
MAX_ARRAY_ELEMENT_ABSVALUE = 1
MIN_ARRAY_ELEMENT_ABSVALUE = 1e-34


hy_dtypes = hyst.sampled_from([np.float64, np.complex128])
hy_float_array_elements = hyst.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False,
                                      min_value=-MAX_ARRAY_ELEMENT_ABSVALUE, max_value=MAX_ARRAY_ELEMENT_ABSVALUE)


@Deprecated("hypothesis.strategies.complex_numbers(allow_subnormal=False)")
@hyst.composite
def hy_complex_array_elements(draw):
    # This is a crutch in place for https://github.com/HypothesisWorks/hypothesis/issues/3390
    parts = hyst.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False,
                        min_value=-MAX_ARRAY_ELEMENT_ABSVALUE, max_value=MAX_ARRAY_ELEMENT_ABSVALUE)
    return complex(draw(parts), draw(parts))


@hyst.composite
def _hy_dims(draw, count, compatible):
    dims = hyst.integers(min_value=0, max_value=34)
    if compatible:
        return draw(equal_tuples(dims, count))
    dim_tuple = draw(hyst.tuples(*[dims for _ in range(count)]))
    for d in range(1, count):
        assume(dim_tuple[d] != dim_tuple[0])
    return dim_tuple


def nothing(*args, **kwargs):
    return hyst.nothing()


def _np_arrays(length, dim, dtype=None):
    if dtype is None:
        return hynp.arrays(dtype=np.float64, shape=(length, dim), elements=hy_float_array_elements) | \
            hynp.arrays(dtype=np.complex128, shape=(length, dim), elements=hy_complex_array_elements())
    if dtype is np.complex128:
        return hynp.arrays(dtype=dtype, shape=(length, dim), elements=hy_complex_array_elements())
    if dtype is np.float64:
        return hynp.arrays(dtype=dtype, shape=(length, dim), elements=hy_float_array_elements)
    raise RuntimeError(f'unsupported dtype={dtype}')


def _numpy_vector_spaces(draw, np_data_list, compatible, count, dims):
    return [(NumpyVectorSpace(d), ar) for d, ar in zip(dims, np_data_list)]


def _numpy_list_vector_spaces(draw, np_data_list, compatible, count, dims):
    return [(NumpyListVectorSpace(d), ar) for d, ar in zip(dims, np_data_list)]


def _block_vector_spaces(draw, np_data_list, compatible, count, dims):
    ret = []
    rr = draw(hyst.randoms())

    def _block_dims(d):
        bd = []
        while d > 1:
            block_size = rr.randint(1, d)
            bd.append(block_size)
            d -= block_size
        if d > 0:
            bd.append(d)
        return bd

    for c, (d, ar) in enumerate(zip(dims, np_data_list)):
        # only redraw after initial for (potentially) incompatible arrays
        if c == 0 or (not compatible and c > 0):
            block_dims = _block_dims(d)
        constituent_spaces = [NumpyVectorSpace(dim) for dim in block_dims]
        # TODO this needs to be relaxed again
        assume(len(constituent_spaces))
        ret.append((BlockVectorSpace(constituent_spaces), ar))
    return ret


_other_vector_space_types = []

if config.HAVE_FENICS:
    _FENICS_spaces = {}

    def _fenics_vector_spaces(draw, np_data_list, compatible, count, dims):
        ret = []
        for d, ar in zip(dims, np_data_list):
            assume(d > 1)
            if d not in _FENICS_spaces:
                _FENICS_spaces[d] = FenicsVectorSpace(df.FunctionSpace(df.UnitIntervalMesh(d - 1), 'Lagrange', 1))
            ret.append((_FENICS_spaces[d], ar))
        return ret
    _other_vector_space_types.append('fenics')

if config.HAVE_NGSOLVE:
    _NGSOLVE_spaces = {}

    def _create_ngsolve_space(dim):
        if dim not in _NGSOLVE_spaces:
            mesh = ngmsh.Mesh(dim=1)
            if dim > 0:
                pids = []
                for i in range(dim + 1):
                    pids.append(mesh.Add(ngmsh.MeshPoint(ngmsh.Pnt(i / dim, 0, 0))))
                for i in range(dim):
                    mesh.Add(ngmsh.Element1D([pids[i], pids[i + 1]], index=1))
            _NGSOLVE_spaces[dim] = NGSolveVectorSpace(ngs.L2(ngs.Mesh(mesh), order=0))
        return _NGSOLVE_spaces[dim]

    def _ngsolve_vector_spaces(draw, np_data_list, compatible, count, dims):
        return [(_create_ngsolve_space(d), ar) for d, ar in zip(dims, np_data_list)]
    _other_vector_space_types.append('ngsolve')

if config.HAVE_DEALII:
    def _dealii_vector_spaces(draw, np_data_list, compatible, count, dims):
        return [(DealIIVectorSpace(d), ar) for d, ar in zip(dims, np_data_list)]
    _other_vector_space_types.append('dealii')

if config.HAVE_DUNEGDT:
    def _dunegdt_vector_spaces(draw, np_data_list, compatible, count, dims):
        return [(DuneXTVectorSpace(d), ar) for d, ar in zip(dims, np_data_list)]
    _other_vector_space_types.append('dunegdt')


_picklable_vector_space_types = ['numpy', 'numpy_list', 'block']


@hyst.composite
def vector_arrays(draw, space_types, count=1, dtype=None, length=None, compatible=True):
    dims = draw(_hy_dims(count, compatible))
    dtype = dtype or draw(hy_dtypes)
    lngs = draw(length or hyst.tuples(*[hy_lengths for _ in range(count)]))
    np_data_list = [draw(_np_arrays(l, dim, dtype=dtype)) for l, dim in zip(lngs, dims)]
    space_type = draw(hyst.sampled_from(space_types))
    space_data = globals()[f'_{space_type}_vector_spaces'](draw, np_data_list, compatible, count, dims)
    ret = [sp.from_numpy(d) for sp, d in space_data]
    assume(len(ret))
    if len(ret) == 1:
        assert count == 1
        # in test funcs where we only need one array this saves a line to access the single list
        # element
        return ret[0]
    assert count > 1
    return ret


def given_vector_arrays(which='all', count=1, dtype=None, length=None, compatible=True, index_strategy=None, **kwargs):
    """This decorator hides the combination details of given

    the decorated function will be first wrapped in a |hypothesis.given| (with expanded `given_args`
    and then in |pytest.mark.parametrize| with selected implementation names. The decorated test
    function must still draw (which a vector_arrays or similar strategy) from the `data` argument in
    the default case.

    Parameters
    ----------
    which
        A list of implementation shortnames, or either of the special values "all" and "picklable".

    kwargs
        passed to `given` decorator as is, use for additional strategies

    count
        how many vector arrays to return (in a list), count=1 is special cased to just return the
        array
    dtype
        dtype of the foundational numpy data the vector array is constructed from
    length
        a hypothesis.strategy how many vectors to generate in each vector array
    compatible
        if count > 1, this switch toggles generation of vector_arrays with compatible `dim`,
        `length` and `dtype`
    """
    @functools.wraps(given)
    def inner_backend_decorator(func):
        try:
            use_imps = {'all': _picklable_vector_space_types  + _other_vector_space_types,
                        'picklable': _picklable_vector_space_types}[which]
        except KeyError:
            use_imps = which
        first_args = {}
        if index_strategy:
            arr_ind_strategy = index_strategy(vector_arrays(
                count=count, dtype=dtype, length=length, compatible=compatible, space_types=use_imps))
            first_args['vectors_and_indices'] = arr_ind_strategy
        else:
            arr_strategy = vector_arrays(count=count, dtype=dtype, length=length, compatible=compatible,
                                         space_types=use_imps)
            if count > 1:
                first_args['vector_arrays'] = arr_strategy
            else:
                first_args['vector_array'] = arr_strategy
        return given(**first_args, **kwargs)(func)

    return inner_backend_decorator


# TODO match st_valid_inds results to this
def valid_inds(v, length=None, random_module=None):
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
        # TODO what's with the magic number here?
        # Maybe related to this?
        # pymortests/vectorarray.py:910: VisibleDeprecationWarning:
        #   Creating an ndarray from nested sequences exceeding
        #   the maximum number of dimensions of 32 is deprecated.
        #   If you mean to do this, you must specify
        #  'dtype=object' when creating the ndarray.
        length = 32
    if len(v) > 0:
        for ind in [-len(v), 0, len(v) - 1]:
            yield ind
        if len(v) == length:
            yield slice(None)
        # this avoids managing random state "against" hypothesis when this function is used in a
        # strategy
        if random_module is None:
            np.random.seed(len(v) * length)
        yield list(np.random.randint(-len(v), len(v), size=length))
    else:
        if len(v) == 0:
            yield slice(0, 0)
        yield []


@hyst.composite
def valid_indices(draw, array_strategy, length=None):
    v = draw(array_strategy)
    ints = hyst.integers(min_value=-len(v), max_value=max(len(v)-1, 0))
    indices = hyst.nothing()
    if length is None:
        indices = indices | hyst.just([]) | hyst.lists(ints, max_size=2*len(v))
    else:
        indices = indices | hyst.slices(length)
    if len(v) > 0:
        inds = [-len(v), 0, len(v) - 1]
        if len(v) == length:
            inds.append(slice(None))
        indices = indices | hyst.lists(ints, max_size=length)
    else:
        inds = []
        if len(v) == 0:
            inds.append(slice(0, 0))
    indices = indices | hyst.sampled_from(inds)
    return v, draw(indices)


# TODO match st_valid_inds_of_same_length results to this
def valid_inds_of_same_length(v1, v2, random_module=None):
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
        # this avoids managing random state "against" hypothesis when this function is used in a
        # strategy
        if random_module is None:
            np.random.seed(len(v1) * len(v2))
        for count in np.linspace(0, min(len(v1), len(v2)), 3).astype(int):
            yield (list(np.random.randint(-len(v1), len(v1), size=count)),
                   list(np.random.randint(-len(v2), len(v2), size=count)))
        yield slice(None), np.random.randint(-len(v2), len(v2), size=len(v1))
        yield np.random.randint(-len(v1), len(v1), size=len(v2)), slice(None)


@hyst.composite
def st_valid_inds_of_same_length(draw, v1, v2):
    len1, len2 = len(v1), len(v2)
    ret = hyst.just(([], []))
    # TODO we should include integer arrays here by chaining
    # `| hynp.integer_array_indices(shape=(LEN_X,))`
    if len1 == len2:
        ints = hyst.integers(min_value=-len1, max_value=max(len1 - 1, 0))
        slicer = hyst.slices(len1) | hyst.lists(ints, max_size=len1)
        ret = ret | hyst.tuples(hyst.shared(slicer, key="st_valid_inds_of_same_length"),
                                hyst.shared(slicer, key="st_valid_inds_of_same_length"))
    if len1 > 0 and len2 > 0:
        mlen = min(len1, len2)
        ints = hyst.integers(min_value=-mlen, max_value=max(mlen - 1, 0))
        slicer = hyst.slices(mlen) | ints | hyst.lists(ints, max_size=mlen)
        ret = ret | hyst.tuples(hyst.shared(slicer, key="st_valid_inds_of_same_length_uneven"),
                                hyst.shared(slicer, key="st_valid_inds_of_same_length_uneven"))
    return draw(ret)


@hyst.composite
def st_scaling_value(draw, v1, v2=None):
    v1 = draw(v1)
    ints = hyst.integers(min_value=-1, max_value=23)
    r1 = draw(ints | hyst.just(np.arange(len(v1))))
    if v2:
        v2 = draw(v2)
        return v1, v2, r1, draw(ints | hyst.just(np.arange(len(v2))))
    return v1, r1


# TODO match st_valid_inds_of_different_length results to this
def valid_inds_of_different_length(v1, v2, random_module):
    # note this potentially yields no result at all for dual 0 length inputs
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
        # this avoids managing random state "against" hypothesis when this function is used in a
        # strategy
        if random_module is None:
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
def st_valid_inds_of_different_length(draw, v1, v2):
    def _filter(x):
        a, b = x
        a_type, b_type = type(a), type(b)
        if a_type != b_type:
            return True  # tuple + scalar index
        if a_type == tuple:
            return len(a) != len(b)
        return False  # both scalars => not of different length

    len1, len2 = len(v1), len(v2)
    # TODO we should include integer arrays here
    val1 = hyst.slices(len1)  # | hynp.integer_array_indices(shape=(len1,))
    val2 = hyst.slices(len2)  # | hynp.integer_array_indices(shape=(len1,))
    ret = hyst.tuples(val1, val2).filter(_filter)
    return draw(ret)


@hyst.composite
def st_valid_inds_of_same_or_different_length(draw, v1, v2):
    return draw(st_valid_inds_of_same_length(v1, v2) | st_valid_inds_of_different_length(v1, v2))


@hyst.composite
def same_and_different_length(draw, array_strategy):
    v = draw(array_strategy)
    if isinstance(v, list):
        return v, draw(st_valid_inds_of_same_or_different_length(*v))
    return v, draw(st_valid_inds_of_same_or_different_length(v, v))


@hyst.composite
def pairs_same_length(draw, array_strategy):
    v = draw(array_strategy)
    if isinstance(v, list):
        return v, draw(st_valid_inds_of_same_length(*v))
    return v, draw(st_valid_inds_of_same_length(v, v))


@hyst.composite
def pairs_diff_length(draw, array_strategy):
    v = draw(array_strategy)
    if isinstance(v, list):
        ind_list = draw(st_valid_inds_of_different_length(*v))
    else:
        ind_list = draw(st_valid_inds_of_different_length(v, v))
    # the consuming tests do not work for None as index
    assume(len(ind_list))
    return v, ind_list


@hyst.composite
def pairs_both_lengths(draw, array_strategy):
    return draw(hyst.one_of(pairs_same_length(array_strategy), pairs_diff_length(array_strategy)))


@hyst.composite
def invalid_indices(draw, array_strategy):
    length = 42
    v = draw(array_strategy)
    assert not isinstance(v, list)
    invalid_inds = (None, len(v), [len(v)], -len(v) - 1, [-len(v) - 1], [0, len(v)],
                    [-len(v) - 1] + [0, ] * (length - 1), list(range(length - 1)) + [len(v)])
    return v, draw(hyst.sampled_from(invalid_inds))


@hyst.composite
def base_vector_arrays(draw, count=1, dtype=None, max_dim=100):
    """Strategy to generate linear independent |VectorArray| inputs for test functions

    Parameters
    ----------
    draw hypothesis control function object
    count how many bases do you want
    dtype dtype for the generated bases, defaults to `np.float_`
    max_dim size limit for the generated

    Returns
    -------
    a list of |VectorArray| linear-independent objects of same dim and length
    """
    dtype = dtype or np.float_
    # simplest way currently of getting a |VectorSpace| to construct our new arrays from
    space_types = _picklable_vector_space_types  + _other_vector_space_types
    space = draw(vector_arrays(count=1, dtype=dtype, length=hyst.just((1,)), compatible=True, space_types=space_types)
                 .filter(lambda x: x[0].space.dim > 0 and x[0].space.dim < max_dim)).space
    length = space.dim

    # this lets hypothesis control np's random state too
    random = draw(hyst.random_module())
    # scipy performs this check although technically numpy accepts a different range
    assume(0 <= random.seed < 2**32 - 1)
    random_correlation = random_correlation_gen(random.seed)

    def _eigs():
        """Sum must equal to `length` for the scipy construct method"""
        min_eig, max_eig = 0.001, 1.
        eigs = np.asarray((max_eig-min_eig)*np.random.random(length-1) + min_eig, dtype=float)
        return np.append(eigs, [length - np.sum(eigs)])

    if length > 1:
        mat = [random_correlation.rvs(_eigs(), tol=1e-12) for _ in range(count)]
        return [space.from_numpy(m) for m in mat]
    else:
        scalar = 4*np.random.random((1, 1))+0.1
        return [space.from_numpy(scalar) for _ in range(count)]


@hyst.composite
def equal_tuples(draw, strategy, count):
    val = draw(strategy)
    return draw(hyst.tuples(*[hyst.just(val) for _ in range(count)]))


# stick to a few representative examples to avoid only seeing degenerate cases
# in the selected examples
mus = hyst.dictionaries(
    keys=hyst.sampled_from(['t', 'foo', 'bar']),
    values=hyst.sampled_from([
        np.array([1.]),
        np.array([1., 32., 3]),
        ExpressionFunction('x+1', 1),
        ExpressionFunction('[1., 0] * x + [0, 1.] * x**2', 1),
        ConstantFunction(np.array([1., 2, 3]))
    ])
).filter(lambda mu: 't' not in mu or (not isinstance(mu['t'], Function) and len(mu['t']) == 1)).map(Mu)
