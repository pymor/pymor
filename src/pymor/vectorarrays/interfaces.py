# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
import sys

import numpy as np

from pymor.core.config import config
from pymor.core.interfaces import BasicInterface, ImmutableInterface, abstractmethod


def _numpy_version_older(version_tuple):
    np_tuple = tuple(int(p) for p in np.__version__.split('.')[:3])
    return np_tuple < version_tuple

_INDEXTYPES = (Number,) if not _numpy_version_older((1, 9)) else (Number, np.intp)


class VectorArrayInterface(BasicInterface):
    """Interface for vector arrays.

    A vector array should be thought of as a list of (possibly high-dimensional) vectors.
    While the vectors themselves will be inaccessible in general (e.g. because they are
    managed by an external PDE solver code), operations on the vectors like addition can
    be performed via this interface.

    It is assumed that the number of vectors is small enough such that scalar data
    associated to each vector can be handled on the Python side. As such, methods like
    :meth:`~VectorArrayInterface.l2_norm` or :meth:`~VectorArrayInterface.gramian` will
    always return |NumPy arrays|.

    An implementation of the `VectorArrayInterface` via |NumPy arrays| is given by
    |NumpyVectorArray|.  In general, it is the implementors decision how memory is
    allocated internally (e.g.  continuous block of memory vs. list of pointers to the
    individual vectors.) Thus, no general assumptions can be made on the costs of operations
    like appending to or removing vectors from the array. As a hint for 'continuous block
    of memory' implementations, :meth:`~VectorSpaceInterface.zeros` provides a `reserve`
    keyword argument which allows to specify to what size the array is assumed to grow.

    As with |Numpy array|, |VectorArrays| can be indexed with numbers, slices and
    lists or one-dimensional |NumPy arrays|. Indexing will always return a new
    |VectorArray| which acts as a view into the original data. Thus, if the indexed
    array is modified via :meth:`~VectorArrayInterface.scal` or :meth:`~VectorArrayInterface.axpy`,
    the vectors in the original array will be changed. Indices may be negative, in
    which case the vector is selected by counting from the end of the array. Moreover
    indices can be repeated, in which case the corresponding vector is selected several
    times. The resulting view will be immutable, however.

    .. note::
        It is disallowed to append vectors to a |VectorArray| view or to remove
        vectors from it. Removing vectors from an array with existing views
        will lead to undefined behavior of these views. As such, it is generally
        advisable to make a :meth:`~VectorArrayInterface.copy` of a view for long
        term storage. Since :meth:`~VectorArrayInterface.copy` has copy-on-write
        semantics, this will usually cause little overhead.

    Attributes
    ----------
    data
        Implementors can provide a `data` property which returns a |NumPy array| of
        shape `(len(v), v.dim)` containing the data stored in the array. Access should
        be assumed to be slow and is mainly intended for debugging / visualization
        purposes or to once transfer data to pyMOR and further process it using NumPy.
        In the case of |NumpyVectorArray|, an actual view of the internally used
        |NumPy array| is returned, so changing it, will alter the |VectorArray|.
        Thus, you cannot assume to own the data returned to you, in general.

    dim
        The dimension of the vectors in the array.
    is_view
        `True` if the array is a view obtained by indexing another array.
    space
        The |VectorSpace| the array belongs to.
    """

    is_view = False

    def zeros(self, count=1, reserve=0):
        """Create a |VectorArray| of null vectors of the same |VectorSpace|.

        This is a shorthand for `self.space.zeros(count, reserve)`.

        Parameters
        ----------
        count
            The number of vectors.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        A |VectorArray| containing `count` vectors whith each component
        zero.
        """
        return self.space.zeros(count, reserve=reserve)

    def empty(self, reserve=0):
        """Create an empty |VectorArray| of the same |VectorSpace|.

        This is a shorthand for `self.space.zeros(0, reserve)`.

        Parameters
        ----------
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        An empty |VectorArray|.
        """
        return self.space.zeros(0, reserve=reserve)

    @property
    def dim(self):
        return self.space.dim

    @abstractmethod
    def __len__(self):
        """The number of vectors in the array."""
        pass

    @abstractmethod
    def __getitem__(self, ind):
        """Return a |VectorArray| view onto a subset of the vectors in the array."""
        pass

    @abstractmethod
    def __delitem__(self, ind):
        """Remove vectors from the array."""
        pass

    @abstractmethod
    def append(self, other, remove_from_other=False):
        """Append vectors to the array.

        Parameters
        ----------
        other
            A |VectorArray| containing the vectors to be appended.
        remove_from_other
            If `True`, the appended vectors are removed from `other`.
            For list-like implementations this can be used to prevent
            unnecessary copies of the involved vectors.
        """
        pass

    @abstractmethod
    def copy(self, deep=False):
        """Returns a copy of a subarray.

        All |VectorArray| implementations in pyMOR have copy-on-write semantics:
        if not specified otherwise by setting `deep` to `True`, the returned
        copy will hold a handle to the same array data as the original array,
        and a deep copy of the data will only be performed when one of the arrays
        is modified.

        Note that for |NumpyVectorArray|, a deep copy is always performed when only
        some vectors in the array are copied.

        Parameters
        ----------
        deep
            Ensure that an actual copy of the array data is made (see above).

        Returns
        -------
        A copy of the |VectorArray|.
        """
        pass

    @abstractmethod
    def scal(self, alpha):
        """BLAS SCAL operation (in-place scalar multiplication).

        This method calculates ::

            self = alpha*self

        If `alpha` is a scalar, each vector is multiplied by this scalar. Otherwise, `alpha`
        has to be a one-dimensional |NumPy array| of the same length as `self`
        containing the factors for each vector.

        Parameters
        ----------
        alpha
            The scalar coefficient or one-dimensional |NumPy array| of coefficients
            with which the vectors in `self` are multiplied.
        """
        pass

    @abstractmethod
    def axpy(self, alpha, x):
        """BLAS AXPY operation.

        This method forms the sum ::

            self = alpha*x + self

        If the length of `x` is 1, the same `x` vector is used for all vectors
        in `self`. Otherwise, the lengths of `self`  and `x` have to agree.
        If `alpha` is a scalar, each `x` vector is multiplied with the same factor `alpha`.
        Otherwise, `alpha` has to be a one-dimensional |NumPy array| of the same length as
        `self` containing the coefficients for each `x` vector.

        Parameters
        ----------
        alpha
            The scalar coefficient or one-dimensional |NumPy array| of coefficients with which
            the vectors in `x` are multiplied.
        x
            A |VectorArray| containing the x-summands.
        """
        pass

    @abstractmethod
    def dot(self, other):
        """Returns the inner products between |VectorArray| elements.

        Parameters
        ----------
        other
            A |VectorArray| containing the second factors.

        Returns
        -------
        A |NumPy array| `result` such that:

            result[i, j] = ( self[i], other[j] ).

        """
        pass

    @abstractmethod
    def pairwise_dot(self, other):
        """Returns the pairwise inner products between |VectorArray| elements.

        Parameters
        ----------
        other
            A |VectorArray| containing the second factors.

        Returns
        -------
        A |NumPy array| `result` such that:

            result[i] = ( self[i], other[i] ).

        """
        pass

    @abstractmethod
    def lincomb(self, coefficients):
        """Returns linear combinations of the vectors contained in the array.

        Parameters
        ----------
        coefficients
            A |NumPy array| of dimension 1 or 2 containing the linear
            coefficients. `coefficients.shape[-1]` has to agree with
            `len(self)`.

        Returns
        -------
        A |VectorArray| `result` such that:

            result[i] = ∑ self[j] * coefficients[i,j]

        in case `coefficients` is of dimension 2, otherwise
        `len(result) == 1` and

            result[0] = ∑ self[j] * coefficients[j].
        """
        pass

    @abstractmethod
    def l1_norm(self):
        """The l1-norms of the vectors contained in the array.

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[i]`.
        """
        pass

    @abstractmethod
    def l2_norm(self):
        """The l2-norms of the vectors contained in the array.

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[i]`.
        """
        pass

    @abstractmethod
    def l2_norm2(self):
        """The squared l2-norms of the vectors contained in the array.

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[i]`.
        """
        pass

    def sup_norm(self):
        """The l-infinity--norms of the vectors contained in the array.

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[i]`.
        """
        if self.dim == 0:
            return np.zeros(len(self))
        else:
            _, max_val = self.amax()
            return max_val

    @abstractmethod
    def components(self, component_indices):
        """Extract components of the vectors contained in the array.

        Parameters
        ----------
        component_indices
            List or 1D |NumPy array| of indices of the vector components that are to
            be returned.

        Returns
        -------
        A |NumPy array| `result` such that `result[i, j]` is the `component_indices[j]`-th
        component of the `i`-th vector of the array.
        """
        pass

    @abstractmethod
    def amax(self):
        """The maximum absolute value of the vectors contained in the array.

        Returns
        -------
        max_ind
            |NumPy array| containing for each vector an index at which the maximum is
            attained.
        max_val
            |NumPy array| containing for each vector the maximum absolute value of its
            components.
        """
        pass

    def gramian(self):
        """Shorthand for `self.dot(self)`."""
        return self.dot(self)

    def __add__(self, other):
        """The pairwise sum of two |VectorArrays|."""
        if isinstance(other, Number):
            assert other == 0
            return self.copy()

        result = self.copy()
        result.axpy(1, other)
        return result

    def __iadd__(self, other):
        """In-place pairwise addition of |VectorArrays|."""
        self.axpy(1, other)
        return self

    __radd__ = __add__

    def __sub__(self, other):
        """The pairwise difference of two |VectorArrays|."""
        result = self.copy()
        result.axpy(-1, other)
        return result

    def __isub__(self, other):
        """In-place pairwise difference of |VectorArrays|."""
        self.axpy(-1, other)
        return self

    def __mul__(self, other):
        """Product by a scalar."""
        result = self.copy()
        result.scal(other)
        return result

    def __imul__(self, other):
        """In-place product by a scalar."""
        self.scal(other)
        return self

    def __neg__(self):
        """Product by -1."""
        result = self.copy()
        result.scal(-1)
        return result

    def check_ind(self, ind):
        """Check if `ind` is an admissable list of indices in the sense of the class documentation."""
        l = len(self)
        return (type(ind) is slice or
                isinstance(ind, _INDEXTYPES) and -l <= ind < l or
                isinstance(ind, (list, np.ndarray)) and all(-l <= i < l for i in ind))

    def check_ind_unique(self, ind):
        """Check if `ind` is an admissable list of non-repeated indices in the sense of the class documentation."""
        l = len(self)
        return (type(ind) is slice or
                isinstance(ind, _INDEXTYPES) and -l <= ind < l or
                isinstance(ind, (list, np.ndarray)) and len(set(i if i >= 0 else l+i for i in ind if -l <= i < l)) == len(ind))

    def len_ind(self, ind):
        """Return the number of given indices."""
        l = len(self)
        return (len(range(*ind.indices(l))) if type(ind) is slice else
                1 if not hasattr(ind, '__len__') else
                len(ind))

    def len_ind_unique(self, ind):
        """Return the number of specified unique indices."""
        l = len(self)
        return (len(range(*ind.indices(l))) if type(ind) is slice else
                1 if isinstance(ind, _INDEXTYPES) else
                len(set(i if i >= 0 else l+i for i in ind)))

    def normalize_ind(self, ind):
        """Normalize given indices such that they are independent of the array length."""
        if type(ind) is slice:
            return slice(*ind.indices(len(self)))
        elif not hasattr(ind, '__len__'):
            ind = ind if 0 <= ind else len(self)+ind
            return slice(ind, ind+1)
        else:
            l = len(self)
            return [i if 0 <= i else l+i for i in ind]

    def sub_index(self, ind, ind_ind):
        """Return indices corresponding to the view `self[ind][ind_ind]`"""
        if type(ind) is slice:
            ind = range(*ind.indices(len(self)))
            if type(ind_ind) is slice:
                if config.PY2:  # make ind a list since is xrange is not sliceable
                    return list(ind)[ind_ind]
                else:
                    result = ind[ind_ind]
                    return slice(result.start, result.stop, result.step)
            elif hasattr(ind_ind, '__len__'):
                return [ind[i] for i in ind_ind]
            else:
                return [ind[ind_ind]]
        else:
            if not hasattr(ind, '__len__'):
                ind = [ind]
            if type(ind_ind) is slice:
                return ind[ind_ind]
            elif hasattr(ind_ind, '__len__'):
                return [ind[i] for i in ind_ind]
            else:
                return [ind[ind_ind]]


class VectorSpaceInterface(ImmutableInterface):
    """Class describing a vector space.

    Vector spaces act as factories for |VectorArrays| of vectors
    contained in them. As such, they hold all data necessary to
    create |VectorArrays| of a given type (e.g. the dimension of
    the vectors, or a socket for communication with an external
    PDE solver).

    New |VectorArrays| of null vectors are created via
    :meth:`~VectorSpaceInterface.zeros`.  The
    :meth:`~VectorSpaceInterface.make_array` method builds a new
    |VectorArray| from given raw data of the underlying linear algebra
    backend (e.g. a |Numpy array| in the case  of |NumpyVectorSpace|).
    Some vector spaces can create new |VectorArrays| from a given
    |Numpy array| via the :meth:`~VectorSpaceInterface.from_data`
    method.

    Each vector space has a string :attr:`~VectorSpaceInterface.id`
    to distinguish mathematically different spaces appearing
    in the formulation of a given problem.

    Vector spaces can be compared for equality via the `==` and `!=`
    operators. To test if a given |VectorArray| is an element of
    the space, the `in` operator can be used.


    Attributes
    ----------
    id
        None, or a string describing the mathematical identity
        of the vector space (for instance to distinguish different
        components in an equation system).
    dim
        The dimension (number of degrees of freedom) of the
        vectors contained in the space.
    is_scalar
        Equivalent to `isinstance(space, NumpyVectorSpace) and space.dim == 1`.
    """

    id = None
    dim = None
    is_scalar = False

    @abstractmethod
    def make_array(*args, **kwargs):
        """Create a |VectorArray| from raw data.

        This method is used in the implementation of |Operators|
        and |Discretizations| to create new |VectorArrays| from
        raw data of the underlying solver backends. The ownership
        of the data is transferred to the newly created array.

        The exact signature of this method depends on the wrapped
        solver backend.
        """
        pass

    @abstractmethod
    def zeros(self, count=1, reserve=0):
        """Create a |VectorArray| of null vectors

        Parameters
        ----------
        count
            The number of vectors.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        A |VectorArray| containing `count` vectors with each component zero.
        """
        pass

    def empty(self, reserve=0):
        """Create an empty |VectorArray|

        This is a shorthand for `self.zeros(0, reserve)`.

        Parameters
        ----------
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        An empty |VectorArray|.
        """
        return self.zeros(0, reserve=reserve)

    def from_data(self, data):
        """Create a |VectorArray| from a |NumPy array|

        Note that this method will not be supported by all vector
        space implementations.

        Parameters
        ----------
        data
            |NumPy| array.

        Returns
        -------
        A |VectorArray| with `data` as data.
        """
        raise NotImplementedError

    def __eq__(self, other):
        return other is self

    def __ne__(self, other):
        return not (self == other)

    def __contains__(self, other):
        return self == getattr(other, 'space', None)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.id)
