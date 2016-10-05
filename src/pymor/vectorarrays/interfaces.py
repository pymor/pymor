# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractproperty, abstractclassmethod

def _numpy_version_older(version_tuple):
    np_tuple = tuple(int(p) for p in np.__version__.split('.')[:3])
    return np_tuple < version_tuple

_INDEXTYPES = (Number,) if not _numpy_version_older((1,9)) else (Number, np.intp)

class VectorArrayInterface(BasicInterface):
    """Interface for vector arrays.

    A vector array should be thought of as a list of (possibly high-dimensional) vectors.
    While the vectors themselves will be inaccessible in general (e.g. because they are
    managed by an external PDE solver code), operations on the vectors like addition can
    be performed via the interface.

    It is moreover assumed that the number of vectors is small enough such that scalar data
    associated to each vector can be handled on the Python side. As such, methods like
    :meth:`~VectorArrayInterface.l2_norm` or :meth:`~VectorArrayInterface.gramian` will
    always return |NumPy arrays|.

    An implementation of the interface via |NumPy arrays| is given by |NumpyVectorArray|.
    In general, it is the implementors decision how memory is allocated internally (e.g.
    continuous block of memory vs. list of pointers to the individual vectors.) Thus, no
    general assumptions can be made on the costs of operations like appending to or removing
    vectors from the array. As a hint for 'continuous block of memory' implementations,
    |VectorArray| constructors should provide a `reserve` keyword argument which allows
    to specify to what size the array is assumed to grow.

    Most methods provide `ind` and/or `o_ind` arguments which are used to specify on which
    vectors the method is supposed to operate. If `ind` (`o_ind`) is `None` the whole array
    is selected. Otherwise, `ind` can be a single index in `range(len(self))`, a `list`
    of indices or a one-dimensional |NumPy array| of indices. An index can be repeated
    in which case the corresponding vector is selected several times.

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
    space
        |VectorSpace| array the array belongs to.
    subtype
        Can be any Python object. Two arrays are compatible (e.g. can be added)
        if they are instances of the same class and have equal subtypes. A valid
        subtype has to be provided to :meth:`~VectorArrayInterface.make_array` and
        the resulting array will be of that subtype. By default, the subtype of an
        array is simply `None`. For |NumpyVectorArray|, the subtype is a single
        integer denoting the dimension of the array. Subtypes for other array classes
        could, e.g., include a socket for communication with a specific PDE solver
        instance.
    """

    @abstractclassmethod
    def make_array(cls, subtype=None, count=0, reserve=0):
        """Create a |VectorArray| of null vectors.

        Parameters
        ----------
        subtype
            The :attr:`~VectorArrayInterface.subtype`, the created array should have.
            What a valid subtype is, is determined by the respective array implementation.
        count
            The number of null vectors to create. For `count == 0`, an empty array is
            returned.
        reserve
            A hint for the backend to which length the array will grow.
        """
        pass

    @classmethod
    def from_data(cls, data, subtype):
        """Create a |VectorArray| from |NumPy array|

        Parameters
        ----------
        data
            |NumPy array|.
        subtype
            The :attr:`~VectorArrayInterface.subtype`, the created array should have.
            What a valid subtype is, is determined by the respective array implementation.
        """
        raise NotImplementedError

    def empty(self, reserve=0):
        """Create an empty |VectorArray| of the same :attr:`~VectorArrayInterface.subtype`.

        Parameters
        ----------
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        An empty |VectorArray|.
        """
        return self.make_array(subtype=self.subtype, reserve=reserve)

    def zeros(self, count=1):
        """Create a |VectorArray| of null vectors of the same :attr:`~VectorArrayInterface.subtype`.

        Parameters
        ----------
        count
            The number of vectors.

        Returns
        -------
        A |VectorArray| containing `count` vectors whith each component
        zero.
        """
        return self.make_array(subtype=self.subtype, count=count)

    @abstractmethod
    def __len__(self):
        """The number of vectors in the array."""
        pass

    @abstractproperty
    def dim(self):
        pass

    @property
    def subtype(self):
        return None

    @property
    def space(self):
        return VectorSpace(type(self), self.subtype)

    @abstractmethod
    def copy(self, ind=None, deep=False):
        """Returns a copy of a subarray.

        All |VectorArray| implementations in pyMOR have copy-on-write semantics:
        if not specified otherwise by setting `deep` to `True`, the returned
        copy will hold a handle to the same array data as the original array,
        and a deep copy of the data will only be performed when one of the arrays
        is modified.

        Note that for |NumpyVectorArray|, a deep copy is always performed when only
        some vectors in array are copied (i.e. `ind` is specified).

        Parameters
        ----------
        ind
            Indices of the vectors that are to be copied (see class documentation).
        deep
            Ensure that an actual copy of the array data is made (see above).

        Returns
        -------
        A copy of the |VectorArray|.
        """
        pass

    @abstractmethod
    def append(self, other, o_ind=None, remove_from_other=False):
        """Append vectors to the array.

        Parameters
        ----------
        other
            A |VectorArray| containing the vectors to be appended.
        o_ind
            Indices of the vectors that are to be appended (see class documentation).
        remove_from_other
            If `True`, the appended vectors are removed from `other`.
            For list-like implementations this can be used to prevent
            unnecessary copies of the involved vectors.
        """
        pass

    @abstractmethod
    def remove(self, ind=None):
        """Remove vectors from the array.

        Parameters
        ----------
        ind
            Indices of the vectors that are to be removed (see class documentation).
        """
        pass

    @abstractmethod
    def scal(self, alpha, ind=None):
        """BLAS SCAL operation (in-place scalar multiplication).

        This method calculates ::

            self[ind] = alpha*self[ind]

        If `alpha` is a scalar, each vector is multiplied by this scalar. Otherwise, `alpha`
        has to be a one-dimensional |NumPy array| of the same length as `self` (`ind`)
        containing the factors for each vector.

        Parameters
        ----------
        alpha
            The scalar coefficient or one-dimensional |NumPy array| of coefficients
            with which the vectors in `self` are multiplied.
        ind
            Indices of the vectors of `self` that are to be scaled (see class documentation).
            Repeated indices are forbidden.
        """
        pass

    @abstractmethod
    def axpy(self, alpha, x, ind=None, x_ind=None):
        """BLAS AXPY operation.

        This method forms the sum ::

            self[ind] = alpha*x[x_ind] + self[ind]

        If the length of `x` (`x_ind`) is 1, the same `x` vector is used for all vectors
        in `self`. Otherwise, the lengths of `self` (`ind`) and `x` (`x_ind`) have to agree.
        If `alpha` is a scalar, each `x` vector is multiplied with the same factor `alpha`.
        Otherwise, `alpha` has to be a one-dimensional |NumPy array| of the same length as
        `self` (`ind`)  containing the coefficients for each `x` vector.

        Parameters
        ----------
        alpha
            The scalar coefficient or one-dimensional |NumPy array| of coefficients with which
            the vectors in `x` are multiplied.
        x
            A |VectorArray| containing the x-summands.
        ind
            Indices of the vectors of `self` that are to be added (see class documentation).
            Repeated indices are forbidden.
        x_ind
            Indices of the vectors in `x` that are to be added (see class documentation).
            Repeated indices are allowed.
        """
        pass

    @abstractmethod
    def dot(self, other, ind=None, o_ind=None):
        """Returns the inner products between |VectorArray| elements.

        Parameters
        ----------
        other
            A |VectorArray| containing the second factors.
        ind
            Indices of the vectors whose inner products are to be taken
            (see class documentation).
        o_ind
            Indices of the vectors in `other` whose inner products are to be
            taken (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that:

            result[i, j] = ( self[ind[i]], other[o_ind[j]] ).

        """
        pass

    @abstractmethod
    def pairwise_dot(self, other, ind=None, o_ind=None):
        """Returns the pairwise inner products between |VectorArray| elements.

        Parameters
        ----------
        other
            A |VectorArray| containing the second factors.
        ind
            Indices of the vectors whose inner products are to be taken
            (see class documentation).
        o_ind
            Indices of the vectors in `other` whose inner products are to be
            taken (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that:

            result[i] = ( self[ind[i]], other[o_ind[i]] ).

        """
        pass

    @abstractmethod
    def lincomb(self, coefficients, ind=None):
        """Returns linear combinations of the vectors contained in the array.

        Parameters
        ----------
        coefficients
            A |NumPy array| of dimension 1 or 2 containing the linear
            coefficients. `coefficients.shape[-1]` has to agree with
            `len(self)`.
        ind
            Indices of the vectors which are linear combined (see class documentation).

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
    def l1_norm(self, ind=None):
        """The l1-norms of the vectors contained in the array.

        Parameters
        ----------
        ind
            Indices of the vectors whose norms are to be calculated (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[ind[i]]`.
        """
        pass

    @abstractmethod
    def l2_norm(self, ind=None):
        """The l2-norms of the vectors contained in the array.

        Parameters
        ----------
        ind
            Indices of the vectors whose norms are to be calculated (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[ind[i]]`.
        """
        pass

    @abstractmethod
    def l2_norm2(self, ind=None):
        """The squared l2-norms of the vectors contained in the array.

        Parameters
        ----------
        ind
            Indices of the vectors whose norms are to be calculated (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[ind][i]`.
        """
        pass

    def sup_norm(self, ind=None):
        """The l-infinity--norms of the vectors contained in the array.

        Parameters
        ----------
        ind
            Indices of the vectors whose norms are to be calculated (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[ind[i]]`.
        """
        if self.dim == 0:
            assert self.check_ind(ind)
            return np.zeros(self.len_ind(ind))
        else:
            _, max_val = self.amax(ind)
            return max_val

    @abstractmethod
    def components(self, component_indices, ind=None):
        """Extract components of the vectors contained in the array.

        Parameters
        ----------
        component_indices
            List or 1D |NumPy array| of indices of the vector components that are to
            be returned.
        ind
            Indices of the vectors whose components are to be retrieved (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that `result[i, j]` is the `component_indices[j]`-th
        component of the `ind[i]`-th vector of the array.
        """
        pass

    @abstractmethod
    def amax(self, ind=None):
        """The maximum absolute value of the vectors contained in the array.

        Parameters
        ----------
        ind
            Indices of the vectors whose maximum absolute value is to be calculated
            (see class documentation).

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

    def gramian(self, ind=None):
        """Shorthand for `self.dot(self, ind=ind, o_ind=ind)`."""
        return self.dot(self, ind=ind, o_ind=ind)

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
        return (ind is None or
                isinstance(ind, _INDEXTYPES) and 0 <= ind < len(self) or
                isinstance(ind, list) and (len(ind) == 0 or 0 <= min(ind) and max(ind) < len(self)) or
                (isinstance(ind, np.ndarray) and ind.ndim == 1
                 and (len(ind) == 0 or 0 <= np.min(ind) and np.max(ind) < len(self))))

    def check_ind_unique(self, ind):
        """Check if `ind` is an admissable list of non-repeated indices in the sense of the class documentation."""
        if ind is None or isinstance(ind, Number) and 0 <= ind < len(self):
            return True
        elif isinstance(ind, list):
            if len(ind) == 0:
                return True
            s = set(ind)
            return len(s) == len(ind) and 0 <= min(s) and max(s) < len(self)
        elif isinstance(ind, np.ndarray) and ind.ndim == 1:
            if len(ind) == 0:
                return True
            u = np.unique(ind)
            return len(u) == len(ind) and 0 <= u[0] and u[-1] < len(self)
        else:
            return False

    def len_ind(self, ind):
        """Return the number of specified indices."""
        return len(self) if ind is None else 1 if isinstance(ind, _INDEXTYPES) else len(ind)

    def len_ind_unique(self, ind):
        """Return the number of specified unique indices."""
        return len(self) if ind is None else 1 if isinstance(ind, Number) else len(set(ind))


class VectorSpace(BasicInterface):
    """Class describing a vector space.

    A vector space is simply the combination of a |VectorArray| type and a
    :attr:`~VectorArrayInterface.subtype`. This data is exactly sufficient to construct
    new arrays using the :meth:`~VectorArrayInterface.make_array` method
    (see the implementation of :meth:`~VectorSpace.zeros`).

    A |VectorArray| `U` is contained in a vector space `space`, if ::

        type(U) == space.type and U.subtype == space.subtype


    Attributes
    ----------
    type
        The type of |VectorArrays| in the space.
    subtype
        The subtype used to construct arrays of the given space.
    """

    def __init__(self, space_type, subtype=None):
        self.type = space_type
        self.subtype = subtype

    def empty(self, reserve=0):
        """Create an empty |VectorArray|

        Parameters
        ----------
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        An empty |VectorArray|.
        """
        return self.type.make_array(subtype=self.subtype, reserve=reserve)

    def zeros(self, count=1):
        """Create a |VectorArray| of null vectors

        Parameters
        ----------
        count
            The number of vectors.

        Returns
        -------
        A |VectorArray| containing `count` vectors with each component zero.
        """
        return self.type.make_array(subtype=self.subtype, count=count)

    def from_data(self, data):
        """Create a |VectorArray| from a |NumPy array|

        Parameters
        ----------
        data
            |NumPy| array.

        Returns
        -------
        A |VectorArray| with `data` as data.
        """
        return self.type.from_data(data, self.subtype)

    @property
    def dim(self):
        return self.empty().dim

    def __eq__(self, other):
        """Two spaces are equal iff their types and subtypes agree."""
        return other.type == self.type and self.subtype == other.subtype

    def __ne__(self, other):
        """Two spaces are equal iff their types and subtypes agree."""
        return other.type != self.type or self.subtype != other.subtype

    def __contains__(self, other):
        """A |VectorArray| is contained in the space, iff it is an instance of its type and has the same subtype."""
        return isinstance(other, self.type) and self.subtype == other.subtype

    def __repr__(self):
        return 'VectorSpace({}, {})'.format(self.type.__name__, repr(self.subtype))

    __str__ = __repr__
