# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractproperty, abstractclassmethod


class VectorArrayInterface(BasicInterface):
    '''Interface for vector arrays.

    A vector array should be thought of as a list of (possibly high-dimensional) vectors.
    While the vectors themselves will be inaccessible in general (e.g. because they are
    managed by external code on large systems), operations on the vectors like addition can
    be performed via the interface.

    It is moreover assumed that the number of vectors is small enough such that scalar data
    associated to each vector can be handled on the python side. I.e. methods like
    :meth:`~VectorArrayInterface.l2_norm` or :meth:`~VectorArrayInterface.gramian` will
    always return |NumPy arrays|.

    An implementation of the interface via |NumPy arrays| is given by |NumpyVectorArray|.
    In general, it is the implementors decision how memory is allocated internally (e.g.
    continuous block of memory vs. list of pointers to the individual vectors.) Thus no
    general assumptions can be made on the costs of operations like appending or removing
    vectors from the array. As a hint for 'continuous block of memory' implementations,
    |VectorArray| constructors should provide a `reserve` keyword argument which allows
    to specify to what sizes the array is assumed to grow.

    Most methods provide `ind` and/or `o_ind` arguments which are used to specify on which
    vectors the method is supposed to operate. If `ind` (`o_ind`) is `None` the whole array
    is selected. Otherwise, `ind` can be a single index in `range(len(self))`, a `list`
    of indices or a one-dimensional |NumPy array| of indices. One index can be repeated
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
        Thus you cannot assume to own the data returned to you, in general.

    dim
        The dimension of the vectors in the array.
    '''

    @abstractclassmethod
    def empty(cls, dim, reserve=0):
        '''Create an empty |VectorArray|

        Parameters
        ----------
        dim
            The dimension of the array.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        An empty |VectorArray|.
        '''
        pass

    @abstractclassmethod
    def zeros(cls, dim, count=1):
        '''Create a |VectorArray| of null vectors

        Parameters
        ----------
        dim
            The dimension of the array.
        count
            The number of vectors.

        Returns
        -------
        A |VectorArray| containing `count` vectors of dimension `dim`
        whith each component zero.
        '''
        pass

    @abstractmethod
    def __len__(self):
        '''The number of vectors in the array.'''
        pass

    @abstractproperty
    def dim(self):
        pass

    @abstractmethod
    def copy(self, ind=None):
        '''Returns a copy of a subarray.

        Parameters
        ----------
        ind
            Indices of the vectors that are to be copied (see class documentation).

        Returns
        -------
        A copy of the |VectorArray|.
        '''
        pass

    @abstractmethod
    def append(self, other, o_ind=None, remove_from_other=False):
        '''Append vectors to the array.

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
        '''
        pass

    @abstractmethod
    def remove(self, ind=None):
        '''Remove vectors from the array.

        Parameters
        ----------
        ind
            Indices of the vectors that are to be removed (see class documentation).
        '''
        pass

    @abstractmethod
    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        '''Replace vectors of the array.

        Parameters
        ----------
        other
            A |VectorArray| containing the replacement vectors.
        ind
            Indices of the vectors that are to be replaced (see class documentation).
            Repeated indices are forbidden.
        o_ind
            Indices of the replacement vectors (see class documentation).
            `len(ind)` has to agree with `len(o_ind)`.
            Repeated indices are allowed.
        remove_from_other
            If `True`, the new vectors are removed from `other`.
            For list-like implementations this can be used to prevent
            unnecessary copies of the involved vectors.
        '''
        pass

    @abstractmethod
    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        '''Check vectors for equality.

        Equality of two vectors should be defined as in
        :func:`pymor.tools.float_cmp_all`.

        The dimensions of `self` and `other` have to agree. If the length
        of `self` (`ind`) resp. `other` (`o_ind`) is 1, the one specified
        vector is compared to all vectors of the other summand. Otherwise
        the length of `ind` and `o_ind` have to agree.

        Parameters
        ----------
        other
            A |VectorArray| containing the vectors to compare with.
        ind
            Indices of the vectors that are to be compared (see class documentation).
        o_ind
            Indices of the vectors in `other` that are to be compared (see class documentation).
        rtol
            See :func:`pymor.tools.float_cmp_all`
        atol
            See :func:`pymor.tools.float_cmp_all`

        Returns
        -------
        |NumPy array| of the truth values of the comparison.
        '''
        pass

    @abstractmethod
    def scal(self, alpha, ind=None):
        '''BLAS SCAL operation (in-place scalar multiplication).

        This method calculates ::

            self[ind] = alpha*self[ind]

        Parameters
        ----------
        alpha
            The scalar coefficient with which the vectors in `self` are multiplied
        ind
            Indices of the vectors of `self` that are to be scaled (see class documentation).
            Repeated indices are forbidden.
        '''
        pass

    @abstractmethod
    def axpy(self, alpha, x, ind=None, x_ind=None):
        '''BLAS AXPY operation.

        This method forms the sum ::

            self[ind] = alpha*x[x_ind] + self[ind]

        The dimensions of `self` and `x` as well as the lengths of `self` (`ind`) and
        `x` (`x_ind`) have to agree.

        Parameters
        ----------
        alpha
            The scalar coefficient with which the vectors in `x` are multiplied
        x
            A |VectorArray| containing the x-summands.
        ind
            Indices of the vectors of `self` that are to be added (see class documentation).
            Repeated indices are forbidden.
        x_ind
            Indices of the vectors in `x` that are to be added (see class documentation).
            Repeated indices are allowed.
        '''
        pass

    @abstractmethod
    def dot(self, other, pairwise, ind=None, o_ind=None):
        '''Returns the scalar products between |VectorArray| elements.

        Parameters
        ----------
        other
            A |VectorArray| containing the second factors.
        pairwise
            See return value documentation.
        ind
            Indices of the vectors whose scalar products are to be taken
            (see class documentation).
        o_ind
            Indices of the vectors in `other` whose scalar products are to be
            taken (see class documentation).

        Returns
        -------
        If pairwise is `True`, returns a |NumPy array| `result` such
        that ::

            result[i] = ( self[ind][i], other[o_ind][i] ).

        If pairwise is `False`, returns a |NumPy array| `result` such
        that ::

            result[i, j] = ( self[ind][i], other[o_ind][j] ).
        '''
        pass

    @abstractmethod
    def lincomb(self, coefficients, ind=None):
        '''Returns linear combinations of the vectors contained in the array.

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
        A |VectorArray| `result` such that ::

            result[i] = ∑ self[j] * coefficients[i,j]

        in case `coefficients` is of dimension 2, otherwise
        `len(result) == 1` and

            result[1] = ∑ self[j] * coefficients[j].
        '''
        pass

    @abstractmethod
    def l1_norm(self, ind=None):
        '''The l1-norms of the vectors contained in the array.

        Parameters
        ----------
        ind
            Indices of the vectors whose norm is to be calculated (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[ind][i]`.
        '''
        pass

    @abstractmethod
    def l2_norm(self, ind=None):
        '''The l2-norms of the vectors contained in the array.

        Parameters
        ----------
        ind
            Indices of the vectors whose norm is to be calculated (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[ind][i]`.
        '''
        pass

    def sup_norm(self, ind=None):
        '''The l-infinity--norms of the vectors contained in the array.

        Parameters
        ----------
        ind
            Indices of the vectors whose norm is to be calculated (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[ind][i]`.
        '''
        if self.dim == 0:
            assert self.check_ind(ind)
            return np.zeros(self.len_ind(ind))
        else:
            _, max_val = self.amax(ind)
            return max_val

    @abstractmethod
    def components(self, component_indices, ind=None):
        '''Extract components of the vectors contained in the array.

        Parameters
        ----------
        component_indices
            List or 1D |NumPy array| of indices of the vector components that are to
            be returned.
        ind
            Indices of the vectors whose components to be calculated (see class documentation).

        Returns
        -------
        A |NumPy array| `result` such that `result[i, j]` is the `component_indices[j]`-th
        component of the `ind[i]`-th vector of the array.
        '''
        pass

    @abstractmethod
    def amax(self, ind=None):
        '''The maximum absolute value of the vectors contained in the array.

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
        '''
        pass

    def gramian(self, ind=None):
        '''Shorthand for `dot(self, pairwise=False, ind=ind, o_ind=ind)`.'''
        return self.dot(self, pairwise=False, ind=ind, o_ind=ind)

    def __add__(self, other):
        '''The pairwise sum of two |VectorArrays|.'''
        if isinstance(other, Number):
            assert other == 0
            return self.copy()

        result = self.copy()
        result.axpy(1, other)
        return result

    def __iadd__(self, other):
        '''In-place pairwise addition of |VectorArrays|.'''
        self.axpy(1, other)
        return self

    __radd__ = __add__

    def __sub__(self, other):
        '''The pairwise difference of two |VectorArrays|.'''
        result = self.copy()
        result.axpy(-1, other)
        return result

    def __isub__(self, other):
        '''In-place pairwise difference of |VectorArrays|.'''
        self.axpy(-1, other)
        return self

    def __mul__(self, other):
        '''Product by a scalar.'''
        result = self.copy()
        result.scal(other)
        return result

    def __imul__(self, other):
        '''In-place product by a scalar.'''
        self.scal(other)
        return self

    def __neg__(self):
        '''Product by -1.'''
        result = self.copy()
        result.scal(-1)
        return result

    def check_ind(self, ind):
        '''Check if `ind` is an admissable list of indices in the sense of the class documentation.'''
        return (ind is None or
                isinstance(ind, Number) and 0 <= ind < len(self) or
                isinstance(ind, list) and (len(ind) == 0 or 0 <= min(ind) and max(ind) < len(self)) or
                (isinstance(ind, np.ndarray) and ind.ndim == 1
                 and (len(ind) == 0 or 0 <= np.min(ind) and np.max(ind) < len(self))))

    def check_ind_unique(self, ind):
        '''Check if `ind` is an admissable list of unique indices in the sense of the class documentation.'''
        if (ind is None or isinstance(ind, Number) and 0 <= ind < len(self)):
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
        '''Return the number of specified indices.'''
        return len(self) if ind is None else 1 if isinstance(ind, Number) else len(ind)

    def len_ind_unique(self, ind):
        '''Return the number of specified unique indices.'''
        return len(self) if ind is None else 1 if isinstance(ind, Number) else len(set(ind))
