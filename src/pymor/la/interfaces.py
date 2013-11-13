# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
from scipy.sparse import issparse

from pymor.core.exceptions import CommunicationError
from pymor.core.interfaces import BasicInterface, abstractmethod, abstractproperty, abstractclassmethod


class Communicable(BasicInterface):

    @abstractmethod
    def _data(self):
        pass

    _communication = 'enable'
    @property
    def communication(self):
        return self._communication

    @communication.setter
    def communication(self, v):
        assert v in set('raise', 'warn', 'enable')
        self._communication = v

    def enable_communication(self):
        self.communication = 'enable'

    def disable_communication(self):
        self.communication = 'raise'

    @property
    def data(self):
        '''Returns a `numpy.ndarray` containing the matrix.

        In case, the content of the `Matrix` cannot be modified via
        the array, the `WRITEABLE` flag has to be set to false in
        order to ensure, that copies are made.
        '''
        if self._communication == 'enable':
            return self._data()
        elif self._communication == 'warn':
            logger = getLogger('pymor.la.vectorarray')
            logger.warn('Communication required for {}'.format(self))
            return self._data()
        else:
            raise CommunicationError


class VectorArrayInterface(BasicInterface):
    '''Interface for vector arrays.

    A vector array should be thought of as a list of (possibly high-dimensional) vectors.
    While the vectors themselves will be inaccessible in general (e.g. because they are
    managed by external code on large systems), operations on the vectors like addition can
    be performed via the interface.

    It is moreover assumed that the count of vectors is small enough such that scalar data
    associated to each vector can be handled on the python side. I.e. methods like `l2_norm()`
    or `gramian()` will always return numpy arrays.

    An implementation of the interface via numpy arrays is given by `NumpyVectorArray`.
    In general, it is the implementors decision how memory is allocated internally (e.g.
    continuous block of memory vs. list of pointers to the individual vectors.) Thus no
    general assumptions can be made on the costs of operations like appending or removing
    vectors from the array. As a hint for 'continuous block of memory' implementations,
    `VectorArray` constructors should provide a `reserve` keyword argument which allows
    to specify to what sizes the array is assumed to grow.

    Most methods provide `ind` and/or `o_ind` arguments which are used to specify on which
    vectors the method is supposed to operate. If `ind` (`o_ind`) is `None` the whole array
    is selected. Otherwise, `ind` can be a single index in `range(len(self))`, a `list`
    of indices or a one-dimensional numpy array of indices. One index can be repeated
    in which case the corresponding vector is selected several times.
    '''

    @abstractclassmethod
    def empty(cls, dim, reserve=0):
        '''Create an empty VectorArray

        Parameters
        ----------
        dim
            The dimension of the array.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        An empty `VectorArray`.
        '''
        pass

    @abstractclassmethod
    def zeros(cls, dim, count=1):
        '''Create an VectorArray of null vectors

        Parameters
        ----------
        dim
            The dimension of the array.
        count
            The number of vectors.

        Returns
        -------
        An `VectorArray` containing `count` vectors of dimension `dim`
        whith each component zero.
        '''
        pass

    @abstractmethod
    def __len__(self):
        '''The number of vectors in the array.'''
        pass

    @abstractproperty
    def dim(self):
        '''The dimension of the vectors in the `VectorArray`.

        Each vector must have the same dimension.
        '''
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
        A copy of the `VectorArray`.
        '''
        pass

    @abstractmethod
    def append(self, other, o_ind=None, remove_from_other=False):
        '''Append vectors to the array.

        Parameters
        ----------
        other
            A `VectorArray` containing the vectors to be appended.
        o_ind
            Indices of the vectors that are to be appended (see class documentation).
        remove_from_other
            If `True`, the appended vectors are removed from `other`.
            For list-like implementations of `VectorArray` this can be
            used to prevent unnecessary copies of the involved vectors.
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
            A `VectorArray` containing the replacement vectors.
        ind
            Indices of the vectors that are to be replaced (see class documentation).
        o_ind
            Indices of the replacement vectors (see class documentation).
            `len(ind)` has to agree with `len(o_ind)`.
        remove_from_other
            If `True`, the new vectors are removed from `other?.
            For list-like implementations of `VectorArray` this can be
            used to prevent unnecessary copies of the involved vectors.
        '''
        pass

    @abstractmethod
    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        '''Check vectors for equality.

        Equality of two vectors should be defined as in `pymor.tools.float_cmp_all`.

        The dimensions of `self` and `other` have to agree. If the length
        of `self` (`ind`) resp. `other` (`o_ind`) is 1, the one specified
        vector is compared to all vectors of the other summand. Otherwise
        the length of ind and o_ind have to agree.

        Parameters
        ----------
        other
            A `VectorArray` containing the vectors to compare with.
        ind
            Indices of the vectors that are to be compared (see class documentation).
        o_ind
            Indices of the vectors in `other` that are to be compared (see class documentation).
        rtol
            See `pymor.tools.float_cmp_all`
        atol
            See `pymor.tools.float_cmp_all`

        Returns
        -------
        Numpy array of the truth values of the comparison.
        '''
        pass

    @abstractmethod
    def scal(self, alpha, ind=None):
        '''BLAS SCAL operation (in-place sclar multiplication).

        This method calculates ::

            self[ind] = alpha*self[ind]

        Parameters
        ----------
        alpha
            The scalar coefficient with which the vectors in `self` are multiplied
        ind
            Indices of the vectors of `self` that are to be scaled (see class documentation).
        '''
        pass

    @abstractmethod
    def axpy(self, alpha, x, ind=None, x_ind=None):
        '''BLAS AXPY operation.

        This method forms the sum ::

            self[ind] = alpha*x[o_ind] + self[ind]

        If the length of `x` (`o_ind`) is 1, the one specified vector is added to all
        vectors of `self` specified by `ind`. Otherwise, the lengths of `self` (`ind`) and
        `x` (`x_ind`) have to be equal. The dimensions of `self` and `x` have to agree.

        Parameters
        ----------
        alpha
            The scalar coefficient with which the vectors in `x` are multiplied
        x
            A `VectorArray` containing the x-summands.
        ind
            Indices of the vectors of `self` that are to be added (see class documentation).
        x_ind
            Indices of the vectors in `x` that are to be added (see class documentation).
        '''
        pass

    @abstractmethod
    def dot(self, other, pairwise, ind=None, o_ind=None):
        '''Returns the scalar products between `VectorArray` elements.

        Parameters
        ----------
        other
            A `VectorArray` containing the second factors.
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
        If pairwise is True, returns a numpy array `result` such
        that ::

            result[i] = ( self[ind][i], other[o_ind][i] ).

        If pairwise is False, returns a numpy array `result` such
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
            A numpy array of dimension 1 or 2 containing the linear
            coeffcients. `coefficients.shape[-1]` has to agree with
            `len(self)`.
        ind
            Indices of the vectors which are linear combined (see class documentation).

        Returns
        -------
        A `VectorArray` `result` such that ::

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
        A numpy array `result` such that `result[i]` contains the norm
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
        A numpy array `result` such that `result[i]` contains the norm
        of `self[ind][i]`.
        '''
        pass

    def sup_norm(self, ind=None):
        '''The l-infintiy--norms of the vectors contained in the array.

        Parameters
        ----------
        ind
            Indices of the vectors whose norm is to be calculated (see class documentation).

        Returns
        -------
        A numpy array `result` such that `result[i]` contains the norm
        of `self[ind][i]`.
        '''
        _, max_val = self.amax(ind)
        return max_val

    @abstractmethod
    def components(self, component_indices, ind=None):
        '''Extract components of the vectors contained in the array.

        Parameters
        ----------
        component_indices
            Indices of the vector components that are to be returned.
        ind
            Indices of the vectors whose components to be calculated (see class documentation).

        Returns
        -------
        A numpy array `result` such that `result[i, j]` is the `component_indices[j]`-th
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
            Numpy array containing for each vector an index at which the maximum is
            attained.
        max_val
            Numpy array containing for each vector the maximum absolute value of its
            components.
        '''
        pass

    def gramian(self, ind=None):
        '''Shorthand for `dot(self, pairwise=False, ind=ind, o_ind=ind)`.'''
        return self.dot(self, pairwise=False, ind=ind, o_ind=ind)

    def __add__(self, other):
        if isinstance(other, Number):
            assert other == 0
            return self.copy()

        result = self.copy()
        result.axpy(1, other)
        return result

    def __iadd__(self, other):
        self.axpy(1, other)
        return self

    __radd__ = __add__

    def __sub__(self, other):
        result = self.copy()
        result.axpy(-1, other)
        return result

    def __isub__(self, other):
        self.axpy(-1, other)
        return self

    def __mul__(self, other):
        result = self.copy()
        result.scal(other)
        return result

    def __imul__(self, other):
        self.scal(other)
        return self

    def __neg__(self):
        result = self.copy()
        result.scal(-1)
        return result

    def check_ind(self, ind):
        '''Check if `ind` is an admissable list of indices in the sense of the class documentation.'''
        return (ind is None or
                isinstance(ind, Number) and 0 <= ind < len(self) or
                isinstance(ind, list) and (len(ind) == 0 or 0 <= min(ind) and max(ind) < len(self)) or
                isinstance(ind, np.ndarray) and ind.ndim == 1
                                            and (len(ind) == 0 or 0 <= np.min(ind) and np.max(ind) < len(self)))

    def len_ind(self, ind):
        return len(self) if ind is None else 1 if isinstance(ind, Number) else len(ind)

    def len_ind_unique(self, ind):
        return len(self) if ind is None else 1 if isinstance(ind, Number) else len(set(ind))
