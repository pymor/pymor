# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
from scipy.sparse import issparse

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractproperty, abstractclassmethod
from pymor.core.exceptions import CommunicationError


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


class VectorArray(BasicInterface):
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
            If None, a copy of the whole array is returned. Otherwise an
            iterable of the indices of the vectors that are to be copied.

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
            If None, the whole array is appended. Otherwise an iterable
            of the indices of the vectors that are to be appended.
        remove_from_other
            If `True`, the appended vectors are removed from `other`.
            For list-like implementations of `VectorArray` this can be
            used to prevent unnecessary copies of the involved vectors.
        '''
        pass

    @abstractmethod
    def remove(self, ind):
        '''Remove vectors to the array.

        Parameters
        ----------
        ind
            If None, the whole array is emptied. Otherwise an iterable
            of the indices of the vectors that are to be removed.
        '''
        pass

    @abstractmethod
    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        '''Append vectors of the array.

        Parameters
        ----------
        other
            A `VectorArray` containing the replacement vectors.
        ind
            If None, the whole array is replaced. Otherwise an iterable
            of the indices of the vectors that are to be replaced.
        o_ind
            An iterable of the indices of the vectors that are to be
            taken from `other`. If None, the whole array is selected.
        remove_from_other
            If `True`, the new vectors are removed from `other?.
            For list-like implementations of `VectorArray` this can be
            used to prevent unnecessary copies of the involved vectors.
        '''
        pass

    @abstractmethod
    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        '''Check vecotrs for equality.

        Equality of two vectors is defined as in `pymor.tools.float_cmp_all`.

        The dimensions of `self` and `other` have to agree. If the length
        of `self` (`ind`) resp. `other` (`o_ind`) is 1, the one specified
        vector is compared to all vectors of the other summand.

        Parameters
        ----------
        other
            A `VectorArray` containing the vectors to compare with.
        ind
            If None, the whole array is compared. Otherwise an iterable
            of the indices of the vectors that are to be compared.
        o_ind
            An iterable of the indices of the vectors that are to be
            taken from `other`. If None, the whole array is selected.
        rtol
            See `pymor.tools.float_cmp_all`
        atol
            See `pymor.tools.float_cmp_all`

        Returns
        -------
        Numpy array of the truth values of the comparinson.
        '''
        pass

    @abstractmethod
    def add_mult(self, other, factor=1., o_factor=1., ind=None, o_ind=None):
        '''Linear combination of two `VectorArray` instances.

        This method forms the sum ::

            self[ind] * factor + other[ind] * factor

        The dimensions of `self` and `other` have to agree. If the length
        of `self` (`ind`) resp. `other` (`o_ind`) is 1, the one specified
        vector is added to all vectors of the other summand.

        Parameters
        ----------
        other
            A `VectorArray` containing the second summands.
        factor
            The factor with which the vectors in `self` are multiplied
        o_factor
            The factor with which the vectors in `other` are multiplied
        ind
            If None, the whole array is added. Otherwise an iterable
            of the indices of the vectors to be added.
        o_ind
            If None, the whole `other` array is added. Otherwise an iterable
            of the indices of the vectors to be added.

        Returns
        -------
        A `VectorArray` of the linear combinations.
        '''
        pass

    @abstractmethod
    def iadd_mult(self, other, factor=1., o_factor=1., ind=None, o_ind=None):
        '''In-place version of `add_mult`.'''
        pass

    @abstractmethod
    def prod(self, other, ind=None, o_ind=None, pairwise=True):
        '''Returns the scalar products between `VectorArray` elements.

        Parameters
        ----------
        other
            A `VectorArray` containing the second factors.
        ind
            If None, all vectors in `self` are taken as factors.
            Otherwise an iterable of the indices of the vectors
            in `self` whose scalar products are to be taken.
        o_ind
            If None, all vectors in `other` are taken as factors.
            Otherwise an iterable of the indices in of the vectors
            in `other` whose scalar products are to be taken.
        pairwise
            See return value documentation.

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
    def lincomb(self, factors, ind=None):
        '''Returns linear combinations of the vectors contained in the array.

        Parameters
        ----------
        factors
            A numpy array of dimension 1 or 2 containing the linear
            coeffcients. `factors.shape[-1]` has to agree with
            `len(self)`.
        ind
            If None, all vectors in `self` are taken for the linear
            combination. Otherwise an iterable of the indices in of
            the vectors in `self` which are to be used.

        Returns
        -------
        A `VectorArray` `result` such that ::

            result[i] = ∑ self[j] * factors[i,j]

        in case `factors` is of dimension 2, otherwise
        `len(result) == 1` and

            result[1] = ∑ self[j] * factors[j].
        '''
        pass

    @abstractmethod
    def lp_norm(self, p, ind=None):
        '''The l^p norms of the vectors contained in the array.

        Parameters
        ----------
        p
            If `p == 0`, the sup-norm is computed, otherwise the
            usual l^p norm.
        ind
            If None, tho norms of all vectors in `self` are computed.
            Otherwise an iterable of the indices in of the vectors in
            `self` whose norms are to be computed.

        Returns
        -------
        A numpy array `result` such that `result[i]` contains the norm
        of `self[ind][i]`.
        '''
        pass

    def l2_norm(self, ind=None):
        '''Shorthand for `lp_norm(2, ind)`.'''
        return self.lp_norm(2, ind)

    def sup_norm(self, ind=None):
        '''Shorthand for `lp_norm(0, ind)`.'''
        return self.lp_norm(0, ind)

    def gramian(self, ind=None):
        '''Shorthand for `prod(self, ind=ind, o_ind=ind, pairwise=Flase)`.'''
        return self.prod(self, ind=ind, o_ind=ind, pairwise=False)

    def _compatible_shape(self, other, ind=None, o_ind=None, broadcast=True):
        if self.dim != other.dim:
            return False
        if broadcast:
            if o_ind == None and len(other) == 1:
                return True
            elif o_ind != None and len(o_ind) == 1:
                return True
        if ind is None:
            if len(self) == 1:
                return True
            if o_ind is None:
                return len(self) == len(other)
            else:
                return len(self) == len(oind)
        else:
            if len(ind) == 1:
                return True
            if o_ind is None:
                return len(ind) == len(other)
            else:
                return len(ind) == len(oind)

    def __add__(self, other):
        return self.add_mult(other)

    def __iadd__(self, other):
        return self.iadd_mult(other)

    __radd__ = __add__

    def __sub__(self, other):
        return self.add_mult(other, o_factor=-1.)

    def __mul__(self, other):
        return self.add_mult(self, factor=other, o_factor=0.)

    def __imul__(self, other):
        return self.iadd_mult(self, factor=other, o_factor=0.)

    def __neg__(self):
        return self.add_mult(None, factor=-1, o_factor=0)
