# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from pymor.tools import Named
from pymor.parameters import Parametric

class DiscreteOperatorInterface(core.BasicInterface, Parametric, Named):
    '''Interface for parameter dependent discrete operators.

    Every discrete operator is viewed as a map ::

        L(μ): R^s -> R^r

    Attributes
    ----------
    dim_source
        The dimension s of the source space.
    dim_range
        The dimension r of the range space.

    Inherits
    --------
    BasicInterface, Parametric, Cachable, Named
    '''

    dim_source = 0
    dim_range = 0

    @core.interfaces.abstractmethod
    def apply(self, U, mu={}):
        '''Evaluate L(U, mu).

        If U is an nd-array, L is applied to the last axis.
        '''
        pass

    def apply2(self, V, U, mu={}, product=None, pairwise=True):
        ''' Treat the operator as a 2-form by calculating (V, A(U)).

        If ( , ) is the euclidean scalar product and A is given by
        multiplication with a matrix B, then

            L.apply2(V, U) = V^T*B*U

        Parameters
        ----------
        product
            `DiscreteOperator` or 2d-array representing the scalar product.
            If None, the euclidean product is chosen.
        pairwise
            If False and V and U are multi-dimensional, then form is applied
            to all combinations of vectors in V and U, i.e.
                L.apply2(V, U).shape = V.shape[:-1] + U.shape[:-1].
            If True, the vectors in V and U are applied in pairs, i.e.
                L.apply2(V, U).shape = V.shape[:-1] ( == U.shape[:-1]).
        '''
        AU = self.apply(U, mu)
        assert not pairwise or AU.shape[:-1] == V.shape[:-1]
        if product is None:
            if pairwise:
                return np.sum(V * AU, axis=-1)
            else:
                return np.dot(V, AU.T)                              # use of np.dot for ndarrays?!
        elif isinstance(product, DiscreteOperatorInterface):
            return product.apply2(V, AU, pairwise)
        else:
            if pairwise:
                return np.sum(V * np.dot(product, AU.T).T, axis=-1)
            else:
                return np.dot(V, np.dot(product, AU.T))

    def __str__(self):
        return '{}: R^{} --> R^{}  (parameter type: {}, class: {})'.format(
                                           self.name, self.dim_source, self.dim_range, self.parameter_type,
                                           self.__class__.__name__)


class LinearDiscreteOperatorInterface(DiscreteOperatorInterface):
    '''Interface for linear parameter dependent discrete operators.

    Inherits
    --------
    DiscreteOperatorInterface
    '''

    @core.interfaces.abstractmethod
    def assemble(self, mu={}):
        '''Assembles the matrix of the operator for given parameter mu.

        This is what has to be implemented, but the user will usually call
        matrix().
        '''
        pass

    def matrix(self, mu={}):
        '''Same as assemble(), but the last result is cached.'''
        mu = self.parse_parameter(mu)
        if self._last_mu is not None and self._last_mu.allclose(mu):
            return self._last_mat
        else:
            self._last_mu = mu.copy() # TODO: add some kind of log message here
            self._last_mat = self.assemble(mu)
            return self._last_mat

    def apply(self, U, mu={}):
        return self.matrix(mu).dot(U.T).T  # TODO: Check what dot really does for a sparse matrix

    _last_mu = None
    _last_mat = None
