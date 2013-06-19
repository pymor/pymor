# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from numbers import Number

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractproperty
from pymor.la import VectorArrayInterface
from pymor.tools import Named
from pymor.parameters import Parametric


class OperatorInterface(BasicInterface, Parametric, Named):
    '''Interface for parameter dependent discrete operators.

    Every discrete operator is viewed as a map ::

        L(μ): R^s -> R^r

    Attributes
    ----------
    dim_source
        The dimension s of the source space.
    dim_range
        The dimension r of the range space.
    type_source
        The `VectorArray` class representing vectors of the source space.
    type_range
        The `VectorArray` class representing vectors of the range space.
    '''

    dim_source = 0
    dim_range = 0

    type_source = None
    type_range = None

    def __init__(self):
        Parametric.__init__(self)

    @abstractmethod
    def apply(self, U, ind=None, mu=None):
        '''Evaluate L(U, mu).

        Parameters
        ----------
        U : VectorArray
            `VectorArray` of vectors to which the operator is applied.
        ind
            If None, the operator is applied to all elements of U.
            Otherwise an iterable of the indices of the vectors to
            which the operator is to be applied.

        Returns
        -------
        `VectorArray` of shape `(len(ind), self.dim_range)`
        '''
        pass

    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
        ''' Treat the operator as a 2-form by calculating (V, A(U)).

        If ( , ) is the euclidean scalar product and A is given by
        multiplication with a matrix B, then ::

            L.apply2(V, U) = V^T*B*U

        Parameters
        ----------
        V : VectorArray
            Left arguments.
        U : VectorArray
            Right arguments.
        V_ind
            If None, the operator is applied to all elements of V.
            Otherwise an iterable of the indices of the vectors to
            which the operator is to be applied.
        U_ind
            If None, the operator is applied to all elements of U.
            Otherwise an iterable of the indices of the vectors to
            which the operator is to be applied.
        product
            `Operator` representing the scalar product.
            If None, the euclidean product is chosen.
        pairwise
            If False and V and U are multi-dimensional, then form is applied
            to all combinations of vectors in V and U, i.e. ::

                L.apply2(V, U).shape = (len(V_ind), len(U_ind)).

            If True, the vectors in V and U are applied in pairs, i.e. ::

                L.apply2(V, U).shape = (len(V_ind),) = (len(U_ind),).

        Returns
        -------
        A numpy array of all operator evaluations.
        '''
        assert isinstance(V, VectorArrayInterface)
        assert isinstance(U, VectorArrayInterface)
        if pairwise:
            lu = len(U_ind) if U_ind is not None else len(U)
            lv = len(V_ind) if V_ind is not None else len(V)
            assert lu == lv
        AU = self.apply(U, ind=U_ind, mu=mu)
        if product is not None:
            AU = product.apply(AU)
        return V.prod(AU, ind=V_ind, pairwise=pairwise)

    def __add__(self, other):
        if isinstance(other, Number):
            assert other == 0.
            return self
        from pymor.operators.constructions import LincombOperator
        return LincombOperator([self, other])

    __radd__ = __add__

    def __mul__(self, other):
        from pymor.operators.constructions import LincombOperator
        return LincombOperator([self], factors=[other])

    def __str__(self):
        return '{}: R^{} --> R^{}  (parameter type: {}, class: {})'.format(
            self.name, self.dim_source, self.dim_range, self.parameter_type,
            self.__class__.__name__)


class LinearOperatorInterface(OperatorInterface):
    '''Interface for linear parameter dependent discrete operators.
    '''

    assembled = False
    sparse = None

    def as_vector_array(self, mu=None):
        if not self.assembled:
            return self.assemble(mu).as_vector_array()
        else:
            raise NotImplementedError

    @abstractmethod
    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None, force=False):
        '''Assembles the matrix of the operator for given parameter mu.

        Returns an assembled parameter independent linear operator.
        '''
        if self.assembled:
            assert mu is None
            return self
        elif self.parameter_type is None:
            assert mu is None
            if force or not self._last_mat:
                self._last_mat = self._assemble(mu)
                return self._last_mat
            else:
                return self._last_mat
        else:
            mu = self.parse_parameter(mu)
            if not force and self._last_mu is not None and self._last_mu.allclose(mu):
                return self._last_mat
            else:
                self._last_mu = mu.copy()  # TODO: add some kind of log message here
                self._last_mat = self._assemble(mu)
                return self._last_mat

    def apply(self, U, ind=None, mu=None):
        if not self.assembled:
            return self.assemble(mu).apply(U, ind=ind)
        else:
            raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Number):
            assert other == 0.
            return self
        from pymor.operators.constructions import LinearLincombOperator
        return LinearLincombOperator([self, other])

    __radd__ = __add__

    def __mul__(self, other):
        from pymor.operators.constructions import LinearLincombOperator
        return LinearLincombOperator([self], factors=[other])

    _last_mu = None
    _last_mat = None
