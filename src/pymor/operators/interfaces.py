# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from numbers import Number

from pymor.core.interfaces import BasicInterface, abstractmethod, abstractproperty, abstractstaticmethod
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

    linear = False

    invert_options = None

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

    @abstractmethod
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
        pass

    @abstractstaticmethod
    def lincomb(operators, coefficients=None, global_names=None, name=None):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __radd__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def apply_inverse(self, U, ind=None, mu=None, options=None):
        pass


class MatrixBasedOperatorInterface(OperatorInterface):
    '''Base class for operators which assemble to a matrix.
    '''

    linear = True
    assembled = False
    sparse = None

    @abstractmethod
    def as_vector_array(self, mu=None):
        pass

    @abstractmethod
    def assemble(self, mu=None, force=False):
        '''Assembles the matrix of the operator for given parameter mu.

        Returns an assembled parameter independent MatrixOperator operator.
        '''
        pass


class LincombOperatorInterface(OperatorInterface):

    operators = None
    coefficients = None

    @abstractmethod
    def evaluate_coefficients(self, mu):
        pass
