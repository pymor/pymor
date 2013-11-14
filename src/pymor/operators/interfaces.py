# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from numbers import Number

from pymor.core import ImmutableInterface, abstractmethod, abstractproperty, abstractstaticmethod
from pymor.parameters import Parametric
from pymor.tools import Named


class OperatorInterface(ImmutableInterface, Parametric, Named):
    '''Interface for parameter dependent discrete operators.

    Every discrete operator is viewed as a map ::

        L(μ): R^s -> R^r

    Attributes
    ----------
    dim_source
        The dimension s of the source space.
    dim_range
        The dimension r of the range space.
    invert_options
        `OrderedDict` of possible options for `apply_inverse()`. Each key
        is a type of inversion algorithm which can be used to invert the
        operator. `invert_options[k]` is a dict containing all options
        along with their default values which can be set for algorithm `k`.
        We always have `invert_options[k]['type'] == k` such that
        `invert_options[k]` can be passed directly to `apply_inverse()`.
    linear
        True if the operator is (known to be) linear.
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
        mu
            The parameter for which to evaluate the operator.

        Returns
        -------
        `VectorArray` of length `len(ind)` and dimension `self.dim_range`
        '''
        pass

    @abstractmethod
    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
        '''Treat the operator as a 2-form by calculating (V, A(U)).

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
        mu
            The parameter for which to evaluate the operator.
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

    @abstractmethod
    def apply_inverse(self, U, ind=None, mu=None, options=None):
        '''Apply the inverse operator.

        Parameters
        ----------
        U : VectorArray
            `VectorArray` of vectors to which the inverse operator is applied.
        ind
            If None, the inverse operator is applied to all elements of U.
            Otherwise an iterable of the indices of the vectors to
            which the operator is to be applied.
        mu
            The parameter for which to evaluate the inverse operator.
        options
            Dictionary of options for the inversion algorithm. The
            dictionary has to contain the key `type` whose value determines
            which inversion algorithm is to be used. All other key-value
            pairs represent options specific to this algorithm.
            `options` can also be given as a string, which is then
            interpreted as the type of inversion algorithm.
            If `options` is `None`, a default algorithm with default
            options is chosen.
            Available algorithms and their default options are provided
            by the `invert_options` attribute.

        Returns
        -------
        `VectorArray` of length `len(ind)` and dimension `self.dim_source`

        Raises
        ------
        InversionError
            Is raised, if the operator cannot be inverted.
        '''
        pass

    @abstractmethod
    def as_vector(self, mu=None):
        '''Return vector representation of linear functional or vector operator.

        This method may only be called on linear operators with
        `dim_range == 1` and `type_source == NumpyVectorArray`
        (functionals) or `dim_source == 1` and `type_source ==NumpyVectorArray`
        (vector like operators).

        In the case of a functional, the identity
            operator.as_vector(mu).dot(U) == operator.apply(U, mu)
        holds. In the case of a vector like operator we have
            operator.as_vector(mu) == operator.apply(NumpyVectorArray(1), mu).
        '''
        pass

    @abstractstaticmethod
    def lincomb(operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        '''Return a linear combination of the given operators.

        Parameters
        ----------
        operators
            List of operators whose linear combination is formed.
        coefficients
            List of coefficients of the linear combination or `None`.
            Coefficients can be either `Number`s or `ParameterFunctional`s.
            If coefficients is `None` a new parameter with shape
            `(num_coefficients,)` is introduced, whose components will
            be taken as the first linear coefficients. The missing coefficients
            are set to 1.
        num_coefficients
            In case of `coefficients == None`, the number of linear coefficients
            which should be treated as a parameter. If `None`, `num_coefficients`
            is set to `len(operators)`.
        coefficients_name
            If coefficients is None, the name of the new parameter which
            holds the linear coefficients.
        name
            Name of the operator.

        Returns
        -------
        Operator representing the linear combination.
        '''
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


class LincombOperatorInterface(OperatorInterface):
    '''An operator representing a linear combination.

    Attributes
    ----------
    operators
        List of operators whose linear combination is formed.
    coefficients
        `None` or list of linear coefficients.
    num_coefficients
        If `coefficients` is `None`, the number of linear
        coefficients which are given by the parameter with
        name `coefficients_name`. The missing coefficients
        are set to 1.
    coefficients_name
        If `coefficients` is `None`, the name of the parameter
        providing the linear coefficients.
    '''

    operators = None
    coefficients = None
    num_coefficients = None
    coefficients_name = None

    @abstractmethod
    def evaluate_coefficients(self, mu):
        '''Evaluate the linear coefficients for a given parameter.'''
        pass
