# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core import ImmutableInterface, abstractmethod, abstractstaticmethod
from pymor.parameters import Parametric
from pymor.tools import Named


class OperatorInterface(ImmutableInterface, Parametric, Named):
    '''Interface for |Parameter| dependent discrete operators.

    Every operator is viewed as a map ::

        A(μ): R^s -> R^r

    Note that there is no special distinction between functionals
    and operators in pyMOR. A functional is simply an operator with
    range dimension 1 and |NumpyVectorArray| as `type_range`.

    Attributes
    ----------
    dim_source
        The dimension s of the source space.
    dim_range
        The dimension r of the range space.
    invert_options
        |OrderedDict| of possible options for :meth:`~OperatorInterface.apply_inverse`.
        Each key is a type of inversion algorithm which can be used to invert the
        operator. `invert_options[k]` is a dict containing all options along with
        their default values which can be set for algorithm `k`. We always have
        `invert_options[k]['type'] == k` such that `invert_options[k]` can be passed
        directly to :meth:`~OperatorInterface.apply_inverse()`.
    linear
        `True` if the operator is linear.
    type_source
        The |VectorArray| class representing vectors of the source space.
    type_range
        The |VectorArray| class representing vectors of the range space.
    '''

    @abstractmethod
    def apply(self, U, ind=None, mu=None):
        '''Apply the operator.

        Parameters
        ----------
        U
            |VectorArray| of vectors to which the operator is applied.
        ind
            The indices of the vectors in `U` to which the operator shall be
            applied. (See the |VectorArray| documentation for further details.)
        mu
            The |Parameter| for which to evaluate the operator.

        Returns
        -------
        |VectorArray| of the operator evaluations.
        '''
        pass

    @abstractmethod
    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
        '''Treat the operator as a 2-form by calculating (V, A(U)).

        In particular, if ( , ) is the Euclidean product and A is a linear operator
        given by multiplication with a matrix M, then ::

            A.apply2(V, U) = V^T*M*U

        Parameters
        ----------
        V
            |VectorArray| of the left arguments V.
        U
            |VectorArray| of the right right arguments U.
        V_ind
            The indices of the vectors in `V` to which the operator shall be
            applied. (See the |VectorArray| documentation for further details.)
        U_ind
            The indices of the vectors in `U` to which the operator shall be
            applied. (See the |VectorArray| documentation for further details.)
        mu
            The |Parameter| for which to evaluate the operator.
        product
            The scalar product used in the expression `(V, A(U))` given as
            an |Operator|.  If `None`, the euclidean product is chosen.
        pairwise
            If `False`, the 2-form is applied to all combinations of vectors
            in `V` and `U`, i.e. ::

                L.apply2(V, U).shape = (len(V_ind), len(U_ind)).

            If `True`, the vectors in `V` and `U` are applied in pairs, i.e.
            `V` and `U` must be of the same length and we have ::

                L.apply2(V, U).shape = (len(V_ind),) = (len(U_ind),).

        Returns
        -------
        A |NumPy array| of all 2-form evaluations.
        '''
        pass

    @abstractmethod
    def apply_inverse(self, U, ind=None, mu=None, options=None):
        '''Apply the inverse operator.

        Parameters
        ----------
        U
            |VectorArray| of vectors to which the inverse operator is applied.
        ind
            The indices of the vectors in `U` to which the operator shall be
            applied. (See the |VectorArray| documentation for further details.)
        mu
            The |Parameter| for which to evaluate the inverse operator.
        options
            Dictionary of options for the inversion algorithm. The dictionary
            has to contain the key `'type'` whose value determines which inversion
            algorithm is to be used. All other items represent options specific to
            this algorithm.  `options` can also be given as a string, which is then
            interpreted as the type of inversion algorithm. If `options` is `None`,
            a default algorithm with default options is chosen.  Available algorithms
            and their default options are provided by
            :attr:`~OperatorInterface.invert_options`.

        Returns
        -------
        |VectorArray| of the inverse operator evaluations.

        Raises
        ------
        InversionError
            The operator could not be inverted.
        '''
        pass

    @abstractmethod
    def as_vector(self, mu=None):
        '''Return vector representation of linear functional or vector operator.

        This method may only be called on linear functionals, i.e. linear operators
        with `dim_range == 1` and |NumpyVectorArray| as :attr:`~OperatorInterface.type_range`,
        or on operators describing vectors, i.e. linear operators with
        `dim_source == 1` |NumpyVectorArray| as :attr:`~OperatorInterface.type_source`.

        In the case of a functional, the identity ::

            self.as_vector(mu).dot(U) == operator.apply(U, mu)

        holds, whereas in the case of a vector like operator we have ::

            operator.as_vector(mu) == operator.apply(NumpyVectorArray(1), mu).

        Parameters
        ----------
        mu
            The |Parameter| for which to return a vector representation.

        Returns
        -------
        V
            |VectorArray| of length 1 containing the vector representation. We have
            `V.dim == self.dim_source`, `type(V) == self.type_source` for functionals
            and `V.dim = self.dim_range`, `type(V) == self.dim_range` for vector-like
            operators.
        '''
        pass

    @abstractstaticmethod
    def lincomb(operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        '''Form a linear combination of the given operators.

        How this linear combiniation is realized will depend on the operators involved.
        E.g. calling `lincomb` on a |NumpyMatrixBasedOperator| and only providing
        such operators will result in a new |NumpyMatrixBasedOperator| that will assemble
        to a |NumpyMatrixOperator|, whereas for arbitrary operators,
        :class:`pymor.operators.basic.LincombOperator` will be returned.

        The linear coefficients may be provided as scalars or |ParameterFunctionals|.
        Alternatively, if no linear coefficients are given, the missing coefficients become
        part of the |Parameter| the combinded |Operator| expects.

        A default implementation is provided in |OperatorBase|.

        Parameters
        ----------
        operators
            List of |Operators| whose linear combination is formed.
        coefficients
            `None` or a list of linear coefficients.
        num_coefficients
            If `coefficients` is `None`, the number of linear coefficients (starting
            at index 0) which are given by the |Parameter| component with name
            `'coefficients_name'`. The missing coefficients are set to `1`.
        coefficients_name
            If `coefficients` is `None`, the name of the |Parameter| component providing
            the linear coefficients.
        name
            Name of the new operator.

        Returns
        -------
        |LincombOperator| representing the linear combination.
        '''
        pass

    @abstractmethod
    def projected(self, source_basis, range_basis, product=None, name=None):
        '''Project operator to subspaces of the source and range space.

        Denote `self` by A. Given a scalar product ( ⋅, ⋅), and vectors b_1, ..., b_N,
        c_1, ..., c_M, the projected operator A_P is defined by ::

            [ A_P(e_j) ]_i = ( c_i, A(b_j) )

        for all i,j, where e_j denotes the j-th canonical basis vector of R^N.

        In particular, if the c_i are orthonormal w.r.t. the given product,
        then A_P is the coordinate representation w.r.t. the b_i/c_i bases
        of the restriction of A to span(b_i) concatenated with the orthogonal
        projection onto span(c_i).

        From another point of view, if A is viewed as a bilinear form
        (see :meth:`~OperatorInterface.apply2`) and ( ⋅, ⋅ ) is the Euclidean
        product, then A_P represents the matrix of the bilinear form restricted
        span(b_i) / spanc(c_i) (w.r.t. the b_i/c_i bases).

        How the projected operator is realized will depend on the implementation
        of the operator to project.  While a projected |NumpyMatrixOperator| will
        again be a |NumpyMatrixOperator|, only a
        :class:`pymor.operators.basic.ProjectedOperator` will be returned
        in general. (Note that the latter will not be suitable to obtain an
        efficient offline/online-decomposition for reduced basis schemes.)

        A default implementation is provided in |OperatorBase|.

        .. warning::
            No check is performed whether the b_i and c_j are orthonormal or linear
            independent.

        Parameters
        ----------
        source_basis
            The b_1, ..., b_N as a |VectorArray| or `None`. If `None`, no restriction of
            the source space is performed.
        range_basis
            The c_1, ..., c_M as a |VectorArray|. If `None`, no projection in the range
            space is performed.
        product
            An |Operator| representing the scalar product.  If `None`, the
            Euclidean product is chosen.
        name
            Name of the projected operator.

        Returns
        -------
        The projected |Operator|.
        '''
        pass

    @abstractmethod
    def __add__(self, other):
        '''Sum of two operators'''
        pass

    @abstractmethod
    def __radd__(self, other):
        '''Sum of two operators'''
        pass

    @abstractmethod
    def __mul__(self, other):
        '''Product of operator by a scalar'''
        pass


class LincombOperatorInterface(OperatorInterface):
    '''|Operator| representing a linear combination.

    The linear coefficients can be scalars or |ParameterFunctionals|.  Alternatively,
    if no linear coefficients are given, the missing coefficients become
    part of the |Parameter| the combined |Operator| expects.

    Attributes
    ----------
    operators
        List of |Operators| whose linear combination is formed.
    coefficients
        `None` or a list of linear coefficients.
    num_coefficients
        If `coefficients` is `None`, the number of linear coefficients (starting
        at index 0) which are given by the |Parameter| component with name
        `'coefficients_name'`. The missing coefficients are set to `1`.
    coefficients_name
        If `coefficients` is `None`, the name of the |Parameter| component providing
        the linear coefficients.
    '''

    @abstractmethod
    def evaluate_coefficients(self, mu):
        '''Evaluate the linear coefficients for a given |Parameter|.'''
        pass
