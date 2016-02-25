# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import ImmutableInterface, abstractmethod
from pymor.parameters.base import Parametric


class OperatorInterface(ImmutableInterface, Parametric):
    """Interface for |Parameter| dependent discrete operators.

    An operator in pyMOR is simply a mapping which for any given
    |Parameter| maps vectors from its source |VectorSpace|
    to vectors in its range |VectorSpace|.

    Note that there is no special distinction between functionals
    and operators in pyMOR. A functional is simply an operator with
    |NumpyVectorSpace| `(1)` as its range |VectorSpace|.

    Attributes
    ----------
    solver_options
        If not `None`, a dict which can contain the follwing keys:

        :'inverse':           solver options used for
                              :meth:`~OperatorInterface.apply_inverse`
        :'inverse_adjoint':   solver options used for
                              :meth:`~OperatorInterface.apply_inverse_adjoint`
        :'jacobian':          solver options for the operators returned
                              by :meth:`~OperatorInterface.jacobian`
                              (has no effect for linear operators)

        If `solver_options` is `None` or a dict entry is missing
        or `None`, default options are used.
        The interpretation of the given solver options is up to
        the operator at hand. In general, a values in `solver_options`
        should either be strings (indicating a solver type) or
        dicts of options, usually with an entry `'type'` which
        specifies the solver type to use and further items which
        configure this solver.
    linear
        `True` if the operator is linear.
    source
        The source |VectorSpace|.
    range
        The range |VectorSpace|.
    """

    solver_options = None

    @abstractmethod
    def apply(self, U, ind=None, mu=None):
        """Apply the operator.

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
        """
        pass

    @abstractmethod
    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None):
        """Treat the operator as a 2-form by calculating (V, A(U)).

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

        Returns
        -------
        A |NumPy array| with shape `(len(V_ind), len(U_ind))` containing the 2-form
        evaluations.
        """
        pass

    @abstractmethod
    def pairwise_apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None):
        """Treat the operator as a 2-form by calculating (V, A(U)).

        Same as :meth:`OperatorInterface.apply2`, except that vectors from `V`
        and `U` are applied in pairs.

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

        Returns
        -------
        A |NumPy array| with shape `(len(V_ind),) == (len(U_ind),)` containing
        the 2-form evaluations.
        """
        pass

    @abstractmethod
    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        """Apply the adjoint operator.

        For a linear operator A the adjoint A^* of A is given by ::

            (A^*v, u)_s = (v, Au)_r

        where ( , )_s and ( , )_r denote the scalar products on the source
        and range space of A. If A and the two products are given by the
        matrices M, P_s and P_r, then::

            A^*v = P_s^(-1) * M^T * P_r * v

        with M^T denoting the transposed of M. Thus, if ( , )_s and ( , )_r
        are the euclidean products, A^*v is simply given by multiplication of
        the matrix of A with v from the left.

        Parameters
        ----------
        U
            |VectorArray| of vectors to which the adjoint operator is applied.
        ind
            The indices of the vectors in `U` to which the operator shall be
            applied. (See the |VectorArray| documentation for further details.)
        mu
            The |Parameter| for which to apply the adjoint operator.
        source_product
            The scalar product |Operator| on the source space.
            If `None`, the euclidean product is chosen.
        range_product
            The scalar product |Operator| on the range space.
            If `None`, the euclidean product is chosen.

        Returns
        -------
        |VectorArray| of the adjoint operator evaluations.
        """
        pass

    @abstractmethod
    def apply_inverse(self, V, ind=None, mu=None, least_squares=False):
        """Apply the inverse operator.

        Parameters
        ----------
        V
            |VectorArray| of vectors to which the inverse operator is applied.
        ind
            The indices of the vectors in `V` to which the inverse operator shall be
            applied. (See the |VectorArray| documentation for further details.)
        mu
            The |Parameter| for which to evaluate the inverse operator.
        least_squares
            If `True`, solve the least squares problem::

                u = argmin ||Au - v||_2.

            Since for an invertible operator the least squares solution agrees
            with the result of the application of the inverse operator,
            setting this option should, in general, have no effect on the result
            for those operators. However, note that when appropriate
            |solver_options| are not set for the operator, most operator
            implementations will choose a least squares solver by default which
            may not be desirable for invertible operators.

        Returns
        -------
        |VectorArray| of the inverse operator evaluations.

        Raises
        ------
        InversionError
            The operator could not be inverted.
        """
        pass

    @abstractmethod
    def apply_inverse_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None,
                              least_squares=False):
        """Apply the inverse adjoint operator.

        Parameters
        ----------
        U
            |VectorArray| of vectors to which the inverse adjoint operator is applied.
        ind
            The indices of the vectors in `U` to which the inverse adjoint operator shall be
            applied. (See the |VectorArray| documentation for further details.)
        mu
            The |Parameter| for which to evaluate the inverse adjoint operator.
        source_product
            See :meth:`~OperatorInterface.apply_adjoint`.
        range_product
            See :meth:`~OperatorInterface.apply_adjoint`.
        least_squares
            If `True`, solve the least squares problem::

                v = argmin ||A*v - u||_2.

            Since for an invertible operator the least squares solution agrees
            with the result of the application of the inverse operator,
            setting this option should, in general, have no effect on the result
            for those operators. However, note that when appropriate
            |solver_options| are not set for the operator, most operator
            implementations will choose a least squares solver by default which
            may not be desirable for invertible operators.

        Returns
        -------
        |VectorArray| of the inverse adjoint operator evaluations.

        Raises
        ------
        InversionError
            The operator could not be inverted.
        """
        pass

    @abstractmethod
    def jacobian(self, U, mu=None):
        """Return the operator's Jacobian.

        Parameters
        ----------
        U
            Length 1 |VectorArray| containing the vector for which to compute
            the Jacobian.
        mu
            The |Parameter| for which to compute the Jacobian.

        Returns
        -------
        |Operator| representing the Jacobian.
        """
        pass

    @abstractmethod
    def as_vector(self, mu=None):
        """Return a vector representation of a linear functional or vector operator.

        This method may only be called on linear functionals, i.e. linear |Operators|
        with |NumpyVectorSpace| `(1)` as :attr:`~OperatorInterface.range`,
        or on operators describing vectors, i.e. linear |Operators| with
        |NumpyVectorSpace| `(1)` as :attr:`~OperatorInterface.source`.

        In the case of a functional, the identity ::

            self.as_vector(mu).dot(U) == self.apply(U, mu)

        holds, whereas in the case of a vector-like operator we have ::

            self.as_vector(mu) == self.apply(NumpyVectorArray(1), mu).

        Parameters
        ----------
        mu
            The |Parameter| for which to return the vector representation.

        Returns
        -------
        V
            |VectorArray| of length 1 containing the vector representation.
            `V` belongs to `self.source` for functionals and to `self.range` for
            vector-like operators.
        """
        pass

    @abstractmethod
    def assemble(self, mu=None):
        """Assemble the operator for a given parameter.

        What the result of the assembly is strongly depends on the given operator.
        For instance, a matrix-based operator will assemble its matrix, a |LincombOperator|
        will try to form the linear combination of its operators, whereas an arbitrary
        operator might simply return a :class:`~pymor.operators.constructions.FixedParameterOperator`.
        The only assured property of the assembled operator is that it no longer
        depends on a |Parameter|.

        Parameters
        ----------
        mu
            The |Parameter| for which to assemble the operator.

        Returns
        -------
        Parameter-independent, assembled |Operator|.
        """
        pass

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
        """Try to assemble a linear combination of the given operators.

        This method is called in the `assemble` method of |LincombOperator| on
        the first of its operator. If an assembly of the given linear combination
        is possible, e.g. the linear combination of the system matrices of the
        operators can be formed, then the assembled operator is returned.
        Otherwise, the method returns `None` to indicate that assembly is not possible.

        Parameters
        ----------
        operators
            List of |Operators| whose linear combination is formed.
        coefficients
            List of the corresponding linear coefficients.
        solver_options
            |solver_options| for the assembled operator.
        name
            Name of the assembled operator.

        Returns
        -------
        The assembled |Operator| if assembly is possible, otherwise `None`.
        """
        return None

    @abstractmethod
    def projected(self, range_basis, source_basis, product=None, name=None):
        """Project the operator to subspaces of the source and range space.

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
        again be a |NumpyMatrixOperator|, only a generic
        :class:`pymor.operators.basic.ProjectedOperator` will be returned
        in general. (Note that the latter will not be suitable to obtain an
        efficient offline/online-decomposition for reduced basis schemes.)

        A default implementation is provided in |OperatorBase|.

        .. warning::
            No check is performed whether the b_i and c_j are orthonormal or linear
            independent.

        Parameters
        ----------
        range_basis
            The c_1, ..., c_M as a |VectorArray|. If `None`, no projection in the range
            space is performed.
        source_basis
            The b_1, ..., b_N as a |VectorArray| or `None`. If `None`, no restriction of
            the source space is performed.
        product
            An |Operator| representing the scalar product.  If `None`, the
            Euclidean product is chosen.
        name
            Name of the projected operator.

        Returns
        -------
        The projected |Operator|.
        """
        pass

    def restricted(self, dofs):
        """Restrict the operator range to a given set of degrees of freedom.

        This method returns a restricted version `restricted_op` of the
        operator along with an array `source_dofs` such that for any
        |VectorArray| `U` in `self.source` the following is true::

            self.apply(U, mu).components(dofs)
                == restricted_op.apply(NumpyVectorArray(U.components(source_dofs)), mu))

        Such an operator is mainly useful for
        :class:`empirical interpolation <pymor.operators.ei.EmpiricalInterpolatedOperator>`
        where the evaluation of the original operator only needs to be known
        for few selected degrees of freedom. If the operator has a small
        stencil, only few `source_dofs` will be needed to evaluate the
        restricted operator which can make its evaluation very fast
        compared to evaluating the original operator. Note that the interface
        does not make any assumptions on the efficiency of evaluating the
        restricted operator.

        Parameters
        ----------
        dofs
            One-dimensional |NumPy array| of degrees of freedom in the operator
            :attr:`~OperatorInterface.range` to which to restrict.

        Returns
        -------
        restricted_op
            The restricted operator as defined above. The operator will have
            |NumpyVectorSpace| `(len(source_dofs))` as :attr:`~OperatorInterface.source`
            and |NumpyVectorSpace| `(len(dofs))` as :attr:`~OperatorInterface.range`.
        source_dofs
            One-dimensional |NumPy array| of source degrees of freedom as
            defined above.
        """
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other):
        """Sum of two operators"""
        pass

    @abstractmethod
    def __radd__(self, other):
        """Sum of two operators"""
        pass

    @abstractmethod
    def __mul__(self, other):
        """Product of operator by a scalar"""
        pass
