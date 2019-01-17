# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import ImmutableInterface, abstractmethod
from pymor.parameters.base import Parametric
from pymor.vectorarrays.numpy import NumpyVectorSpace


class OperatorInterface(ImmutableInterface, Parametric):
    """Interface for |Parameter| dependent discrete operators.

    An operator in pyMOR is simply a mapping which for any given
    |Parameter| maps vectors from its `source` |VectorSpace|
    to vectors in its `range` |VectorSpace|.

    Note that there is no special distinction between functionals
    and operators in pyMOR. A functional is simply an operator with
    |NumpyVectorSpace| `(1)` as its `range` |VectorSpace|.

    Attributes
    ----------
    solver_options
        If not `None`, a dict which can contain the following keys:

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
        the operator at hand. In general, values in `solver_options`
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
    H
        The adjoint operator, i.e. ::

            self.H.apply(V, mu) == self.apply_adjoint(V, mu)

        for all V, mu.
    """

    solver_options = None

    @property
    def H(self):
        from pymor.operators.constructions import AdjointOperator
        return AdjointOperator(self)

    @abstractmethod
    def apply(self, U, mu=None):
        """Apply the operator to a |VectorArray|.

        Parameters
        ----------
        U
            |VectorArray| of vectors to which the operator is applied.
        mu
            The |Parameter| for which to evaluate the operator.

        Returns
        -------
        |VectorArray| of the operator evaluations.
        """
        pass

    @abstractmethod
    def apply2(self, V, U, mu=None):
        """Treat the operator as a 2-form by computing ``V.dot(self.apply(U))``.

        If the operator is a linear operator given by multiplication with a matrix
        M, then `apply2` is given as::

            op.apply2(V, U) = V^T*M*U.

        In the case of complex numbers, note that `apply2` is anti-linear in the
        first variable by definition of `dot`.

        Parameters
        ----------
        V
            |VectorArray| of the left arguments V.
        U
            |VectorArray| of the right right arguments U.
        mu
            The |Parameter| for which to evaluate the operator.

        Returns
        -------
        A |NumPy array| with shape `(len(V), len(U))` containing the 2-form
        evaluations.
        """
        pass

    @abstractmethod
    def pairwise_apply2(self, V, U, mu=None):
        """Treat the operator as a 2-form by computing ``V.dot(self.apply(U))``.

        Same as :meth:`OperatorInterface.apply2`, except that vectors from `V`
        and `U` are applied in pairs.

        Parameters
        ----------
        V
            |VectorArray| of the left arguments V.
        U
            |VectorArray| of the right right arguments U.
        mu
            The |Parameter| for which to evaluate the operator.

        Returns
        -------
        A |NumPy array| with shape `(len(V),) == (len(U),)` containing
        the 2-form evaluations.
        """
        pass

    @abstractmethod
    def apply_adjoint(self, V, mu=None):
        """Apply the adjoint operator.

        For any given linear |Operator| `op`, |Parameter| `mu` and
        |VectorArrays| `U`, `V` in the :attr:`~OperatorInterface.source`
        resp. :attr:`~OperatorInterface.range` we have::

            op.apply_adjoint(V, mu).dot(U) == V.dot(op.apply(U, mu))

        Thus, when `op` is represented by a matrix `M`, `apply_adjoint` is
        given by left-multplication of (the complex conjugate of) `M` with `V`.

        Parameters
        ----------
        V
            |VectorArray| of vectors to which the adjoint operator is applied.
        mu
            The |Parameter| for which to apply the adjoint operator.

        Returns
        -------
        |VectorArray| of the adjoint operator evaluations.
        """
        pass

    @abstractmethod
    def apply_inverse(self, V, mu=None, least_squares=False):
        """Apply the inverse operator.

        Parameters
        ----------
        V
            |VectorArray| of vectors to which the inverse operator is applied.
        mu
            The |Parameter| for which to evaluate the inverse operator.
        least_squares
            If `True`, solve the least squares problem::

                u = argmin ||op(u) - v||_2.

            Since for an invertible operator the least squares solution agrees
            with the result of the application of the inverse operator,
            setting this option should, in general, have no effect on the result
            for those operators. However, note that when no appropriate
            |solver_options| are set for the operator, most implementations
            will choose a least squares solver by default which may be
            undesirable.

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
    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        """Apply the inverse adjoint operator.

        Parameters
        ----------
        U
            |VectorArray| of vectors to which the inverse adjoint operator is applied.
        mu
            The |Parameter| for which to evaluate the inverse adjoint operator.
        least_squares
            If `True`, solve the least squares problem::

                v = argmin ||op^*(v) - u||_2.

            Since for an invertible operator the least squares solution agrees
            with the result of the application of the inverse operator,
            setting this option should, in general, have no effect on the result
            for those operators. However, note that when no appropriate
            |solver_options| are set for the operator, most operator
            implementations will choose a least squares solver by default which
            may be undesirable.

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
        """Return the operator's Jacobian as a new |Operator|.

        Parameters
        ----------
        U
            Length 1 |VectorArray| containing the vector for which to compute
            the Jacobian.
        mu
            The |Parameter| for which to compute the Jacobian.

        Returns
        -------
        Linear |Operator| representing the Jacobian.
        """
        pass

    def as_range_array(self, mu=None):
        """Return a |VectorArray| representation of the operator in its range space.

        In the case of a linear operator with |NumpyVectorSpace| as
        :attr:`~OperatorInterface.source`, this method returns for every |Parameter|
        `mu` a |VectorArray| `V` in the operator's :attr:`~OperatorInterface.range`,
        such that ::

            V.lincomb(U.to_numpy()) == self.apply(U, mu)

        for all |VectorArrays| `U`.

        Parameters
        ----------
        mu
            The |Parameter| for which to return the |VectorArray| representation.

        Returns
        -------
        V
            The |VectorArray| defined above.
        """
        assert isinstance(self.source, NumpyVectorSpace) and self.linear
        raise NotImplementedError

    def as_source_array(self, mu=None):
        """Return a |VectorArray| representation of the operator in its source space.

        In the case of a linear operator with |NumpyVectorSpace| as
        :attr:`~OperatorInterface.range`, this method returns for every |Parameter|
        `mu` a |VectorArray| `V` in the operator's :attr:`~OperatorInterface.source`,
        such that ::

            self.range.make_array(V.dot(U).T) == self.apply(U, mu)

        for all |VectorArrays| `U`.

        Parameters
        ----------
        mu
            The |Parameter| for which to return the |VectorArray| representation.

        Returns
        -------
        V
            The |VectorArray| defined above.
        """
        assert isinstance(self.range, NumpyVectorSpace) and self.linear
        raise NotImplementedError

    def as_vector(self, mu=None, *, space=None):
        """Return a vector representation of a linear functional or vector operator.

        Depending on the operator's :attr:`~OperatorInterface.source` and
        :attr:`~OperatorInterface.range`, this method is equivalent to calling
        :meth:`~OperatorInterface.as_range_array` or :meth:`~OperatorInterface.as_source_array`
        respectively. The resulting |VectorArray| is required to have length 1.

        Note that in case both :attr:`~OperatorInterface.source` and
        :attr:`~OperatorInterface.range` are one-dimensional |NumpyVectorSpaces|
        but with different :attr:`ids <pymor.vectorarrays.interfaces.VectorSpaceInterface.id>`,
        it is impossible to determine which space to choose. In this case,
        the desired space has to be specified via the `space` parameter.

        Parameters
        ----------
        mu
            The |Parameter| for which to return the vector representation.
        space
            See above.

        Returns
        -------
        V
            |VectorArray| of length 1 containing the vector representation.
        """
        if not self.linear:
            raise TypeError('This nonlinear operator does not represent a vector or linear functional.')
        if space is not None:
            if self.range == space:
                V = self.as_range_array(mu)
                assert len(V) == 1
                return V
            elif self.source == space:
                V = self.as_source_array(mu)
                assert len(V) == 1
                return V
            else:
                raise TypeError('This operator cannot be represented by a VectorArray in the given space.')
        elif self.source.is_scalar:
            if self.range.is_scalar and self.range.id != self.source.id:
                raise TypeError("Cannot determine space of VectorArray representation (specify 'space' parameter).")
            return self.as_range_array(mu)
        elif self.range.is_scalar:
            if self.source.is_scalar and self.source.id != self.range.id:
                raise TypeError("Cannot determine space of VectorArray representation (specify 'space' parameter).")
            return self.as_source_array(mu)
        else:
            raise TypeError('This operator does not represent a vector or linear functional.')

    @abstractmethod
    def assemble(self, mu=None):
        """Assemble the operator for a given parameter.

        The result of the method strongly depends on the given operator.
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

        This method is called in the :meth:`assemble` method of |LincombOperator| on
        the first of its operators. If an assembly of the given linear combination
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

    def restricted(self, dofs):
        """Restrict the operator range to a given set of degrees of freedom.

        This method returns a restricted version `restricted_op` of the
        operator along with an array `source_dofs` such that for any
        |VectorArray| `U` in `self.source` the following is true::

            self.apply(U, mu).dofs(dofs)
                == restricted_op.apply(NumpyVectorArray(U.dofs(source_dofs)), mu))

        Such an operator is mainly useful for
        :class:`empirical interpolation <pymor.operators.ei.EmpiricalInterpolatedOperator>`
        where the evaluation of the original operator only needs to be known
        for few selected degrees of freedom. If the operator has a small
        stencil, only few `source_dofs` will be needed to evaluate the
        restricted operator which can make its evaluation very fast
        compared to evaluating the original operator.

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
        """Sum of two operators."""
        pass

    def __sub__(self, other):
        return self + (- other)

    @abstractmethod
    def __mul__(self, other):
        """Product of operator by a scalar."""
        pass

    def __rmul__(self, other):
        return self * other

    @abstractmethod
    def __matmul__(self, other):
        """Concatenation of two operators."""
        pass

    def __neg__(self):
        return self * (-1.)
