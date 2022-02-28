# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.algorithms import genericsolvers
from pymor.core.base import abstractmethod
from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError, LinAlgError
from pymor.parameters.base import ParametricObject
from pymor.parameters.functionals import ParameterFunctional
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


class Operator(ParametricObject):
    """Interface for |Parameter| dependent discrete operators.

    An operator in pyMOR is simply a mapping which for any given
    |parameter values| maps vectors from its `source` |VectorSpace|
    to vectors in its `range` |VectorSpace|.

    Note that there is no special distinction between functionals
    and operators in pyMOR. A functional is simply an operator with
    |NumpyVectorSpace| `(1)` as its `range` |VectorSpace|.

    Attributes
    ----------
    solver_options
        If not `None`, a dict which can contain the following keys:

        :'inverse':           solver options used for
                              :meth:`~Operator.apply_inverse`
        :'inverse_adjoint':   solver options used for
                              :meth:`~Operator.apply_inverse_adjoint`
        :'jacobian':          solver options for the operators returned
                              by :meth:`~Operator.jacobian`
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
            The |parameter values| for which to evaluate the operator.

        Returns
        -------
        |VectorArray| of the operator evaluations.
        """
        pass

    def apply2(self, V, U, mu=None):
        """Treat the operator as a 2-form and apply it to V and U.

        This method is usually implemented as ``V.inner(self.apply(U))``.
        In particular, if the operator is a linear operator given by multiplication
        with a matrix M, then `apply2` is given as::

            op.apply2(V, U) = V^T*M*U.

        In the case of complex numbers, note that `apply2` is anti-linear in the
        first variable by definition of `inner`.

        Parameters
        ----------
        V
            |VectorArray| of the left arguments V.
        U
            |VectorArray| of the right arguments U.
        mu
            The |parameter values| for which to evaluate the operator.

        Returns
        -------
        A |NumPy array| with shape `(len(V), len(U))` containing the 2-form
        evaluations.
        """
        assert self.parameters.assert_compatible(mu)
        assert isinstance(V, VectorArray)
        assert isinstance(U, VectorArray)
        AU = self.apply(U, mu=mu)
        return V.inner(AU)

    def pairwise_apply2(self, V, U, mu=None):
        """Treat the operator as a 2-form and apply it to V and U in pairs.

        This method is usually implemented as ``V.pairwise_inner(self.apply(U))``.
        In particular, if the operator is a linear operator given by multiplication
        with a matrix M, then `apply2` is given as::

            op.apply2(V, U)[i] = V[i]^T*M*U[i].

        In the case of complex numbers, note that `pairwise_apply2` is anti-linear in the
        first variable by definition of `pairwise_inner`.


        Parameters
        ----------
        V
            |VectorArray| of the left arguments V.
        U
            |VectorArray| of the right arguments U.
        mu
            The |parameter values| for which to evaluate the operator.

        Returns
        -------
        A |NumPy array| with shape `(len(V),) == (len(U),)` containing
        the 2-form evaluations.
        """
        assert self.parameters.assert_compatible(mu)
        assert isinstance(V, VectorArray)
        assert isinstance(U, VectorArray)
        assert len(U) == len(V)
        AU = self.apply(U, mu=mu)
        return V.pairwise_inner(AU)

    def apply_adjoint(self, V, mu=None):
        """Apply the adjoint operator.

        For any given linear |Operator| `op`, |parameter values| `mu` and
        |VectorArrays| `U`, `V` in the :attr:`~Operator.source`
        resp. :attr:`~Operator.range` we have::

            op.apply_adjoint(V, mu).dot(U) == V.inner(op.apply(U, mu))

        Thus, when `op` is represented by a matrix `M`, `apply_adjoint` is
        given by left-multplication of (the complex conjugate of) `M` with `V`.

        Parameters
        ----------
        V
            |VectorArray| of vectors to which the adjoint operator is applied.
        mu
            The |parameter values| for which to apply the adjoint operator.

        Returns
        -------
        |VectorArray| of the adjoint operator evaluations.
        """
        if self.linear:
            raise NotImplementedError
        else:
            raise LinAlgError('Operator not linear.')

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        """Apply the inverse operator.

        Parameters
        ----------
        V
            |VectorArray| of vectors to which the inverse operator is applied.
        mu
            The |parameter values| for which to evaluate the inverse operator.
        initial_guess
            |VectorArray| with the same length as `V` containing initial guesses
            for the solution.  Some implementations of `apply_inverse` may
            ignore this parameter.  If `None` a solver-dependent default is used.
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
        assert V in self.range
        assert initial_guess is None or initial_guess in self.source and len(initial_guess) == len(V)
        from pymor.operators.constructions import FixedParameterOperator
        assembled_op = self.assemble(mu)
        if assembled_op != self and not isinstance(assembled_op, FixedParameterOperator):
            return assembled_op.apply_inverse(V, initial_guess=initial_guess, least_squares=least_squares)

        options = self.solver_options.get('inverse') if self.solver_options else None
        options = (None if options is None else
                   {'type': options} if isinstance(options, str) else
                   options.copy())
        solver_type = None if options is None else options['type']

        if self.linear:
            if solver_type is None or solver_type == 'to_matrix':
                if solver_type is None:
                    self.logger.warning(f'No specialized linear solver available for {self}.')
                    self.logger.warning('Trying to solve by converting to NumPy matrix.')
                from pymor.algorithms.rules import NoMatchingRuleError
                try:
                    from pymor.algorithms.to_matrix import to_matrix
                    from pymor.operators.numpy import NumpyMatrixOperator
                    mat = to_matrix(assembled_op)
                    mat_op = NumpyMatrixOperator(mat)
                    v = mat_op.range.from_numpy(V.to_numpy())
                    i = None if initial_guess is None else v.source.from_numpy(initial_guess.to_numpy())
                    u = mat_op.apply_inverse(v, initial_guess=i, least_squares=least_squares)
                    return self.source.from_numpy(u.to_numpy())
                except (NoMatchingRuleError, NotImplementedError):
                    if solver_type == 'to_matrix':
                        raise InversionError
                    else:
                        self.logger.warning('Failed.')
            self.logger.warning('Solving with unpreconditioned iterative solver.')
            return genericsolvers.apply_inverse(assembled_op, V, initial_guess=initial_guess,
                                                options=options, least_squares=least_squares)
        else:
            from pymor.algorithms.newton import newton
            from pymor.core.exceptions import NewtonError

            assert solver_type is None or solver_type == 'newton'
            options = options or {}
            options.pop('type', None)
            options['least_squares'] = least_squares

            with self.logger.block('Solving nonlinear problem using newton algorithm ...'):
                R = V.empty(reserve=len(V))
                for i in range(len(V)):
                    try:
                        R.append(newton(self, V[i],
                                        initial_guess=initial_guess[i] if initial_guess is not None else None,
                                        mu=mu,
                                        **options)[0])
                    except NewtonError as e:
                        raise InversionError(e) from e
            return R

    def apply_inverse_adjoint(self, U, mu=None, initial_guess=None, least_squares=False):
        """Apply the inverse adjoint operator.

        Parameters
        ----------
        U
            |VectorArray| of vectors to which the inverse adjoint operator is applied.
        mu
            The |parameter values| for which to evaluate the inverse adjoint operator.
        initial_guess
            |VectorArray| with the same length as `U` containing initial guesses
            for the solution.  Some implementations of `apply_inverse_adjoint` may
            ignore this parameter.  If `None` a solver-dependent default is used.
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
        from pymor.operators.constructions import FixedParameterOperator
        if not self.linear:
            raise LinAlgError('Operator not linear.')
        assembled_op = self.assemble(mu)
        if assembled_op != self and not isinstance(assembled_op, FixedParameterOperator):
            return assembled_op.apply_inverse_adjoint(U, initial_guess=initial_guess, least_squares=least_squares)
        else:
            # use generic solver for the adjoint operator
            from pymor.operators.constructions import AdjointOperator
            options = {'inverse': self.solver_options.get('inverse_adjoint') if self.solver_options else None}
            adjoint_op = AdjointOperator(self, with_apply_inverse=False, solver_options=options)
            return adjoint_op.apply_inverse(U, mu=mu, initial_guess=initial_guess, least_squares=least_squares)

    def jacobian(self, U, mu=None):
        """Return the operator's Jacobian as a new |Operator|.

        Parameters
        ----------
        U
            Length 1 |VectorArray| containing the vector for which to compute
            the Jacobian.
        mu
            The |parameter values| for which to compute the Jacobian.

        Returns
        -------
        Linear |Operator| representing the Jacobian.
        """
        if self.linear:
            if self.parametric:
                return self.assemble(mu)
            else:
                return self
        else:
            raise NotImplementedError

    def d_mu(self, parameter, index=0):
        """Return the operator's derivative with respect to a given parameter.

        Parameters
        ----------
        parameter
            The parameter w.r.t. which to return the derivative.
        index
            Index of the parameter's component w.r.t which to return the derivative.

        Returns
        -------
        New |Operator| representing the partial derivative.
        """
        if parameter in self.parameters:
            raise NotImplementedError
        else:
            from pymor.operators.constructions import ZeroOperator
            return ZeroOperator(self.range, self.source, name=self.name + '_d_mu')

    def as_range_array(self, mu=None):
        """Return a |VectorArray| representation of the operator in its range space.

        In the case of a linear operator with |NumpyVectorSpace| as
        :attr:`~Operator.source`, this method returns for given |parameter values|
        `mu` a |VectorArray| `V` in the operator's :attr:`~Operator.range`,
        such that ::

            V.lincomb(U.to_numpy()) == self.apply(U, mu)

        for all |VectorArrays| `U`.

        Parameters
        ----------
        mu
            The |parameter values| for which to return the |VectorArray|
            representation.

        Returns
        -------
        V
            The |VectorArray| defined above.
        """
        assert isinstance(self.source, NumpyVectorSpace) and self.linear
        assert self.source.dim <= as_array_max_length()
        return self.apply(self.source.from_numpy(np.eye(self.source.dim)), mu=mu)

    def as_source_array(self, mu=None):
        """Return a |VectorArray| representation of the operator in its source space.

        In the case of a linear operator with |NumpyVectorSpace| as
        :attr:`~Operator.range`, this method returns for given |parameter values|
        `mu` a |VectorArray| `V` in the operator's :attr:`~Operator.source`,
        such that ::

            self.range.make_array(V.inner(U).T) == self.apply(U, mu)

        for all |VectorArrays| `U`.

        Parameters
        ----------
        mu
            The |parameter values| for which to return the |VectorArray|
            representation.

        Returns
        -------
        V
            The |VectorArray| defined above.
        """
        assert isinstance(self.range, NumpyVectorSpace) and self.linear
        assert self.range.dim <= as_array_max_length()
        return self.apply_adjoint(self.range.from_numpy(np.eye(self.range.dim)), mu=mu)

    def as_vector(self, mu=None):
        """Return a vector representation of a linear functional or vector operator.

        Depending on the operator's :attr:`~Operator.source` and
        :attr:`~Operator.range`, this method is equivalent to calling
        :meth:`~Operator.as_range_array` or :meth:`~Operator.as_source_array`
        respectively. The resulting |VectorArray| is required to have length 1.

        Parameters
        ----------
        mu
            The |parameter values| for which to return the vector representation.

        Returns
        -------
        V
            |VectorArray| of length 1 containing the vector representation.
        """
        if not self.linear:
            raise TypeError('This nonlinear operator does not represent a vector or linear functional.')
        if self.source.is_scalar:
            return self.as_range_array(mu)
        elif self.range.is_scalar:
            return self.as_source_array(mu)
        else:
            raise TypeError('This operator does not represent a vector or linear functional.')

    def assemble(self, mu=None):
        """Assemble the operator for given |parameter values|.

        The result of the method strongly depends on the given operator.
        For instance, a matrix-based operator will assemble its matrix, a |LincombOperator|
        will try to form the linear combination of its operators, whereas an arbitrary
        operator might simply return a
        :class:`~pymor.operators.constructions.FixedParameterOperator`.
        The only assured property of the assembled operator is that it no longer
        depends on a |Parameter|.

        Parameters
        ----------
        mu
            The |parameter values| for which to assemble the operator.

        Returns
        -------
        Parameter-independent, assembled |Operator|.
        """
        if self.parametric:
            from pymor.operators.constructions import FixedParameterOperator

            return FixedParameterOperator(self, mu=mu, name=self.name + '_assembled')
        else:
            return self

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
        """Try to assemble a linear combination of the given operators.

        Returns a new |Operator| which represents the sum ::

            c_1*O_1 + ... + c_N*O_N + s*I

        where `O_i` are |Operators|, `c_i`, `s` scalar coefficients and `I` the identity.

        This method is called in the :meth:`assemble` method of |LincombOperator| on
        the first of its operators. If an assembly of the given linear combination
        is possible, e.g. the linear combination of the system matrices of the
        operators can be formed, then the assembled operator is returned.
        Otherwise, the method returns `None` to indicate that assembly is not possible.

        Parameters
        ----------
        operators
            List of |Operators| `O_i` whose linear combination is formed.
        coefficients
            List of the corresponding linear coefficients `c_i`.
        identity_shift
            The coefficient `s`.
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
            :attr:`~Operator.range` to which to restrict.

        Returns
        -------
        restricted_op
            The restricted operator as defined above. The operator will have
            |NumpyVectorSpace| `(len(source_dofs))` as :attr:`~Operator.source`
            and |NumpyVectorSpace| `(len(dofs))` as :attr:`~Operator.range`.
        source_dofs
            One-dimensional |NumPy array| of source degrees of freedom as
            defined above.
        """
        raise NotImplementedError

    def _add_sub(self, other, sign):
        if not isinstance(other, Operator):
            return NotImplemented
        from pymor.operators.constructions import LincombOperator
        if self.name != 'LincombOperator' or not isinstance(self, LincombOperator):
            if other.name == 'LincombOperator' and isinstance(other, LincombOperator):
                operators = (self,) + other.operators
                coefficients = (1.,) + (other.coefficients if sign == 1. else tuple(-c for c in other.coefficients))
            else:
                operators, coefficients = (self, other), (1., sign)
        elif other.name == 'LincombOperator' and isinstance(other, LincombOperator):
            operators = self.operators + other.operators
            coefficients = self.coefficients + (other.coefficients if sign == 1.
                                                else tuple(-c for c in other.coefficients))
        else:
            operators, coefficients = self.operators + (other,), self.coefficients + (sign,)

        return LincombOperator(operators, coefficients, solver_options=self.solver_options)

    def _radd_sub(self, other, sign):
        if other == 0:
            return self
        assert not isinstance(other, Operator)  # this is always handled by _add_sub
        return NotImplemented

    def __add__(self, other):
        return self._add_sub(other, 1.)

    def __sub__(self, other):
        return self._add_sub(other, -1.)

    def __radd__(self, other):
        return self._radd_sub(other, 1.)

    def __rsub__(self, other):
        return self._radd_sub(other, -1.)

    def __mul__(self, other):
        assert isinstance(other, (Number, ParameterFunctional))
        from pymor.operators.constructions import LincombOperator
        if self.name != 'LincombOperator' or not isinstance(self, LincombOperator):
            return LincombOperator((self,), (other,))
        else:
            return self.with_(coefficients=tuple(c * other for c in self.coefficients))

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        """Concatenation of two operators."""
        if not isinstance(other, Operator):
            return NotImplemented
        from pymor.operators.constructions import ConcatenationOperator
        if isinstance(other, ConcatenationOperator):
            return NotImplemented
        else:
            return ConcatenationOperator((self, other))

    def __neg__(self):
        return self * (-1.)

    def __str__(self):
        return f'{self.name}: R^{self.source.dim} --> R^{self.range.dim}  ' \
               f'(parameters: {self.parameters}, class: {self.__class__.__name__})'


@defaults('value')
def as_array_max_length(value=100):
    return value
