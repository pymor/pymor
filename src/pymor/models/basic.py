# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.timestepping import TimeStepper
from pymor.models.interface import Model
from pymor.operators.constructions import IdentityOperator, VectorOperator, ZeroOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


class StationaryModel(Model):
    """Generic class for models of stationary problems.

    This class describes discrete problems given by the equation::

        L(u(μ), μ) = F(μ)

    with a vector-like right-hand side F and a (possibly non-linear) operator L.

    Note that even when solving a variational formulation where F is a
    functional and not a vector, F has to be specified as a vector-like
    |Operator| (mapping scalars to vectors). This ensures that in the complex
    case both L and F are anti-linear in the test variable.

    Parameters
    ----------
    operator
        The |Operator| L.
    rhs
        The vector F. Either a |VectorArray| of length 1 or a vector-like
        |Operator|.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    dual_operator
        The dual |Operator| L* of L, which is e.g. used for computing the gradient
        of the output_functional. See Section 1.6.2 in [HPUU09]_ for more details.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    """

    def __init__(self, operator, rhs, output_functional=None, products=None,
                 error_estimator=None, dual_operator=None,
                 visualizer=None, name=None):

        if isinstance(rhs, VectorArray):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')

        assert rhs.range == operator.range and rhs.source.is_scalar and rhs.linear
        assert output_functional is None or output_functional.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = operator.source
        self.linear = operator.linear and (output_functional is None or output_functional.linear)
        if output_functional is not None:
            self.dim_output = output_functional.range.dim

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    {"linear" if self.linear else "non-linear"}\n'
            f'    solution_space:  {self.solution_space}\n'
            f'    dim_output:      {self.dim_output}\n'
        )

    def _compute_solution(self, mu=None, **kwargs):
        return self.operator.apply_inverse(self.rhs.as_range_array(mu), mu=mu)

    def _compute_solution_d_mu_single_direction(self, parameter, index, solution, mu):
        lhs_d_mu = VectorOperator(self.operator.d_mu(parameter, index).apply(solution, mu=mu))
        rhs_d_mu = self.rhs.d_mu(parameter, index)
        rhs_operator = rhs_d_mu - lhs_d_mu
        return self.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu)

    _compute_allowed_kwargs = frozenset({'adjoint_approach'})

    def _compute_output_d_mu(self, U, mu, adjoint_approach=True):
        """compute the gradient of the output functional  w.r.t. the parameters

        Parameters
        ----------
        U
            Internal model state for the given |Parameter value|
        mu
            |Parameter value| for which to compute the gradient
        adjoint_approach
            Use the adjoint approach for a more efficient way of computing the gradient.
            See Section 1.6.2 in [HPUU09]_ for more details.
            So far, this approach is only valid for linear output functionals and symmetric operators.

        Returns
        -------
        The gradient as a numpy array.
        """
        if adjoint_approach is False:
            return super()._compute_output_d_mu(U, mu)
        else:
            assert self.output_functional is not None
            assert self.output_functional.linear
            assert self.dual_operator is not None
            dual_problem = self.with_(operator=self.dual_operator, rhs=self.output_functional.H)
            P = dual_problem.solve(mu)
            gradient = []
            for (parameter, size) in self.parameters.items():
                for index in range(size):
                    output_partial_dmu = self.output_functional.d_mu(parameter, index).apply(U, mu=mu).to_numpy()[0,0]
                    lhs_d_mu = self.operator.d_mu(parameter, index).apply2(U, P, mu=mu)
                    rhs_d_mu = self.rhs.d_mu(parameter, index).apply_adjoint(P, mu=mu).to_numpy()[0,0]
                    gradient.append((output_partial_dmu + rhs_d_mu - lhs_d_mu)[0,0])
        return np.array(gradient)


class InstationaryModel(Model):
    """Generic class for models of instationary problems.

    This class describes instationary problems given by the equations::

        M * ∂_t u(t, μ) + L(u(μ), t, μ) = F(t, μ)
                                u(0, μ) = u_0(μ)

    for t in [0,T], where L is a (possibly non-linear) time-dependent
    |Operator|, F is a time-dependent vector-like |Operator|, and u_0 the
    initial data. The mass |Operator| M is assumed to be linear.

    Parameters
    ----------
    T
        The final time T.
    initial_data
        The initial data `u_0`. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`) which
        applied to `NumpyVectorArray(np.array([1]))` will yield the
        initial data for given |parameter values|.
    operator
        The |Operator| L.
    rhs
        The right-hand side F.
    mass
        The mass |Operator| `M`. If `None`, the identity is assumed.
    time_stepper
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
        to be used by :meth:`~pymor.models.interface.Model.solve`.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    """

    def __init__(self, T, initial_data, operator, rhs, mass=None, time_stepper=None, num_values=None,
                 output_functional=None, products=None, error_estimator=None, visualizer=None, name=None):

        if isinstance(rhs, VectorArray):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')
        if isinstance(initial_data, VectorArray):
            assert initial_data in operator.source
            initial_data = VectorOperator(initial_data, name='initial_data')
        mass = mass or IdentityOperator(operator.source)
        rhs = rhs or ZeroOperator(operator.source, NumpyVectorSpace(1))

        assert isinstance(time_stepper, TimeStepper)
        assert initial_data.source.is_scalar
        assert operator.source == initial_data.range
        assert rhs.linear and rhs.range == operator.range and rhs.source.is_scalar
        assert mass.linear and mass.source == mass.range == operator.source
        assert output_functional is None or output_functional.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator, visualizer=visualizer, name=name)

        self.parameters_internal = {'t': 1}
        self.__auto_init(locals())
        self.solution_space = operator.source
        self.linear = operator.linear and (output_functional is None or output_functional.linear)
        if output_functional is not None:
            self.dim_output = output_functional.range.dim

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    {"linear" if self.linear else "non-linear"}\n'
            f'    T: {self.T}\n'
            f'    solution_space:  {self.solution_space}\n'
            f'    dim_output:      {self.dim_output}\n'
        )

    def with_time_stepper(self, **kwargs):
        return self.with_(time_stepper=self.time_stepper.with_(**kwargs))

    def _compute_solution(self, mu=None, **kwargs):
        mu = mu.with_(t=0.)
        U0 = self.initial_data.as_range_array(mu)
        U = self.time_stepper.solve(operator=self.operator,
                                    rhs=None if isinstance(self.rhs, ZeroOperator) else self.rhs,
                                    initial_data=U0,
                                    mass=None if isinstance(self.mass, IdentityOperator) else self.mass,
                                    initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)
        return U

    def to_lti(self):
        """Convert model to |LTIModel|.

        This method interprets the given model as an |LTIModel|
        in the following way::

            - self.operator        -> A
            self.rhs               -> B
            self.output_functional -> C
            None                   -> D
            self.mass              -> E
        """
        if self.output_functional is None:
            raise ValueError('No output defined.')
        A = - self.operator
        B = self.rhs
        C = self.output_functional
        E = self.mass

        if not all(op.linear for op in [A, B, C, E]):
            raise ValueError('Operators not linear.')

        from pymor.models.iosys import LTIModel
        return LTIModel(A, B, C, E=E, visualizer=self.visualizer)
