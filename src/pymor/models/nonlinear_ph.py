# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.models.interface import Model
from pymor.operators.constructions import IdentityOperator, LinearInputOperator, VectorOperator, ZeroOperator
from pymor.solvers.newton import NewtonSolver
from pymor.vectorarrays.interface import VectorArray


class NonlinearPHModel(Model):
    r"""Nonlinear port-Hamiltonian input-output model.

    The model represents

    .. math::
        E(\mu) \dot{x}(t, \mu) & = (J(\mu) - R(\mu)) z(x, \mu) + (G(\mu) - P(\mu)) u(t), \\
                     y(t, \mu) & = (G(\mu) + P(\mu))^T z(x, \mu) + (S(\mu) - N(\mu)) u(t),

    where the co-energy vector is

    .. math::
        z(x, \mu) = Q(\mu)x + dh(x, \mu).

    Here, ``dh`` is an |Operator| representing the nonlinear contribution
    to the co-energy vector.

    The system is port-Hamiltonian when there is a continuously
    differentiable Hamiltonian :math:`\mathcal{H}` satisfying

    .. math::
        \nabla_x \mathcal{H}(x, \mu)
        = E(\mu)^T z(x, \mu),

    and

    .. math::
        \Gamma(\mu)
        =
        \begin{bmatrix}
            J(\mu) & G(\mu) \\
            -G(\mu)^T & N(\mu)
        \end{bmatrix},
        \qquad
        \mathcal{W}(\mu)
        =
        \begin{bmatrix}
            R(\mu) & P(\mu) \\
            P(\mu)^T & S(\mu)
        \end{bmatrix}

    satisfy

    .. math::
        \Gamma(\mu)^T = -\Gamma(\mu),
        \qquad
        \mathcal{W}(\mu)
        = \mathcal{W}(\mu)^T
        \succcurlyeq 0.

    In the special case :math:`E=I`, with symmetric :math:`Q`
    the Hamiltonian is

    .. math::
        \mathcal{H}(x, \mu)
        =
        \frac{1}{2}x^TQ(\mu)x + h(x, \mu).

    Parameters
    ----------
    J
        The |Operator| J.
    R
        The |Operator| R.
    G
        The |Operator| G.
    dh
        The nonlinear |Operator| ``dh`` or `None` (then ``dh`` is assumed to be zero).
    P
        The |Operator| P or `None` (then P is assumed to be zero).
    S
        The |Operator| S or `None` (then S is assumed to be zero).
    N
        The |Operator| N or `None` (then N is assumed to be zero).
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    Q
        The |Operator| Q or `None` (then Q is assumed to be identity).
    T
        The final time T. If `None`, no time-dependent solution can
        be computed.
    initial_data
        The initial data :math:`x_0`. Either a |VectorArray| of length
        1 or, for the |Parameter|-dependent case, a vector-like
        |Operator|, i.e. a linear |Operator| with ``source.dim == 1``,
        which yields the initial data for given |parameter values|.
        If `None`, the initial data is assumed to be zero.
    time_stepper
        The :class:`time-stepper
        <pymor.algorithms.timestepping.TimeStepper>` to be used by
        :meth:`~pymor.models.interface.Model.solve`.

        If `None` and `T` is specified, an
        :class:`~pymor.algorithms.timestepping.ImplicitEulerTimeStepper`
        with a Newton solver is used.
    nt
        The number of time steps used by the default implicit Euler
        time-stepper. This argument is ignored when `time_stepper` is
        specified.
    num_values
        The number of returned vectors of the solution trajectory. If
        `None`, each intermediate vector that is calculated is
        returned.
    error_estimator
        An error estimator for the problem. This can be any object
        with an ``estimate_error(U, mu, model)`` method. If
        `error_estimator` is not `None`, an ``estimate_error(U, mu)``
        method is added to the model which calls
        ``error_estimator.estimate_error(U, mu, self)``.
    visualizer
        A visualizer for the problem. This can be any object with a
        ``visualize(U, model, ...)`` method. If `visualizer` is not
        `None`, a ``visualize(U, *args, **kwargs)`` method is added to
        the model which forwards its arguments to the visualizer's
        ``visualize`` method.
    name
        Name of the system.

    Attributes
    ----------
    order
        The order of the system.
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
    J
        The |Operator| J.
    R
        The |Operator| R.
    G
        The |Operator| G.
    dh
        The nonlinear |Operator| ``dh``.
    P
        The |Operator| P.
    S
        The |Operator| S.
    N
        The |Operator| N.
    E
        The |Operator| E.
    Q
        The |Operator| Q.
    T
        The final time.
    initial_data
        The initial-data.
    time_stepper
        The time-stepper used for time integration.
    solution_space
        The |VectorSpace| in which the state trajectory lives.
    """

    linear = False
    cache_region = 'memory'

    def __init__(self, J, R, G, dh=None, P=None, S=None, N=None, E=None, Q=None,
                 T=None, initial_data=None, time_stepper=None, nt=None, num_values=None,
                 error_estimator=None, visualizer=None, name=None):
        assert J.linear
        assert J.source == J.range

        assert R.linear
        assert R.source == J.source
        assert R.range == J.source

        assert G.linear
        assert G.range == J.source

        P = P or ZeroOperator(G.range, G.source)
        assert P.linear
        assert P.source == G.source
        assert P.range == J.source

        S = S or ZeroOperator(G.source, G.source)
        assert S.linear
        assert S.source == G.source
        assert S.range == G.source

        N = N or ZeroOperator(G.source, G.source)
        assert N.linear
        assert N.source == G.source
        assert N.range == G.source

        E = E or IdentityOperator(J.source)
        assert E.linear
        assert E.source == J.source
        assert E.range == J.source

        Q = Q or IdentityOperator(J.source)
        assert Q.linear
        assert Q.source == J.source
        assert Q.range == J.source

        dh = dh or ZeroOperator(J.source, J.source)
        assert dh.source == J.source
        assert dh.range == J.source

        assert T is None or T > 0

        if T is not None:
            if initial_data is None:
                initial_data = J.source.zeros(1)
            if isinstance(initial_data, VectorArray):
                assert initial_data in J.source
                assert len(initial_data) == 1
                initial_data = VectorOperator(initial_data, name='initial_data')
            assert initial_data.source.is_scalar
            assert initial_data.range == J.source

            if time_stepper is None:
                newton_solver = NewtonSolver(relax='armijo', error_measure='residual')
                time_stepper = ImplicitEulerTimeStepper(nt=nt, solver=newton_solver)
        else:
            if initial_data is not None:
                raise ValueError('Initial data is given but T is not.')
            if time_stepper is not None:
                raise ValueError('Time-stepper is given but T is not.')

        super().__init__(dim_input=G.source.dim, error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())
        self.solution_space = J.source
        self.dim_input = G.source.dim
        self.dim_output = G.source.dim

        # R-J instead of J-R as the ph_operator is applied on the LHS later
        self.ph_operator = (R - J) @ (Q + dh)
        #self.ph_operator = NonlinearPHOperator(J, R, Q, dh)
        self.rhs = LinearInputOperator(G - P)
        self.output_operator = G + P
        self.feedthrough = LinearInputOperator(S - N)

    def __str__(self):

        string = (
            f' {self.name}\n'
            f'    nonlinear port-Hamiltonian system\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
            f'    solution_space:      {self.solution_space}'
        )
        return string

    def _grad_H(self, X, mu=None):
        return self.Q.apply(X, mu=mu) + self.dh.apply(X, mu=mu)

    def _compute(self, quantities, data, mu):
        if 'solution' in quantities or 'output' in quantities:
            assert self.T is not None

            compute_solution = 'solution' in quantities
            compute_output = 'output' in quantities

            iterator = self.time_stepper.iterate(
                0,
                self.T,
                self.initial_data.as_range_array(mu),
                self.ph_operator,
                rhs=self.rhs,
                mass=None if isinstance(self.E, IdentityOperator) else self.E,
                mu=mu,
                num_values=self.num_values,
            )

            if self.num_values is None:
                try:
                    n = self.time_stepper.estimate_time_step_count(0, self.T) + 1
                except NotImplementedError:
                    n = 0
            else:
                n = self.num_values + 1

            if compute_solution:
                data['solution'] = self.solution_space.empty(reserve=n)

            if compute_output:
                data['output'] = np.empty((self.dim_output, n))
                data_output_extra = []

            for i, (x, t) in enumerate(iterator):
                if compute_solution:
                    data['solution'].append(x)
                if compute_output:
                    mu_t = mu.at_time(t)
                    grad = self._grad_H(x, mu=mu_t)
                    y = self.output_operator.apply_adjoint(grad, mu=mu_t).to_numpy()
                    y += self.feedthrough.as_range_array(mu=mu_t).to_numpy()

                    if i < n:
                        data['output'][:, i] = y.ravel()
                    else:
                        data_output_extra.append(y)
            if compute_output:
                if data_output_extra:
                    data['output'] = np.hstack([data['output'], data_output_extra])

                if data['output'].shape[1] < i + 1:
                    data['output'] = data['output'][:, :i + 1]

            if compute_solution:
                quantities.remove('solution')
            if compute_output:
                quantities.remove('output')

        super()._compute(quantities, data, mu=mu)
