# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Generic time-stepping algorithms for the solution of instationary problems.

The algorithms are generic in the sense that each algorithms operates exclusively
on |Operators| and |VectorArrays|. In particular, the algorithms
can also be used to turn an arbitrary stationary |Model| provided
by an external library into an instationary |Model|.

Currently, implementations of :func:`explicit_euler` and :func:`implicit_euler`
time-stepping are provided, based on the :class:`ExplicitEulerTimeStepper` and
:class:`ImplicitEulerTimeStepper` classes. These derive from :class:`TimeStepper`,
which defines a common interface that has to be fulfilled by the time-steppers used
by |InstationaryModel|.
"""

from numbers import Number
import numpy as np

from pymor.core.base import ImmutableObject, abstractmethod
from pymor.operators.constructions import IdentityOperator, VectorArrayOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu
from pymor.tools import floatcmp
from pymor.vectorarrays.interface import VectorArray


class TimeStepper(ImmutableObject):
    """Interface for time-stepping algorithms.

    Algorithms implementing this interface solve time-dependent initial value problems
    of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t),
                     u(mu, t_0) = u_0(mu).

    Time-steppers used by |InstationaryModel| have to fulfill this interface. Time evolution can either be performed
    by calling :meth:`solve`, or by manually calling :meth:`bootstrap` and :meth:`step`.

    Note that :meth:`bootstrap` and :meth:`step` are designed to support a minimal implementation of :meth:`solve`,
    as the outer loop in :meth:`solve` may arise manually in several places.

    Parameters
    ----------
    initial_time
        The time at which to begin time-stepping.
    end_time
        The time until which to perform time-stepping.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate vector that is
        calculated is returned. Else an interpolation of the calculated vectors on an equidistant temporal grid is
        returned, using an appropriate interpolation of the respective time stepper.

    Attributes
    ----------
    steps
        The number of previous steps required to determine the next. I.e. steps == 1 for a single-step method and
        steps > 1 for a multi-step method.
    """

    initial_time = None
    end_time = None
    steps = None
    implicit = False
    num_values = None

    def __init__(self, initial_time, end_time, num_values=None):
        assert isinstance(initial_time, Number)
        assert isinstance(end_time, Number)
        assert end_time > initial_time
        assert not num_values or (isinstance(num_values, Number) and num_values > 1)
        if num_values:
            self._save_every = (end_time - initial_time) / (num_values - 1)
        self.__auto_init(locals())

    def bootstrap(self, initial_data, operator, rhs=None, mass=None, mu=None, reserve=False):
        """Performs required initial work, e.g.

        - setting the initial values for single-step methods or an initial bootstrapping for multi-step methods; or
        - pre-assembling stationary operators.

        Note: internally calls :meth:_bootstrap, which has to be implemented by every time stepper.

        Parameters
        ----------
        initial_data
            The solution vector at `initial_time`.
        operator
            The |Operator| A.
        rhs
            The right-hand side F (either |VectorArray| of length 1 or |Operator| with
            `source.dim == 1`). If `None`, zero right-hand side is assumed.
        mass
            The |Operator| M. If `None`, the identity operator is assumed.
        mu
            |Parameter values| for which `operator` and `rhs` are to be evaluated. The current
            time is added to `mu` with key `t`.
        reserve
            If True and self.num_values is specified, pre-allocates the returned |VectorArray| for all time steps.

        Returns
        -------
        t
            Simply self.initial_time, to be used in loops as in :meth:`solve`
        U
            An empty |VectorArray| to hold the solution trajectory, to be used in loops as in :meth:`solve`.
        data
            A dictionary containing information required in :meth:`step` (like previously computed steps), as well
            as cached data required by the time stepper (such as pre-assembled operators).
        """
        if not self.logging_disabled: # explicitly checking if logging is disabled saves some cpu cycles
            self.logger.debug(f'bootstrapping (mu={mu}) ...')

        assert isinstance(operator, Operator)
        assert operator.source == operator.range

        rhs = rhs or ZeroOperator(operator.source, NumpyVectorSpace(1))
        assert isinstance(rhs, (Operator, VectorArray))
        if isinstance(rhs, Operator):
            assert rhs.source.dim == 1
            assert rhs.range == operator.range
        else:
            assert rhs in operator.range
            assert len(rhs) == 1
            rhs = VectorArrayOperator(rhs)

        mass = mass or IdentityOperator(operator.source)
        assert isinstance(mass, Operator)
        assert operator.source == mass.source == mass.range

        if isinstance(initial_data, Operator):
            assert initial_data.source.dim == 1
            assert initial_data.range == operator.source
        else:
            assert initial_data in operator.source and len(initial_data) == 1
            initial_data = VectorOperator(initial_data)

        assert isinstance(self.steps, Number) and self.steps >= 1

        if self.steps == 1:
            # single-step method
            U0, data = self._bootstrap(initial_data=initial_data, operator=operator, rhs=rhs, mass=mass, mu=mu)
            assert len(U0) == 1
            data['time_points'] = [self.initial_time,]
            data['_previous_steps'] = U0
            data['_previous_interpolation'] = lambda t: U0
        else:
            # multi-step method
            time_points, steps, interpolation, data = self._bootstrap(
                    initial_data=initial_data, operator=operator, rhs=rhs, mass=mass, mu=mu)
            assert len(time_points) == self.steps
            assert len(steps) == self.steps
            assert floatcmp.float_cmp(self.initial_time, time_points[0])
            data['time_points'] = time_points
            data['_previous_steps'] = steps
            data['_previous_interpolation'] = interpolation

        data['mu'] = mu
        if self.num_values:
            data['_next_requested'] = self.initial_time
        else:
            data['_next_requested'] = 0

        if reserve and self.num_values:
            solution_array = operator.source.empty(reserve=self.num_values)
        else:
            solution_array = operator.source.empty()

        return self.initial_time, solution_array, data

    @abstractmethod
    def _bootstrap(self, initial_data, operator, rhs, mass, mu):
        """Called in :meth:`bootstrap` to allow for any preparatory work and interpolation of initial values.

        The provided arguments are checked in
        :meth:`bootstrap` before being passed along. Implementors are expected to carry out any possible pre-assembly
        and to store all required data (including operator, rhs, mass) in data. In addition to this data,

        - single-step methods are expected to simply return the interpolated initial data, while
        - multi-step methods are expected return the required initial steps along with their corresponding time points
          and an interpolation method.

        We thus have
        - U0, data = self._bootstrap(...) if self.steps == 1 and
        - time_points, steps, interpolation, data = self._bootstrap(...) if self.steps > 1

        Parameters
        ----------
        initial_data
            An |Operator| representing the solution at self.initial_time.
        operator
            An |Operator| representing A.
        rhs
            An |Operator| representing F.
        mass
            An |Operator| representing M.
        mu
            |Parameter values| for which `operator` and `rhs` are to be evaluated. The current
            time is added to `mu` with key `t`.

        Returns
        -------
        (U0, data)
            If self.steps == 1, where U0 is the solution at self.initial_time for the given mu.
        (time_points, steps, interpolation, data)
            If self.steps > 1, where time_points is a list of time instances (with time_points[0] == self.initial_time)
            and steps a |VectorArray| of solution snapshots with len(time_instances) == len(steps), such that steps[k]
            is the solution at time instance time_points[k], for 0 <= k < self.steps; and interpolation is a function
            t -> U(t) which returns a |VectorArray| of length 1, containing the solution U(t) at time instance t, for
            self.initial_time <= t <= time_points[-1].
        """
        pass

    def step(self, t_n, data, mu=None):
        """Computes the next requested solution snapshot.

        Returns (t_next, U_next), such that t_next can be used in loops and U_next can be appended to the soliution
        trajectory returned by :meth:`bootstrap`.

        Behaviour depends on self.num_values:
        - If self.num_values is provided, performs as many actual steps (by calling self._step) as required to obtain
          an interpolation of the solution, U(t_next) at the next required interpolation point t_next >= t_n, and
          returns (t_next, U(t_next).
        - If self.num_values is not provided, performs a single step (by calling self._step) starting from t_n, to
          compute (t_{n + 1}, U(t_{n + 1})).

        Note that the determination of the required step length, dt = t_{n + 1} - t_n, depends on the time stepper
        and is carried out in :meth:_step. The actually stepped time instances {t_n}_{n <= 0} can be found in
        data['time_points'].

        Parameters
        ----------
        t_n
            Current time as basis for further computation, i.e. the time instance corresponding to the last computed
            solution snapshot U(t_n), which is contained in data['_previous_steps'][-1].
        data
            Dictionary containing cached data pre-computed in :meth:bootstrap as well as the required last
            solution snapshots in data['_previous_steps'] of length self.steps.
        mu
            |Parameter values| for which `operator` and `rhs` are to be evaluated. The current time (as determined
            by the time stepper) is to be added to `mu` with key `t`.

        Returns
        -------
        t_next
            The next time instance determined by the time stepper (either the next computed one or the next
            interpolation point).
        U_next
            A |VectorArray| of length 1 holding the value of the solution at t_next (either the result of
            :meth:`self._step` or an interpolation).
        data
            A dictionary containing information required in :meth:`step` (like previously computed steps), as well
            as cached data required by the time stepper (such as pre-assembled operators).
        """
        assert data['mu'] == mu

        def step_and_cache(t, logging_prefix=''):
            if not self.logging_disabled:
                self.logger.debug(f'{logging_prefix}t={t}: stepping (mu={mu}) ...')
            _t, _U, _interpolate = self._step(t, data, mu)
            data['time_points'] += [_t,]
            if self.steps == 1:
                data['_previous_steps'] = _U
            else:
                data['_previous_steps'] = data['_previous_steps'][1:].copy()
                data['_previous_steps'].append(_U)
            data['_previous_interpolation'] = _interpolate
            return _t, _U, _interpolate

        if not self.num_values:
            # the trajectory is requested as is
            if '_next_requested' in data:
                # - special case: initial values
                if self.steps == 1:
                    data.pop('_next_requested')
                    t = data['time_points'][-1]
                    U = data['_previous_steps'][-1].copy()
                    return t, U
                else:
                    step = data['_next_requested']
                    t = data['time_points'][step]
                    U = data['_previous_steps'][step]
                    if step == self.steps - 1:
                        data.pop('_next_requested')
                    else:
                        data['_next_requested'] += 1
            else:
                # - the usual case, perform a step
                t_np1, U_np1, _ = step_and_cache(t_n)
                return t_np1, U_np1
        else:
            # and interpolation of the trajectory is requested
            # if not self.logging_disabled:
                # self.logger.debug(f't={t_n}: stepping for {mu}:')
            if not data['_next_requested'] > data['time_points'][-1]:
                if not self.logging_disabled:
                    self.logger.debug(
                            f't={t_n}: existing data suffices for t={data["_next_requested"]}, interpolating ...')
                # the computed data still suffices
                # - compute the interpolation at the next requested point
                t_np1 = data['_next_requested']
                U_np1 = data['_previous_interpolation'](t_np1)
                # - increment the next requested point
                data['_next_requested'] += self._save_every
                return t_np1, U_np1
            else:
                if not self.logging_disabled:
                    self.logger.debug(f't={t_n}: data missing for t={data["_next_requested"]}, stepping:')
                # we do not have enough data
                # - check if t_n is artificial from the last interpolation
                if data['time_points'][-1] > t_n:
                    if self.steps > 1:
                        raise RuntimeError('Not implemented for multi-step method yet!')
                    t_n = data['time_points'][-1]
                    if not self.logging_disabled:
                        self.logger.debug(f'  moving forward to already computed t={t_n} ...')
                # - take enough actual steps
                while t_n < data['_next_requested']:
                    t_n, _, _ = step_and_cache(t_n, '  ')
                # - compute the interpolation at the next requested point
                # if not self.logging_disabled:
                    # self.logger.debug(f'  interpolating ...')
                t_np1 = data['_next_requested']
                U_np1 = data['_previous_interpolation'](t_np1)
                # - increment the next requested point
                data['_next_requested'] += self._save_every
                return t_np1, U_np1

    @abstractmethod
    def _step(self, t_n, data, mu=None):
        """Called in :meth:`step` to perform the actual next step of the time evolution.

        Parameters
        ----------
        t_n
            Current time as basis for further computation, i.e. the time instance corresponding to the last computed
            solution snapshot U(t_n), which is contained in data['_previous_steps'][-1].
        data
            Dictionary containing cached data pre-computed in :meth:`bootstrap` as well as the required last
            solution snapshots in data['_previous_steps'] of length self.steps.
        mu
            |Parameter values| for which `operator` and `rhs` are to be evaluated. The current time (as determined
            by the time stepper) is to be added to `mu` with key `t`.

        Returns
        -------
        t_np1
            The next time instance t_{n + 1} = t_n + dt, where dt has to be determined by the implementor.
        U_np1
            A |VectorArray| of length 1 containing U(t_{n + 1}).
        interpolation
            A function t -> U(t) which returns a |VectorArray| of length 1, containing the solution U(t) at time
            instance t, for t_n <= t <= t_np1.
        """
        pass

    def solve(self, initial_data, operator, rhs=None, mass=None, mu=None):
        """Apply time-stepper to the equation ::

            M * d_t u + A(u, mu, t) = F(mu, t),
                         u(mu, t_0) = u_0(mu).

        Parameters
        ----------
        initial_data
            The solution vector at `self.initial_time`.
        operator
            The |Operator| A.
        rhs
            The right-hand side F (either |VectorArray| of length 1 or |Operator| with
            `source.dim == 1`). If `None`, zero right-hand side is assumed.
        mass
            The |Operator| M. If `None`, the identity operator is assumed.
        mu
            |Parameter values| for which `operator` and `rhs` are evaluated. The current
            time is added to `mu` with key `t`.

        Returns
        -------
        |VectorArray| containing the solution trajectory. Of length self.num_values if num_values was provided on
        construction, else of unknown length.
        """

        t, U, data = self.bootstrap(
                initial_data=initial_data, operator=operator, rhs=rhs, mass=mass, mu=mu, reserve=True)

        while not (t > self.end_time or np.allclose(t, self.end_time)):
            t, U_t = self.step(t, data, mu=mu)
            U.append(U_t)

        return U


class ImplicitEulerTimeStepper(TimeStepper):
    """Implicit Euler :class:`TimeStepper`.

    Solves equations of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t),
                     u(mu, t_0) = u_0(mu),

    by an implicit Euler time integration.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    initial_time
        The time at which to begin time-stepping.
    end_time
        The time until which to perform time-stepping.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate vector that is
        calculated is returned. Else an interpolation of the calculated vectors on an equidistant temporal grid is
        returned, using an appropriate interpolation of the respective time stepper.
    solver_options
        The |solver_options| used to invert `M + dt*A`.
        The special values `'mass'` and `'operator'` are
        recognized, in which case the solver_options of
        M (resp. A) are used.
    interpolation_order
        Polynomial order (in time) of the interpolation, if num_values is specified: either 0 or 1.
    """

    steps = 1

    def __init__(self, nt, initial_time, end_time, num_values=None, solver_options='operator', interpolation_order=1):
        super().__init__(initial_time=initial_time, end_time=end_time, num_values=num_values)

        assert isinstance(nt, Number)
        assert nt > 0
        assert interpolation_order == 0 or interpolation_order == 1
        self.dt = (self.end_time - self.initial_time) / nt
        self.__auto_init(locals())


    def _bootstrap(self, initial_data, operator, rhs, mass, mu):
        if mu:
            mu = mu.with_(t=self.initial_time)
        else:
            mu = Mu({'t': self.initial_time})
        data = {'_mass': mass}
        # prepare lhs
        M_dt_A = (mass + operator * self.dt).with_(
                solver_options=operator.solver_options if self.solver_options == 'operator' else \
                               mass.solver_options if self.solver_options == 'mass' else \
                               self.solver_options)
        if not M_dt_A.parametric or 't' not in M_dt_A.parameters:
            M_dt_A = M_dt_A.assemble(mu)
        data['_M_dt_A'] = M_dt_A
        # prepare rhs
        if not isinstance(rhs, ZeroOperator):
            dt_F = rhs * self.dt
            if not dt_F.parametric or 't' not in dt_F.parameters:
                dt_F = dt_F.assemble(mu)
            data['_dt_F'] = dt_F
        return initial_data.as_range_array(mu), data

    def _step(self, t_n, data, mu=None):
        U_n = data['_previous_steps'][-1]
        t_np1 = t_n + self.dt
        if mu:
            mu = mu.with_(t=t_np1)
        else:
            mu = Mu({'t': t_np1})
        rhs = data['_mass'].apply(U_n)
        if '_dt_F' in data:
            rhs += data['_dt_F'].as_vector(mu)
        U_np1 = data['_M_dt_A'].apply_inverse(rhs, mu=mu, initial_guess=U_n)

        def interpolate(t):
            assert floatcmp.almost_less(t_n, t)
            assert floatcmp.almost_less(t, t_np1)
            if self.interpolation_order == 0:
                # return previous value, old default in pyMOR
                return U_n
            else:
                # compute P1-Lagrange interpolation
                dt = t_np1 - t_n
                return (t_np1 - t)/dt * U_n + (t - t_n)/dt * U_np1

        return t_np1, U_np1, interpolate


class ExplicitEulerTimeStepper(TimeStepper):
    """Explicit Euler :class:`TimeStepper`.

    Solves equations of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t).
                     u(mu, t_0) = u_0(mu),

    by an explicit Euler time integration.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    initial_time
        The time at which to begin time-stepping.
    end_time
        The time until which to perform time-stepping.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate vector that is
        calculated is returned. Else an interpolation of the calculated vectors on an equidistant temporal grid is
        returned, using an appropriate interpolation of the respective time stepper.
    solver_options
        The |solver_options| used to invert `M + dt*A`.
        The special values `'mass'` and `'operator'` are
        recognized, in which case the solver_options of
        M (resp. A) are used.
    interpolation_order
        Polynomial order (in time) of the interpolation, if num_values is specified: either 0 or 1.
    """

    # Felix: To better express the math, I would like to simply write
    #   U_np1 = M.apply_inverse(F.as_vector(mu)*self.dt + M.apply(U_n) - A.apply(U_n, mu=mu)*self.dt)
    # in _step, using the original M, A and F in _bootstrap. But testing a 1d linear advection upwind
    # FV FOM with 256 spatial DoFs and 1e4 timesteps on my Thinkpad X390 (performance CPU governor, disabled turbo
    # boost) gave an increase of
    #   49.69282579421997s -> 50.267810344696045s
    # and testing a resulting 10-dimensional POD-based ROM gave an increase of:
    #   3.0758347511291504s -> 3.794224977493286
    # Due to the latter, using the less readable code can still be justified.

    steps = 1

    def __init__(self, nt, initial_time, end_time, num_values=None, solver_options='operator', interpolation_order=1):
        super().__init__(initial_time=initial_time, end_time=end_time, num_values=num_values)

        assert isinstance(nt, Number)
        assert nt > 0
        assert interpolation_order == 0 or interpolation_order == 1
        self.dt = (self.end_time - self.initial_time) / nt
        self.__auto_init(locals())

    def _bootstrap(self, initial_data, operator, rhs, mass, mu):
        if mu:
            mu = mu.with_(t=self.initial_time)
        else:
            mu = Mu({'t': self.initial_time})
        data = {}
        # prepare mass
        if isinstance(mass, IdentityOperator):
            mass = None
        data['_mass'] = mass
        # prepare operator
        if not (operator.parametric and 't' in operator.parameters):
            operator = operator.assemble(mu)
        data['_operator'] = operator
        # prepare rhs
        if isinstance(rhs, ZeroOperator):
            rhs = None
        elif not (rhs.parametric and 't' in rhs.parameters):
            rhs = rhs.as_vector(mu)
        data['_rhs'] = rhs
        return initial_data.as_range_array(mu), data

    def _step(self, t_n, data, mu=None):
        # extract data
        U_n = data['_previous_steps'][-1]
        t_np1 = t_n + self.dt
        if mu:
            mu = mu.with_(t=t_n)
        else:
            mu = Mu({'t': t_n})
        M, A, F = data['_mass'], data['_operator'], data['_rhs']
        # the actual step
        if M:
            U_np1 = M.apply(U_n)
        else:
            U_np1 = U_n.copy()
        if isinstance(F, Operator):
            F = F.as_vector(mu)
        if F:
            U_np1.axpy(self.dt, F - A.apply(U_n, mu=mu))
        else:
            U_np1.axpy(-self.dt, A.apply(U_n, mu=mu))
        if M:
            U_np1 = M.apply_inverse(U_np1)

        def interpolate(t):
            assert floatcmp.almost_less(t_n, t)
            assert floatcmp.almost_less(t, t_np1)
            if self.interpolation_order == 0:
                # return previous value, old default in pyMOR
                return U_n
            else:
                # compute P1-Lagrange interpolation
                dt = t_np1 - t_n
                return (t_np1 - t)/dt * U_n + (t - t_n)/dt * U_np1

        return t_np1, U_np1, interpolate


def implicit_euler(A, F, M, U0, t0, t1, nt, mu=None, num_values=None, solver_options='operator'):
    time_stepper = ImplicitEulerTimeStepper(nt=nt, initial_time=t0, end_time=t1, num_values=num_values,
                                            solver_options=solver_options, interpolation_order=0)
    return time_stepper.solve(mu=mu)


def explicit_euler(A, F, U0, t0, t1, nt, mu=None, num_values=None):
    time_stepper = ExplicitEulerTimeStepper(
            nt=nt, initial_time=t0, end_time=t1, num_values=num_values, interpolation_order=0)
    return time_stepper.solve(mu=mu)

