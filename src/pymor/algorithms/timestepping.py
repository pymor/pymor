# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Generic time-stepping algorithms for the solution of instationary problems.

The algorithms are generic in the sense that each algorithms operates exclusively
on |Operators| and |VectorArrays|. In particular, the algorithms
can also be used to turn an arbitrary stationary |Model| provided
by an external library into an instationary |Model|.

Currently, implementations of :class:`ExplicitEulerTimeStepper`, :class:`ImplicitEulerTimeStepper`
and :class:`ExplicitRungeKuttaTimeStepper` are provided, deriving from the common
:class:`TimeStepper` interface.
"""

from numbers import Number
import numpy as np

from pymor.core.base import BasicObject, ImmutableObject, abstractmethod
from pymor.operators.constructions import IdentityOperator, VectorArrayOperator, VectorOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu
from pymor.tools import floatcmp
from pymor.tools.deprecated import Deprecated
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


class TimeStepper(ImmutableObject):
    """Interface for time-stepping algorithms.

    Algorithms implementing this interface solve time-dependent initial value problems
    of the form ::

        M(mu) * d_t u + A(u, mu, t) = F(mu, t),
                         u(mu, t_0) = u_0(mu).

    Time evolution can be performed by calling :meth:`solve`.

    .. note::
        The actual work is done in an iterator derived from :class:`TimeStepperIterator`.

    Parameters
    ----------
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate
        vector that is calculated is returned. Else an interpolation of the calculated vectors on an
        equidistant temporal grid is returned, using an appropriate interpolation of the respective
        time stepper.
    interpolation
        Type of temporal interpolation to be used. Currently implemented are: piecewise constant
        (P0) and piecewise linear (P1).

    Attributes
    ----------
    num_values
        Length of the solution trajectory if known a priori, otherwise zero.
    """

    IteratorType = None
    available_interpolations = ('P0', 'P1')
    num_values = 0

    def __init__(self, num_values=None, interpolation='P1'):
        if num_values is None:
            num_values = 0
        assert (isinstance(num_values, Number) and int(num_values) == num_values)
        num_values = int(num_values)
        if num_values == 0:
            self._disable_interpolation = True
        else:
            self._disable_interpolation = False
        assert interpolation in self.available_interpolations
        self.__auto_init(locals())

    def solve(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None,
              return_iter=False, return_times=False):
        """Apply time-stepper to the equation ::

            M(mu, t) * d_t u + A(u, mu, t) = F(mu, t),
                                u(mu, t_0) = u_0(mu).

        Parameters
        ----------
        initial_time
            The time at which to begin time-stepping.
        end_time
            The time until which to perform time-stepping.
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
            |Parameter values| for which `operator` and `rhs` are evaluated. The current
            time is added to `mu` with key `t`.
        return_iter
            Determines the return data, see below.
        return_times
            Determines the return data, see below.

        Returns
        -------
        U
            If `return_iter` is `False` and `return_times` is `False` (the default), where `U` is a
            |VectorArray| containing the solution trajectory.
        (U, t)
            If `return_iter` is `False` and `return_times` is `True`, where `t` is the list of time
            points corresponding to the solution vectors in `U`.
        iterator
            If `return_iter` is `True`, an iterator yielding either `U_n` (if `return_times` is
            `False`) or `(U_n, t_n)` in each step.
        """

        # all checks are delegated to TimeStepperIterator
        iterator = self.IteratorType(self, initial_time, end_time, initial_data, operator, rhs=rhs, mass=mass, mu=mu,
                                     return_iter=return_iter, return_times=return_times)
        if return_iter:
            return iterator
        elif return_times:
            U = operator.source.empty(reserve=self.num_values)
            t = []
            for U_n, t_n in iterator:
                U.append(U_n, remove_from_other=True)
                t.append(t_n)
            return U, t
        else:
            U = operator.source.empty(reserve=self.num_values)
            for U_n in iterator:
                U.append(U_n, remove_from_other=True)
            return U


class TimeStepperIterator(BasicObject):
    """Base class to derive time-stepper iterators from.

    See :meth:`TimeStepper.solve` for a documentation of the init parameters except `stepper`.

    .. note::
        Derived classes usually only have to implement :meth:`_step`, and optionally
        :meth:`_interpolate` if none of the provided interpolations are suitable. Using the
        interpolation from this base class requires the implementor to store certain data in each
        step (see :meth:`_interpolate`).

    Parameters
    ----------
    stepper
        The associated :class:`TimeStepper`.
    """

    def __init__(
            self, stepper, initial_time, end_time, initial_data, operator, rhs, mass, mu, return_iter, return_times):
        """
        .. note::
            The iterator keeps track of: self.t, always corresponding to the time instance of the
            last returned U; self._last_stepped_point, always corresponding to the time instance of
            the last value actually computed by _step (except after init, where it points to
            something not meaningful); self._next_interpolation_point, always corresponding to the
            next point in time where a value needs to be returned.
        """
        # check input
        assert isinstance(stepper, TimeStepper)

        assert isinstance(initial_time, Number)
        assert isinstance(end_time, Number)
        assert end_time > initial_time

        assert isinstance(operator, Operator)

        rhs = rhs if rhs is not None else ZeroOperator(operator.source, NumpyVectorSpace(1))
        if isinstance(rhs, VectorArray):
            assert len(rhs) == 1
            rhs = VectorArrayOperator(rhs)
        assert isinstance(rhs, Operator)
        assert rhs.source.dim == 1
        assert operator.range == rhs.range

        mass = mass if mass is not None else IdentityOperator(operator.source)
        assert isinstance(mass, Operator)
        assert mass.linear
        assert operator.range == mass.range

        if isinstance(initial_data, VectorArray):
            assert len(initial_data) == 1
            initial_data = VectorOperator(initial_data)
        assert isinstance(initial_data, Operator)
        assert initial_data.source.dim == 1
        assert initial_data.range == operator.source

        self.__auto_init(locals())

        self.t = initial_time
        self.mu = mu or Mu()
        self.initial_data = initial_data.as_vector(self.mu.with_(t=initial_time))

        # prepare interpolation
        if not stepper._disable_interpolation:
            self._interpolation_points_increment = (end_time - initial_time) / (stepper.num_values - 1)
            self._last_stepped_point = initial_time - 1
            self._next_interpolation_point = initial_time

    @abstractmethod
    def _step(self):
        """Called in :meth:`_interpolated_step` to compute the next step of the time evolution.

        Returns
        -------
        U_np1
            A |VectorArray| of length 1 containing U(t_{n + 1}).
        t_np1
            The next time instance t_{n + 1} = t_n + dt, where dt has to be determined by the
            implementor.
        """
        pass

    def _interpolate(self, t):
        """Called in :meth:`_interpolated_step` to compute an interpolated value of the solution.

        If not overridden in derived classes, requires the following data to be present after a call
        to :meth:`_step`:
        - P0: self.t_n, self.t_np1, self.U_n
        - P1: self.t_n, self.t_np1, self.U_n, self.U_np1

        Parameters
        ----------
        t
            The interpolation point within the latest computed step interval.

        Returns
        -------
        U
            A |VectorArray| of length 1 containing U(t).
        """
        t_n, t_np1 = self.t_n, self.t_np1
        assert floatcmp.almost_less(t_n, t)
        assert floatcmp.almost_less(t, t_np1)
        if self.stepper.interpolation == 'P0':
            if not self.logging_disabled:
                self.logger.debug(f't={self._next_interpolation_point}: -> (P0) returning U_n')
            # return previous value, old default in pyMOR
            return self.U_n.copy()
        elif self.stepper.interpolation == 'P1':
            # compute P1-Lagrange interpolation
            U_n, U_np1 = self.U_n, self.U_np1
            # but return node values if t is close enough
            if floatcmp.float_cmp(t, t_n):
                if not self.logging_disabled:
                    self.logger.debug(f't={self._next_interpolation_point}: -> (P1) returning U_n')
                return U_n.copy()
            elif floatcmp.float_cmp(t, t_np1):
                if not self.logging_disabled:
                    self.logger.debug(f't={self._next_interpolation_point}: -> (P1) returning U_np1')
                return U_np1.copy()
            else:
                if not self.logging_disabled:
                    self.logger.debug(f't={self._next_interpolation_point}: -> (P1) computing intermediate value')
                dt = t_np1 - t_n
                return (t_np1 - t)/dt * U_n + (t - t_n)/dt * U_np1
        else:
            raise NotImplementedError(f'{self.interpolation}-interpolation not available!')

    def _interpolated_step(self):
        """
        Returns `(U_next, t_next)` (if `return_times == True`, otherwise `U_next`), behavior depends
        on `num_values`: If `num_values` is provided, performs as many actual steps
        (by calling :meth:`_step`) as required to obtain an interpolation of the solution, U(t_next)
        at the next required interpolation point t_next >= t_n, and returns (U(t_next), t_next). If
        `num_values` is not provided, performs a single step (by calling :meth:`_step`) starting
        from t_n, to compute (U(t_{n + 1}), t_{n + 1}).
        """
        if floatcmp.almost_less(self.end_time, self.t):
            # this is the end
            raise StopIteration
        elif self.stepper._disable_interpolation:
            # the trajectory is requested as is
            U, self.t = self._step()
            if self.return_times:
                return U, self.t
            else:
                return U
        else:
            # an interpolation of the trajectory is requested
            if self._last_stepped_point < self.initial_time:
                # this is the start, take a step to have data and interpolation available next time,
                # but return initial_data
                _, self._last_stepped_point = self._step()
                self.t = self.initial_time
                self._next_interpolation_point += self._interpolation_points_increment
                if self.return_times:
                    return self.initial_data.copy(), self.initial_time
                else:
                    return self.initial_data.copy()
            else:
                # if we do not have enough data, take enough actual steps
                while self._last_stepped_point < self._next_interpolation_point:
                    _, self._last_stepped_point = self._step()
                    # self._last_stepped_point = self.t
                # compute the interpolation at the next requested point
                if not self.logging_disabled:
                    self.logger.debug(f't={self._next_interpolation_point}: interpolating ...')
                self.t = self._next_interpolation_point
                U_next = self._interpolate(self.t)
                self._next_interpolation_point += self._interpolation_points_increment
                if self.return_times:
                    return U_next, self.t
                else:
                    return U_next

    def __iter__(self):
        return self

    def __next__(self):
        return self._interpolated_step()


class SingleStepTimeStepperIterator(TimeStepperIterator):
    """Base class for iterators of single-step methods.

    See :meth:`TimeStepperIterator` for a documentation of the init parameters.

    Note: derived classes only have to implement :meth:`_step_function`, and optionally
    :meth:`_interpolate`
    """

    @abstractmethod
    def _step_function(self, U_n, t_n):
        pass

    def _step(self):
        if not hasattr(self, 'U_n'):
            if not self.logging_disabled:
                self.logger.debug(f't={self.t}: returning initial data (mu={self.mu}) ...')
            self.t_n = self.initial_time   # setting these just in case ...
            self.U_n = self.initial_data   # ... for _interpolate
            self.t_np1 = self.initial_time
            self.U_np1 = self.initial_data.copy()
            return self.U_np1.copy(), self.t_np1
        else:
            # this is the first actual step or a usual step
            self.t_n = self.t_np1
            self.U_n = self.U_np1.copy()
            if not self.logging_disabled:
                self.logger.debug(f't={self.t}: stepping (mu={self.mu}) ...')
            self.U_np1, self.t_np1 = self._step_function(self.U_n, self.t_n)
            return self.U_np1.copy(), self.t_np1


class ImplicitEulerIterator(SingleStepTimeStepperIterator):

    def __init__(
            self, stepper, initial_time, end_time, initial_data, operator, rhs, mass, mu, return_iter, return_times):
        super().__init__(
                stepper, initial_time, end_time, initial_data, operator, rhs, mass, mu, return_iter, return_times)
        self.dt = dt = (end_time - initial_time) / stepper.nt
        # use the ones from base, these are checked and converted in super().__init__()
        A, F, M, mu = self.operator, self.rhs, self.mass, self.mu

        # prepare the step function U_np1 = (M + dt A)^{-1}(M U_n + dt F)
        M_dt_A = (M + A * dt).with_(
            solver_options=(A.solver_options if stepper.solver_options == 'operator' else
                            M.solver_options if stepper.solver_options == 'mass' else
                            stepper.solver_options)
        )
        if not _depends_on_time(M_dt_A, mu):
            M_dt_A = M_dt_A.assemble(mu)

        if isinstance(F, ZeroOperator):
            def step_function(U_n, t_n):
                t_np1 = t_n + dt
                mu_t = mu.with_(t=t_np1)
                return M_dt_A.apply_inverse(M.apply(U_n, mu=mu_t), mu=mu_t, initial_guess=U_n), t_np1
        else:
            dt_F = F * dt
            if not _depends_on_time(dt_F, mu):
                dt_F = dt_F.assemble(mu)

            def step_function(U_n, t_n):
                t_np1 = t_n + dt
                mu_t = mu.with_(t=t_np1)
                return (
                    M_dt_A.apply_inverse(M.apply(U_n, mu=mu_t) + dt_F.as_vector(mu_t), mu=mu_t, initial_guess=U_n),
                    t_np1)

        self.step_function = step_function

    def _step_function(self, U_n, t_n):
        return self.step_function(U_n, t_n)


class ImplicitEulerTimeStepper(TimeStepper):
    """Implicit Euler :class:`TimeStepper`.

    Solves equations of the form ::

        M(mu, t) * d_t u + A(u, mu, t) = F(mu, t),
                            u(mu, t_0) = u_0(mu),

    by an implicit Euler time integration, implemented in :class:`ImplicitEulerIterator`.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate
        vector that is calculated is returned. Else an interpolation of the calculated vectors on an
        equidistant temporal grid is returned, using an appropriate interpolation of the respective
        time stepper.
    solver_options
        The |solver_options| used to invert `M + dt*A`. The special values `'mass'` and `'operator'`
        are recognized, in which case the solver_options of `M` (resp. `A`) are used.
    interpolation
        Type of temporal interpolation to be used. Currently implemented are: piecewise constant
        (P0) and piecewise linear (P1).
    """

    IteratorType = ImplicitEulerIterator

    def __init__(self, nt, num_values=None, solver_options='operator', interpolation='P1'):
        super().__init__(num_values, interpolation)
        if not num_values:
            self.num_values = nt + 1

        assert isinstance(nt, Number)
        assert nt > 0

        self.__auto_init(locals())


class ExplicitEulerIterator(SingleStepTimeStepperIterator):

    def __init__(
            self, stepper, initial_time, end_time, initial_data, operator, rhs, mass, mu, return_iter, return_times):
        super().__init__(
                stepper, initial_time, end_time, initial_data, operator, rhs, mass, mu, return_iter, return_times)
        self.dt = dt = (end_time - initial_time) / stepper.nt
        # use the ones from base, these are checked and converted in super().__init__()
        A, F, M, mu = self.operator, self.rhs, self.mass, self.mu

        # prepare the step function U_np1 = M^{-1}(M U_n + dt F - dt A U_n)
        if not _depends_on_time(M, mu):
            M = M.assemble(mu)
        if not _depends_on_time(A, mu):
            A = A.assemble(mu)

        if isinstance(F, ZeroOperator):
            if isinstance(M, IdentityOperator):
                def step_function(U_n, t_n):
                    t_np1 = t_n + dt
                    mu_t = mu.with_(t=t_n)
                    return U_n - dt*A.apply(U_n, mu=mu_t), t_np1
            else:
                def step_function(U_n, t_n):
                    t_np1 = t_n + dt
                    mu_t = mu.with_(t=t_n)
                    return (
                        M.apply_inverse(M.apply(U_n, mu=mu_t) - dt*A.apply(U_n, mu=mu_t), mu=mu_t, initial_guess=U_n),
                        t_np1)
        else:
            if not _depends_on_time(F, mu):
                F = F.assemble(mu)
            if isinstance(M, IdentityOperator):
                def step_function(U_n, t_n):
                    t_np1 = t_n + dt
                    mu_t = mu.with_(t=t_n)
                    return U_n + dt*(F.as_vector(mu_t) - A.apply(U_n, mu=mu_t)), t_np1
            else:
                def step_function(U_n, t_n):
                    t_np1 = t_n + dt
                    mu_t = mu.with_(t=t_n)
                    return (
                        M.apply_inverse(M.apply(U_n, mu=mu_t) + dt(F.as_vector(mu_t) - A.apply(U_n, mu=mu_t)),
                                        mu=mu_t, initial_guess=U_n),
                        t_np1)

        self.step_function = step_function

    def _step_function(self, U_n, t_n):
        return self.step_function(U_n, t_n)


class ExplicitEulerTimeStepper(TimeStepper):
    """Explicit Euler :class:`TimeStepper`.

    Solves equations of the form ::

        M(mu, t) * d_t u + A(u, mu, t) = F(mu, t),
                            u(mu, t_0) = u_0(mu),

    by an explicit Euler time integration, implemented in :class:`ExplicitEulerIterator`.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate
        vector that is calculated is returned. Else an interpolation of the calculated vectors on an
        equidistant temporal grid is returned, using an appropriate interpolation of the respective
        time stepper.
    interpolation
        Type of temporal interpolation to be used. Currently implemented are: piecewise constant
        (P0) and piecewise linear (P1).
    """

    IteratorType = ExplicitEulerIterator

    def __init__(self, nt, num_values=None, interpolation='P1'):
        super().__init__(num_values, interpolation)
        if not num_values:
            self.num_values = nt + 1

        assert isinstance(nt, Number)
        assert nt > 0

        self.__auto_init(locals())


class ExplicitRungeKuttaIterator(SingleStepTimeStepperIterator):

    def __init__(
            self, stepper, initial_time, end_time, initial_data, operator, rhs, mass, mu, return_iter, return_times):
        super().__init__(
                stepper, initial_time, end_time, initial_data, operator, rhs, mass, mu, return_iter, return_times)
        self.dt = (end_time - initial_time) / stepper.nt
        # use the ones from base, these are checked and converted in super().__init__()
        A, F, M, mu = self.operator, self.rhs, self.mass, self.mu

        # prepare the function f in d_t y = f(t, y)
        if not _depends_on_time(M, mu):
            M = M.assemble(mu)
        if not _depends_on_time(A, mu):
            A = A.assemble(mu)

        if isinstance(F, ZeroOperator):
            if isinstance(M, IdentityOperator):
                def f(t, y):
                    mu_t = mu.with_(t=t)
                    return -1*A.apply(y, mu=mu_t)
            else:
                def f(t, y):
                    mu_t = mu.with_(t=t)
                    return M.apply_inverse(-1*A.apply(y, mu=mu_t), mu=mu_t, initial_guess=y)
        else:
            if not _depends_on_time(F, mu):
                F = F.assemble(mu)
            if isinstance(M, IdentityOperator):
                def f(t, y):
                    mu_t = mu.with_(t=t)
                    return F.as_vector(mu=mu_t) - A.apply(y, mu=mu_t)
            else:
                def f(t, y):
                    mu_t = mu.with_(t=t) or Mu({'t': t})
                    return M.apply_inverse(F.as_vector(mu=mu_t) - A.apply(y, mu=mu_t), mu=mu_t, initial_guess=y)

        self.f = f

    def _step_function(self, U_n, t_n):
        c, A, b = self.stepper.butcher_tableau
        # compute stages
        s = len(c)
        stages = U_n.space.empty(reserve=s+1)
        stages.append(U_n)
        for j in range(s):
            t_n_j = t_n + self.dt*c[j]
            U_n_j = stages.lincomb(np.hstack(([1], self.dt*A[j][:j])))
            U_n_j = self.f(t_n_j, U_n_j)
            stages.append(U_n_j, remove_from_other=True)
        # compute step
        U_np1 = stages.lincomb(np.hstack(([1], self.dt*b)))

        return U_np1, t_n + self.dt


class ExplicitRungeKuttaTimeStepper(TimeStepper):
    """Explicit Runge-Kutta :class:`TimeStepper`.

    Solves equations of the form ::

        M(mu, t) * d_t u + A(u, mu, t) = F(mu, t).
                            u(mu, t_0) = u_0(mu),

    by a Runge-Kutta method, implemented in :class:`ExplicitRungeKuttaIterator`.

    Parameters
    ----------
    method
        Either a string identifying the method or a tuple (c, A, b) of butcher tableaus.
    nt
        The number of time-steps the time-stepper will perform.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate
        vector that is calculated is returned. Else an interpolation of the calculated vectors on an
        equidistant temporal grid is returned, using an appropriate interpolation of the respective
        time stepper.
    """

    IteratorType = ExplicitRungeKuttaIterator

    available_RK_methods = {
        'explicit_euler': (np.array([0]), np.array([[0]]), np.array([1])),
        'RK1':            (np.array([0]), np.array([[0]]), np.array([1])),
        'heun2':          (np.array([0, 1]), np.array([[0, 0], [1, 0]]), np.array([1/2, 1/2])),
        'midpoint':       (np.array([0, 1/2]), np.array([[0, 0], [1/2, 0]]), np.array([0, 1])),
        'ralston':        (np.array([0, 2/3]), np.array([[0, 0], [2/3, 0]]), np.array([1/4, 3/4])),
        'RK2':            (np.array([0, 1/2]), np.array([[0, 0], [1/2, 0]]), np.array([0, 1])),
        'simpson':        (np.array([0, 1/2, 1]),
                           np.array([[0, 0, 0], [1/2, 0, 0], [-1, 2, 0]]),
                           np.array([1/6, 4/6, 1/6])),
        'heun3':          (np.array([0, 1/3, 2/3]),
                           np.array([[0, 0, 0], [1/3, 0, 0], [0, 2/3, 0]]),
                           np.array([1/4, 0, 3/4])),
        'RK3':            (np.array([0, 1/2, 1]),
                           np.array([[0, 0, 0], [1/2, 0, 0], [-1, 2, 0]]),
                           np.array([1/6, 4/6, 1/6])),
        '3/8':            (np.array([0, 1/3, 2/3, 1]),
                           np.array([[0, 0, 0, 0], [1/3, 0, 0, 0], [-1/3, 1, 0, 0], [1, -1, 1, 0]]),
                           np.array([1/8, 3/8, 3/8, 1/8])),
        'RK4':            (np.array([0, 1/2, 1/2, 1]),
                           np.array([[0, 0, 0, 0], [1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]]),
                           np.array([1/6, 1/3, 1/3, 1/6])),
    }

    def __init__(self, method, nt, num_values=None, interpolation='P1'):
        super().__init__(num_values, interpolation)
        if not num_values:
            self.num_values = nt + 1

        assert isinstance(method, (tuple, str))
        if isinstance(method, str):
            assert method in self.available_RK_methods.keys()
            self.butcher_tableau = self.available_RK_methods[method]
        else:
            raise RuntimeError('Arbitrary butcher tableaus not implemented yet!')

        assert isinstance(nt, Number)
        assert nt > 0

        self.__auto_init(locals())


def _depends_on_time(obj, mu):
    if not mu:
        return False
    return 't' in obj.parameters or any(mu.is_time_dependent(k) for k in obj.parameters)
