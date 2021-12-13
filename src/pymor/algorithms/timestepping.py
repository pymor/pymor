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

from pymor.core.base import BasicObject, ImmutableObject, abstractmethod
from pymor.operators.constructions import IdentityOperator, VectorArrayOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu
from pymor.tools import floatcmp
from pymor.tools.deprecated import Deprecated
from pymor.vectorarrays.interface import VectorArray


class TimeStepper(ImmutableObject):
    """Interface for time-stepping algorithms.

    Algorithms implementing this interface solve time-dependent initial value problems
    of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t),
                     u(mu, t_0) = u_0(mu).

    Time-steppers used by |InstationaryModel| have to fulfill this interface. Time evolution can be performed
    by calling :meth:`solve`.

    Note that the actual work is done in an iterator derived from :class:`TimeStepperIterator`.

    Parameters
    ----------
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate vector that is
        calculated is returned. Else an interpolation of the calculated vectors on an equidistant temporal grid is
        returned, using an appropriate interpolation of the respective time stepper.
    interpolation
        Type of temporal interpolation to be used. Currently implemented are: piecewise constant (P0) and piecewise
        linear (P1).
    """

    IteratorType = None
    available_interpolations = ('P0', 'P1')

    def __init__(self, num_values=None, interpolation='P1'):
        assert not num_values or (isinstance(num_values, Number) and num_values > 1)
        assert interpolation in self.available_interpolations
        self.__auto_init(locals())

    def solve(self, t0, t1, U0, A, F=None, M=None, mu=None, return_iter=False, return_times=False):
        """Apply time-stepper to the equation ::

            M * d_t u + A(u, mu, t) = F(mu, t),
                         u(mu, t_0) = u_0(mu).

        Parameters
        ----------
        t0
            The time at which to begin time-stepping.
        t1
            The time until which to perform time-stepping.
        U0
            The solution vector at `t0`.
        A
            The |Operator| A.
        F
            The right-hand side F (either |VectorArray| of length 1 or |Operator| with
            `source.dim == 1`). If `None`, zero right-hand side is assumed.
        M
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
            If `return_iter` is `False` and `return_times` is `False` (the default), where `U` is a |VectorArray|
            containing the solution trajectory.
        (U, t)
            If `return_iter` is `False` and `return_times` is `True`, where `t` is the list of time points corresponding to
            the solution vectors in `U`.
        iterator
            If `return_iter` is `True`, an iterator yielding either `U_n` (if `return_times` is `False`) or `(U_n, t_n)` in
            each step.
        """

        # all checks are delegated to TimeStepperIterator
        iterator = self.IteratorType(
                self, t0, t1, U0, A, F=F, M=M, mu=mu, return_iter=return_iter, return_times=return_times)
        if return_iter:
            return iterator
        elif return_times:
            U = A.source.empty(reserve=self.num_values or 0)
            t = []
            for U_n, t_n in iterator:
                U.append(U_n, remove_from_other=True)
                t.append(t_n)
            return U, t
        else:
            U = A.source.empty(reserve=self.num_values or 0)
            for U_n in iterator:
                U.append(U_n)
            return U


class TimeStepperIterator(BasicObject):
    """Base class to derive time-stepper iterators from.

    See :meth:`TimeStepper.solve` for a documentation of the init parameters except `stepper`.

    Note: derived classes usually only have to implement :meth:`_step`, and optionally :meth:`_interpolate` if none of
          the provided interpolations are suitable. Using the interpolation from this base class requires the
          implementor to store certain data in each step (see :meth:`_interpolate`).

    Parameters
    ----------
    stepper
        The associated :class:`TimeStepper`.
    """

    def __init__(self, stepper, t0, t1, U0, A, F, M, mu, return_iter, return_times):
        # check input
        assert isinstance(stepper, TimeStepper)

        assert isinstance(t0, Number)
        assert isinstance(t1, Number)
        assert t1 > t0

        assert isinstance(A, Operator)

        F = F if F is not None else ZeroOperator(A.source, NumpyVectorSpace(1))
        assert isinstance(F, (Operator, VectorArray))
        if isinstance(F, Operator):
            assert F.source.dim == 1
            assert F.range == A.range
        else:
            assert F in A.range
            assert len(F) == 1
            F = VectorArrayOperator(F)
        assert A.range == F.range

        M = M or IdentityOperator(A.source)
        assert isinstance(M, Operator)
        assert A.range == M.range

        if isinstance(U0, Operator):
            assert U0.source.dim == 1
            assert U0.range == A.source
        else:
            assert U0 in A.source and len(U0) == 1
            U0 = VectorOperator(U0)

        self.__auto_init(locals())

        self.t = t0
        self.mu = mu or Mu()
        self.U0 = U0.as_vector(self.mu.with_(t=t0))

        # prepare interpolation
        if stepper.num_values:
            self._interpolation_points_increment = (t1 - t0) / (stepper.num_values - 1)
            self._last_stepped_point = t0 - 1
            self._next_interpolation_point = t0

    @abstractmethod
    def _step(self):
        """Called in :meth:`_interpolated_step` to compute the next step of the time evolution.

        The iterator is assumed to be in the n-th time-step, `self.t == t_n`, the current state of the solution is
        often available as `self.U_n` (depending on the interpolation and the choice of the implementor).

        Returns
        -------
        U_np1
            A |VectorArray| of length 1 containing U(t_{n + 1}).
        t_np1
            The next time instance t_{n + 1} = t_n + dt, where dt has to be determined by the implementor.
        """
        pass

    def _interpolate(self, t):
        """Called in :meth:`_interpolated_step` to compute an interpolated value of the solution.

        If not overridden in derived classes, requires the following data to be present after a call to :meth:`_step`:
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
            # return previous value, old default in pyMOR
            return self.U_n.copy()
        elif self.stepper.interpolation == 'P1':
            # compute P1-Lagrange interpolation
            U_n, U_np1 = self.U_n, self.U_np1
            # but return node values if t is close enough
            if floatcmp.float_cmp(t, t_n):
                return U_n
            elif floatcmp.float_cmp(t, t_np1):
                return U_np1
            else:
                dt = t_np1 - t_n
                return (t_np1 - t)/dt * U_n + (t - t_n)/dt * U_np1
        else:
            raise NotImplementedError(f'{self.interpolation}-interpolation not available!')

    def _interpolated_step(self):
        """
        Returns `(U_next, t_next)` (if `return_times == True`, otherwise `U_next`), behavior depends on `num_values`:
        If `num_values` is provided, performs as many actual steps (by calling :meth:`_step`) as required to obtain
        an interpolation of the solution, U(t_next) at the next required interpolation point t_next >= t_n, and
        returns (U(t_next), t_next). If `num_values` is not provided, performs a single step (by calling :meth:`_step`)
        starting from t_n, to compute (U(t_{n + 1}), t_{n + 1}).
        """
        if floatcmp.almost_less(self.t1, self.t):
            # this is the end
            raise StopIteration
        elif self.stepper.num_values:
            # an interpolation of the trajectory is requested
            if self._last_stepped_point < self.t0:
                # this is the start, take a step to have data and interpolation available next time, but return U0
                _, self.t = self._step()
                self._last_stepped_point = self.t
                self._next_interpolation_point += self._interpolation_points_increment
                if self.return_times:
                    return self.U0, self.t0
                else:
                    return self.U0
            elif self._last_stepped_point > self._next_interpolation_point:
                # we have enough data, simply interpolate
                if not self.logging_disabled:
                    self.logger.debug(f't={self._next_interpolation_point}: interpolating ...')
                t_next = self._next_interpolation_point
                U_next = self._interpolate(t_next)
                self._next_interpolation_point += self._interpolation_points_increment
                if self.return_times:
                    return U_next, t_next
                else:
                    return U_next
            else:
                # we do not have enough data, take enough actual steps
                while self.t < self._next_interpolation_point:
                    _, self.t = self._step()
                    self._last_stepped_point = self.t
                # compute the interpolation at the next requested point
                if not self.logging_disabled:
                    self.logger.debug(f't={self._next_interpolation_point}: interpolating ...')
                t_next = self._next_interpolation_point
                U_next = self._interpolate(t_next)
                self._next_interpolation_point += self._interpolation_points_increment
                if self.return_times:
                    return U_next, t_next
                else:
                    return U_next
        else:
            # the trajectory is requested as is
            U, self.t = self._step()
            if self.return_times:
                return U, t
            else:
                return U

    def __iter__(self):
        return self

    def __next__(self):
        return self._interpolated_step()


class SingleStepTimeStepperIterator(TimeStepperIterator):
    """Base class for iterators of single-step methods.

    See :meth:`TimeStepperIterator` for a documentation of the init parameters.

    Note: derived classes only have to implement :meth:`_step_function`, and optionally :meth:`_interpolate`
    """

    @abstractmethod
    def _step_function(self, U_n, t_n):
        pass

    def _step(self):
        if not hasattr(self, 'U_n'):
            if not self.logging_disabled:
                self.logger.debug(f't={self.t}: returning initial data (mu={self.mu}) ...')
            self.t_n = self.t0   # setting these just in case ...
            self.U_n = self.U0   # ... for _interpolate
            self.t_np1 = self.t0
            self.U_np1 = self.U0.copy()
            return self.U_np1, self.t_np1
        else:
            # this is the first actual step or a usual step
            self.t_n = self.t_np1
            self.U_n = self.U_np1.copy()
            if not self.logging_disabled:
                self.logger.debug(f't={self.t}: stepping (mu={self.mu}) ...')
            self.U_np1, self.t_np1 = self._step_function(self.U_n, self.t_n)
            return self.U_np1, self.t_np1


class ImplicitEulerIterator(SingleStepTimeStepperIterator):

    def __init__(self, stepper, t0, t1, U0, A, F, M, mu, return_iter, return_times):
        super().__init__(stepper, t0, t1, U0, A, F, M, mu, return_iter, return_times)
        self.dt = dt = (t1 - t0) / stepper.nt
        # use the ones from base, these are checked and converted in super().__init__()
        A, F, M, mu = self.A, self.F, self.M, self.mu

        # prepare the step function U_np1 = (M + dt A)^{-1}(M U_n + dt F)
        M_dt_A = (M + A * dt).with_(
                solver_options=A.solver_options if stepper.solver_options == 'operator' else \
                               M.solver_options if stepper.solver_options == 'mass' else \
                               stepper.solver_options)
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

        M * d_t u + A(u, mu, t) = F(mu, t),
                     u(mu, t_0) = u_0(mu),

    by an implicit Euler time integration, implemented in :class:`ImplicitEulerIterator`.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate vector that is
        calculated is returned. Else an interpolation of the calculated vectors on an equidistant temporal grid is
        returned, using an appropriate interpolation of the respective time stepper.
    solver_options
        The |solver_options| used to invert `M + dt*A`. The special values `'mass'` and `'operator'` are recognized,
        in which case the solver_options of `M` (resp. `A`) are used.
    interpolation
        Type of temporal interpolation to be used. Currently implemented are: piecewise constant (P0) and piecewise
        linear (P1).
    """

    IteratorType = ImplicitEulerIterator

    def __init__(self, nt, num_values=None, solver_options='operator', interpolation='P1'):
        super().__init__(num_values, interpolation)

        assert isinstance(nt, Number)
        assert nt > 0

        self.__auto_init(locals())


class ExplicitEulerIterator(SingleStepTimeStepperIterator):

    def __init__(self, stepper, t0, t1, U0, A, F, M, mu, return_iter, return_times):
        super().__init__(stepper, t0, t1, U0, A, F, M, mu, return_iter, return_times)
        self.dt = dt = (t1 - t0) / stepper.nt
        # use the ones from base, these are checked and converted in super().__init__()
        A, F, M, mu = self.A, self.F, self.M, self.mu

        # prepare the step function U_np1 = M^{-1}(M U_n + dt F - dt A U_n)
        if not 't' in M.parameters:
            M = M.assemble(mu)
        if not 't' in A.parameters:
            A = A.assemble(mu)

        if isinstance(F, ZeroOperator):
            if isinstance(M, IdentityOperator):
                def step_function(U_n, t_n):
                    t_np1 = t_n + dt
                    mu_t = mu.with_(t=t_np1)
                    return U_n - dt*A.apply(U_n, mu=mu_t), t_np1
            else:
                def step_function(U_n, t_n):
                    t_np1 = t_n + dt
                    mu_t = mu.with_(t=t_np1)
                    return (
                        M.apply_inverse(M.apply(U_n, mu=mu_t) - dt*A.apply(U_n, mu=mu_t), mu=mu_t, initial_guess=U_n),
                        t_np1)
        else:
            if not F.parametric or 't' not in F.parameters:
                F = F.assemble(mu)
            if isinstance(M, IdentityOperator):
                def step_function(U_n, t_n):
                    t_np1 = t_n + dt
                    mu_t = mu.with_(t=t_np1)
                    return U_n + dt*(F.as_vector(mu_t) - A.apply(U_n, mu=mu_t)), t_np1
            else:
                def step_function(U_n, t_n):
                    t_np1 = t_n + dt
                    mu_t = mu.with_(t=t_np1)
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

        M * d_t u + A(u, mu, t) = F(mu, t),
                     u(mu, t_0) = u_0(mu),

    by an explicit Euler time integration, implemented in :class:`ExplicitEulerIterator`.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate vector that is
        calculated is returned. Else an interpolation of the calculated vectors on an equidistant temporal grid is
        returned, using an appropriate interpolation of the respective time stepper.
    interpolation
        Type of temporal interpolation to be used. Currently implemented are: piecewise constant (P0) and piecewise
        linear (P1).
    """

    IteratorType = ExplicitEulerIterator

    def __init__(self, nt, num_values=None, interpolation='P1'):
        super().__init__(num_values, interpolation)

        assert isinstance(nt, Number)
        assert nt > 0

        self.__auto_init(locals())


class ExplicitRungeKuttaIterator(SingleStepTimeStepperIterator):

    def __init__(self, stepper, t0, t1, U0, A, F, M, mu, return_iter, return_times):
        super().__init__(stepper, t0, t1, U0, A, F, M, mu, return_iter, return_times)
        self.dt = dt = (t1 - t0) / stepper.nt
        # use the ones from base, these are checked and converted in super().__init__()
        A, F, M, mu = self.A, self.F, self.M, self.mu

        # prepare the function f in d_t y = f(t, y)
        if not isinstance(M, IdentityOperator) and not _depends_on_time(M, mu):
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
        stages = U_n.space.empty(reserve=s)
        for j in range(s):
            t_n_j = t_n + self.dt*c[j]
            U_n_j = U_n.copy()
            for l in range(j):
                U_n_j.axpy(self.dt*A[j][l], stages[l])
            U_n_j = self.f(t_n_j, U_n_j)
            stages.append(U_n_j, remove_from_other=True)
        # compute step
        U_np1 = U_n.copy()
        for j in range(s):
            U_np1.axpy(self.dt*b[j], stages[j])

        return U_np1, t_n + self.dt


class ExplicitRungeKuttaTimeStepper(TimeStepper):
    """Explicit Runge-Kutta :class:`TimeStepper`.

    Solves equations of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t).
                     u(mu, t_0) = u_0(mu),

    by a Runge-Kutta method, implemented in :class:`ExplicitRungeKuttaIterator`.

    Parameters
    ----------
    method
        Either a string identifying the method or a tuple (c, A, b) of butcher tableaus.
    nt
        The number of time-steps the time-stepper will perform.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each intermediate vector that is
        calculated is returned. Else an interpolation of the calculated vectors on an equidistant temporal grid is
        returned, using an appropriate interpolation of the respective time stepper.
    """

    IteratorType = ExplicitRungeKuttaIterator

    available_RK_methods = {
            'explicit_euler': (np.array([0,]), np.array([[0,],]), np.array([1,])),
            'RK1':            (np.array([0,]), np.array([[0,],]), np.array([1,])),
            'heun2':          (np.array([0, 1]), np.array([[0, 0], [1, 0]]), np.array([1/2, 1/2])),
            'midpoint' :      (np.array([0, 1/2]), np.array([[0, 0], [1/2, 0]]), np.array([0, 1])),
            'ralston' :       (np.array([0, 2/3]), np.array([[0, 0], [2/3, 0]]), np.array([1/4, 3/4])),
            'RK2' :           (np.array([0, 1/2]), np.array([[0, 0], [1/2, 0]]), np.array([0, 1])),
            'simpson' :       (np.array([0, 1/2, 1]),
                               np.array([[0, 0, 0], [1/2, 0, 0], [-1, 2, 0]]),
                               np.array([1/6, 4/6, 1/6])),
            'heun3' :         (np.array([0, 1/3, 2/3]),
                               np.array([[0, 0, 0], [1/3, 0, 0], [0, 2/3, 0]]),
                               np.array([1/4, 0, 3/4])),
            'RK3' :           (np.array([0, 1/2, 1]),
                               np.array([[0, 0, 0], [1/2, 0, 0], [-1, 2, 0]]),
                               np.array([1/6, 4/6, 1/6])),
            '3/8' :           (np.array([0, 1/3, 2/3, 1]),
                               np.array([[0, 0, 0, 0], [1/3, 0, 0, 0], [-1/3, 1, 0, 0], [1, -1, 1, 0]]),
                               np.array([1/8, 3/8, 3/8, 1/8])),
            'RK4' :           (np.array([0, 1/2, 1/2, 1]),
                               np.array([[0, 0, 0, 0], [1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]]),
                               np.array([1/6, 1/3, 1/3, 1/6])),
            }

    def __init__(self, method, nt, num_values=None, interpolation='P1'):
        super().__init__(num_values, interpolation)

        assert isinstance(method, (tuple, str))
        if isinstance(method, str):
            assert method in self.available_RK_methods.keys()
            self.butcher_tableau = (self.available_RK_methods[method])
        else:
            raise RuntimeError('Arbitrary butcher arrays not implemented yet!')

        assert isinstance(nt, Number)
        assert nt > 0

        self.__auto_init(locals())


def _depends_on_time(obj, mu):
    if not mu:
        return False
    return 't' in obj.parameters or any(mu.is_time_dependent(k) for k in obj.parameters)


@Deprecated('Will be removed after the 2021.2 release, use ImplicitEulerTimeStepper directly.')
def implicit_euler(A, F, M, U0, t0, t1, nt, mu=None, num_values=None, solver_options='operator'):
    time_stepper = ImplicitEulerTimeStepper(nt=nt, initial_time=t0, end_time=t1, num_values=num_values,
                                            solver_options=solver_options, interpolation_order=0)
    return time_stepper.solve(mu=mu)


@Deprecated('Will be removed after the 2021.2 release, use ExplicitEulerTimeStepper directly.')
def explicit_euler(A, F, U0, t0, t1, nt, mu=None, num_values=None):
    time_stepper = ExplicitEulerTimeStepper(
            nt=nt, initial_time=t0, end_time=t1, num_values=num_values, interpolation_order=0)
    return time_stepper.solve(mu=mu)
