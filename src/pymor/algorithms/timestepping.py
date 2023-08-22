# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Generic time-stepping algorithms for the solution of instationary problems.

The algorithms are generic in the sense that each algorithms operates exclusively
on |Operators| and |VectorArrays|. In particular, the algorithms
can also be used to turn an arbitrary stationary |Model| provided
by an external library into an instationary |Model|.

The :class:`TimeStepper` defines a common interface that has to be fulfilled by
the time-steppers used by |InstationaryModel|.
"""

from pymor.core.base import ImmutableObject, abstractmethod
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu
from pymor.vectorarrays.interface import VectorArray


class TimeStepper(ImmutableObject):
    """Interface for time-stepping algorithms.

    Algorithms implementing this interface solve time-dependent initial value problems
    of the form ::

        M(mu) * d_t u + A(u, mu, t) = F(mu, t),
                         u(mu, t_0) = u_0(mu).

    Time-steppers used by |InstationaryModel| have to fulfill
    this interface.
    """

    def estimate_time_step_count(self, initial_time, end_time):
        """Estimate the number of time steps.

        Parameters
        ----------
        initial_time
            The time at which to begin time-stepping.
        end_time
            The time until which to perform time-stepping.
        """
        raise NotImplementedError

    def solve(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        """Apply time-stepper to the equation.

        The equation is of the form ::

            M(mu) * d_t u + A(u, mu, t) = F(mu, t),
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
        num_values
            The number of returned vectors of the solution trajectory. If `None`, each
            intermediate vector that is calculated is returned.

        Returns
        -------
        |VectorArray| containing the solution trajectory.
        """
        try:
            num_time_steps = self.estimate_time_step_count(initial_time, end_time)
        except NotImplementedError:
            num_time_steps = 0
        iterator = self.iterate(initial_time, end_time, initial_data, operator, rhs=rhs, mass=mass, mu=mu,
                                num_values=num_values)
        U = operator.source.empty(reserve=num_values if num_values else num_time_steps + 1)
        for U_n, _ in iterator:
            U.append(U_n)
        return U

    @abstractmethod
    def iterate(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        """Iterate time-stepper to the equation.

        The equation is of the form ::

            M(mu) * d_t u + A(u, mu, t) = F(mu, t),
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
        num_values
            The number of returned vectors of the solution trajectory. If `None`, each
            intermediate vector that is calculated is returned.

        Returns
        -------
        Generator yielding tuples (U, t) of snapshots and times.
        """
        pass


class ImplicitEulerTimeStepper(TimeStepper):
    """Implicit Euler time-stepper.

    Solves equations of the form ::

        M(mu) * d_t u + A(u, mu, t) = F(mu, t),
                         u(mu, t_0) = u_0(mu).

    by implicit Euler time integration.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    solver_options
        The |solver_options| used to invert `M + dt*A`.
        The special values `'mass'` and `'operator'` are
        recognized, in which case the solver_options of
        M (resp. A) are used.
    """

    def __init__(self, nt, solver_options='operator'):
        self.__auto_init(locals())

    def estimate_time_step_count(self, initial_time, end_time):
        return self.nt

    def iterate(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        A, F, M, U0, t0, t1, nt = operator, rhs, mass, initial_data, initial_time, end_time, self.nt
        assert isinstance(A, Operator)
        assert isinstance(F, (type(None), Operator, VectorArray))
        assert isinstance(M, (type(None), Operator))
        assert A.source == A.range
        num_values = num_values or nt + 1
        dt = (t1 - t0) / nt
        DT = (t1 - t0) / (num_values - 1)

        if F is None:
            F_time_dep = False
        elif isinstance(F, Operator):
            assert F.source.dim == 1
            assert F.range == A.range
            F_time_dep = _depends_on_time(F, mu)
            if not F_time_dep:
                dt_F = F.as_vector(mu) * dt
        else:
            assert len(F) == 1
            assert F in A.range
            F_time_dep = False
            dt_F = F * dt

        if M is None:
            from pymor.operators.constructions import IdentityOperator
            M = IdentityOperator(A.source)

        assert A.source == M.source == M.range
        assert not M.parametric
        assert U0 in A.source
        assert len(U0) == 1

        num_ret_values = 1
        yield U0, t0

        options = (A.solver_options if self.solver_options == 'operator' else
                   M.solver_options if self.solver_options == 'mass' else
                   self.solver_options)
        M_dt_A = (M + A * dt).with_(solver_options=options)
        if not _depends_on_time(M_dt_A, mu):
            M_dt_A = M_dt_A.assemble(mu)

        t = t0
        U = U0.copy()
        if mu is None:
            mu = Mu()

        for n in range(nt):
            t += dt
            mu = mu.with_(t=t)
            rhs = M.apply(U)
            if F_time_dep:
                dt_F = F.as_vector(mu) * dt
            if F:
                rhs += dt_F
            U = M_dt_A.apply_inverse(rhs, mu=mu, initial_guess=U)
            while t - t0 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, t


class ExplicitEulerTimeStepper(TimeStepper):
    """Explicit Euler time-stepper.

    Solves equations of the form ::

        M(mu) * d_t u + A(u, mu, t) = F(mu, t),
                         u(mu, t_0) = u_0(mu).

    by explicit Euler time integration.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    """

    def __init__(self, nt):
        self.__auto_init(locals())

    def estimate_time_step_count(self, initial_time, end_time):
        return self.nt

    def iterate(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        if mass is not None:
            raise NotImplementedError
        A, F, U0, t0, t1, nt = operator, rhs, initial_data, initial_time, end_time, self.nt
        assert isinstance(A, Operator)
        assert F is None or isinstance(F, (Operator, VectorArray))
        assert A.source == A.range
        num_values = num_values or nt + 1

        if isinstance(F, Operator):
            assert F.source.dim == 1
            assert F.range == A.range
            F_time_dep = _depends_on_time(F, mu)
            if not F_time_dep:
                F_ass = F.as_vector(mu)
        elif isinstance(F, VectorArray):
            assert len(F) == 1
            assert F in A.range
            F_time_dep = False
            F_ass = F

        assert len(U0) == 1
        assert U0 in A.source

        A_time_dep = _depends_on_time(A, mu)
        if not A_time_dep:
            A = A.assemble(mu)

        dt = (t1 - t0) / nt
        DT = (t1 - t0) / (num_values - 1)
        num_ret_values = 1
        yield U0, t0

        t = t0
        U = U0.copy()
        if mu is None:
            mu = Mu()

        if F is None:
            for n in range(nt):
                t += dt
                mu = mu.with_(t=t)
                U.axpy(-dt, A.apply(U, mu=mu))
                while t - t0 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                    num_ret_values += 1
                    yield U, t
        else:
            for n in range(nt):
                t += dt
                mu = mu.with_(t=t)
                if F_time_dep:
                    F_ass = F.as_vector(mu)
                U.axpy(dt, F_ass - A.apply(U, mu=mu))
                while t - t0 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                    num_ret_values += 1
                    yield U, t


class ImplicitMidpointTimeStepper(TimeStepper):
    """Implicit midpoint rule time-stepper. Symplectic integrator + preserves quadratic invariants.

    Solves equations of the form ::

        M(mu) * d_t u + A(u, mu, t) = F(mu, t),
                         u(mu, t_0) = u_0(mu).

    by implicit midpoint time integration.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    solver_options
        The |solver_options| used to invert `M - dt/2*A`.
        The special values `'mass'` and `'operator'` are
        recognized, in which case the solver_options of
        M (resp. A) are used.
    """

    def __init__(self, nt, solver_options='operator'):
        self.__auto_init(locals())

    def estimate_time_step_count(self, initial_time, end_time):
        return self.nt

    def iterate(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        if not operator.linear:
            raise NotImplementedError
        A, F, M, U0, t0, t1, nt = operator, rhs, mass, initial_data, initial_time, end_time, self.nt
        assert isinstance(A, Operator)
        assert isinstance(F, (type(None), Operator, VectorArray))
        assert isinstance(M, (type(None), Operator))
        assert A.source == A.range
        num_values = num_values or nt + 1
        dt = (t1 - t0) / nt
        DT = (t1 - t0) / (num_values - 1)

        if F is None:
            F_time_dep = False
        elif isinstance(F, Operator):
            assert F.source.dim == 1
            assert F.range == A.range
            F_time_dep = _depends_on_time(F, mu)
            if not F_time_dep:
                dt_F = F.as_vector(mu) * dt
        else:
            assert len(F) == 1
            assert F in A.range
            F_time_dep = False
            dt_F = F * dt

        if M is None:
            from pymor.operators.constructions import IdentityOperator
            M = IdentityOperator(A.source)

        assert A.source == M.source == M.range
        assert not M.parametric
        assert U0 in A.source
        assert len(U0) == 1

        num_ret_values = 1
        yield U0, t0

        if self.solver_options == 'operator':
            options = A.solver_options
        elif self.solver_options == 'mass':
            options = M.solver_options
        else:
            options = self.solver_options

        M_dt_A_impl = (M + A * (dt/2)).with_(solver_options=options)
        if not _depends_on_time(M_dt_A_impl, mu):
            M_dt_A_impl = M_dt_A_impl.assemble(mu)
        M_dt_A_expl = (M - A * (dt/2)).with_(solver_options=options)
        if not _depends_on_time(M_dt_A_expl, mu):
            M_dt_A_expl = M_dt_A_expl.assemble(mu)

        t = t0
        U = U0.copy()
        if mu is None:
            mu = Mu()

        for n in range(nt):
            mu = mu.with_(t=t + dt/2)
            t += dt
            rhs = M_dt_A_expl.apply(U, mu=mu)
            if F_time_dep:
                dt_F = F.as_vector(mu) * dt
            if F:
                rhs += dt_F
            U = M_dt_A_impl.apply_inverse(rhs, mu=mu)
            while t - t0 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, t


class DiscreteTimeStepper(TimeStepper):
    """Discrete time-stepper.

    Solves equations of the form ::

        M(mu) * u_k+1 + A(u_k, mu, k) = F(mu, k).
                           u(mu, k_0) = u_0(mu).

    by direct time stepping.
    """

    def __init__(self):
        pass

    def estimate_time_step_count(self, initial_time, end_time):
        return end_time - initial_time

    def iterate(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        A, F, M, U0, k0, k1 = operator, rhs, mass, initial_data, initial_time, end_time
        assert isinstance(A, Operator)
        assert isinstance(F, (type(None), Operator, VectorArray))
        assert isinstance(M, (type(None), Operator))
        assert A.source == A.range
        nt = k1 - k0
        num_values = num_values or nt + 1
        dt = 1
        DT = nt / (num_values - 1)

        if F is None:
            F_time_dep = False
        elif isinstance(F, Operator):
            assert F.source.dim == 1
            assert F.range == A.range
            F_time_dep = _depends_on_time(F, mu)
            if not F_time_dep:
                Fk = F.as_vector(mu)
        else:
            assert len(F) == 1
            assert F in A.range
            F_time_dep = False
            Fk = F

        if M is None:
            from pymor.operators.constructions import IdentityOperator
            M = IdentityOperator(A.source)

        assert A.source == M.source == M.range
        assert U0 in A.source
        assert len(U0) == 1

        num_ret_values = 1
        yield U0, k0

        if not _depends_on_time(M, mu):
            M = M.assemble(mu)

        U = U0.copy()
        if mu is None:
            mu = Mu()

        for k in range(k0, k0 + nt):
            mu = mu.with_(t=k)
            rhs = -A.apply(U, mu=mu)
            if F_time_dep:
                Fk = F.as_vector(mu)
            if F:
                rhs += Fk
            U = M.apply_inverse(rhs, mu=mu, initial_guess=U)
            while k - k0 + 1 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, k


def _depends_on_time(obj, mu):
    if not mu:
        return False
    return 't' in obj.parameters or any(mu.is_time_dependent(k) for k in obj.parameters)
