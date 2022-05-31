# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from pymor.algorithms.timestepping import ImplicitMidpointTimeStepper
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import ConcatenationOperator, NumpyConversionOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyVectorSpace
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
from pymor.parameters.base import Mu
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.interface import VectorArray


class QuadraticHamiltonianModel(InstationaryModel):
    """Generic class for quadratic Hamiltonian systems.

    This class describes Hamiltonian systems given by the equations::

        ∂_t u(t, μ) = J * H_op(t, μ) * u(t, μ) + J * h(t, μ)
            u(0, μ) = x_0(μ)

    for t in [0,T], where H_op is a linear time-dependent |Operator|,
    J is a canonical Poisson matrix, h is a (possibly) time-dependent
    vector-like |Operator|, and x_0 the initial data.
    The right-hand side of the Hamiltonian equation is J times the
    gradient of the Hamiltonian

        Ham(u, t, μ) = 1/2* u * H_op(t, μ) * u + u * h(t, μ)

    Parameters
    ----------
    T
        The final time T.
    initial_data
        The initial data `x_0`. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`) which
        applied to `NumpyVectorArray(np.array([1]))` will yield the
        initial data for a given |Parameter|.
    H_op
        The |Operator| H_op.
    h
        The state-independet part of the Hamiltonian h.
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
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    """

    def __init__(self, T, initial_data, H_op, h=None, time_stepper=None, num_values=None,
                 output_functional=None, visualizer=None, name=None):
        assert isinstance(H_op, Operator) and H_op.linear and H_op.range == H_op.source \
               and H_op.range.dim % 2 == 0

        if isinstance(H_op.range, NumpyVectorSpace):
            # make H_op compatible with blocked phase_space
            assert H_op.range.dim % 2 == 0, 'H_op.range has to be even dimensional'
            half_dim = H_op.range.dim // 2
            phase_space = BlockVectorSpace([NumpyVectorSpace(half_dim)] * 2)
            H_op = ConcatenationOperator([
                NumpyConversionOperator(phase_space, direction='from_numpy'),
                H_op,
                NumpyConversionOperator(phase_space, direction='to_numpy'),
            ])

        if h is None:
            h = H_op.range.zeros()

        if isinstance(h, VectorArray):
            h = VectorOperator(h, name='h')

        if isinstance(h.range, NumpyVectorSpace):
            # make h compatible with blocked phase_space
            assert h.range.dim % 2 == 0, 'h.range has to be even dimensional'
            half_dim = h.range.dim // 2
            phase_space = H_op.range
            h = ConcatenationOperator([
                NumpyConversionOperator(phase_space, direction='from_numpy'),
                h,
            ])
        assert h.range is H_op.range

        if isinstance(initial_data.space, NumpyVectorSpace):
            # make initial_data compatible with blocked phase_space
            initial_data = H_op.source.from_numpy(initial_data.to_numpy())

        # J based on blocked phase_space
        self.J = CanonicalSymplecticFormOperator(H_op.source)

        # the contract expand structure is mainly relevant for reduced quadratic Hamiltonian systems
        # minus is required since operator in an InstationaryModel is on the LHS
        operator = -ConcatenationOperator([self.J, H_op])
        rhs = ConcatenationOperator([self.J, h])
        super().__init__(T, initial_data, operator, rhs,
                         time_stepper=time_stepper,
                         num_values=num_values,
                         output_functional=output_functional,
                         visualizer=visualizer,
                         name=name)
        self.__auto_init(locals())

    @classmethod
    def with_implicit_midpoint(cls, T, initial_data, H_op, nt, h=None, time_stepper=None,
                               num_values=None, output_functional=None, visualizer=None, name=None):
        """Init QuadraticHamiltonianModel with ImplicitMidpointTimeStepper.

        Parameters
        ----------
        nt
            Number of time steps in ImplicitMidpointTimeStepper.
        others
            See __init__.
        """
        time_stepper = ImplicitMidpointTimeStepper(nt)
        cls(T, initial_data, H_op, nt, h=h, time_stepper=time_stepper, num_values=num_values,
            output_functional=output_functional, visualizer=visualizer, name=name)

    def eval_hamiltonian(self, u, mu=None):
        """Evaluate a quadratic Hamiltonian function

        Ham(u, t, μ) = 1/2 * u * H_op(t, μ) * u + u * h(t, μ).
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        # compute linear part
        ham_h = self.h.apply_adjoint(u, mu=mu)
        # compute quadratic part
        ham_H = ham_h.space.make_array(self.H_op.pairwise_apply2(u, u, mu=mu)[:, np.newaxis])

        return 1/2 * ham_H + ham_h
