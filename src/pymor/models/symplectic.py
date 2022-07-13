# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.timestepping import ImplicitMidpointTimeStepper
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import ConcatenationOperator, NumpyConversionOperator, VectorOperator
from pymor.operators.block import BlockOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyVectorSpace
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
from pymor.parameters.base import Mu
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.interface import VectorArray


class BaseQuadraticHamiltonianModel(InstationaryModel):
    """Base class of quadratic Hamiltonian systems.

    To formulate a quadratic Hamiltonian system it is advised to use a |QuadraticHamiltonianModel|
    which works with a |BlockVectorSpace| as `phase_space` to be compatible with the current
    implementation of the symplectic basis generation techniques.
    """

    def __init__(self, T, initial_data, J, H_op, h=None, time_stepper=None, nt=None, num_values=None,
                 output_functional=None, visualizer=None, name=None):

        # interface to use ImplicitMidpointTimeStepper via parameter nt
        if time_stepper is not None and nt is not None:
            # this case is required to use "with_" in combination with this model
            assert hasattr(time_stepper, 'nt') and time_stepper.nt == nt
        if time_stepper is None and nt is None:
            raise ValueError('Specify time_stepper or nt (or both)')
        if time_stepper is None:
            time_stepper = ImplicitMidpointTimeStepper(nt)

        assert (isinstance(J, Operator) and isinstance(H_op, Operator)
                and J.range == J.source == H_op.range == H_op.source)

        # minus (in J.H) is required since operator in an InstationaryModel is on the LHS
        if isinstance(H_op.range, BlockVectorSpace) and isinstance(H_op, BlockOperator):
            assert H_op.blocks.shape == (2, 2)
            assert isinstance(J, CanonicalSymplecticFormOperator)
            # compute by hand: operator = J.H * H_op
            operator = BlockOperator([
                [-H_op.blocks[1, 0], -H_op.blocks[1, 1]],
                [H_op.blocks[0, 0], H_op.blocks[0, 1]]
            ])
        else:
            operator = ConcatenationOperator([J.H, H_op])
        rhs = ConcatenationOperator([J, h])

        super().__init__(T, initial_data, operator, rhs,
                         time_stepper=time_stepper,
                         num_values=num_values,
                         output_functional=output_functional,
                         visualizer=visualizer,
                         name=name)
        self.__auto_init(locals())

    def eval_hamiltonian(self, u, mu=None):
        """Evaluate a quadratic Hamiltonian function.

        Evaluation follows the formula::

            Ham(u, t, μ) = 1/2 * u * H_op(t, μ) * u + u * h(t, μ)
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        # compute linear part
        ham_h = self.h.apply_adjoint(u, mu=mu).to_numpy().ravel()
        # compute quadratic part
        ham_H = self.H_op.pairwise_apply2(u, u, mu=mu)

        return 1/2 * ham_H + ham_h


class QuadraticHamiltonianModel(BaseQuadraticHamiltonianModel):
    """Generic class for quadratic Hamiltonian systems.

    This class describes Hamiltonian systems given by the equations::

        ∂_t u(t, μ) = J * H_op(t, μ) * u(t, μ) + J * h(t, μ)
            u(0, μ) = u_0(μ)

    for t in [0,T], where H_op is a linear time-dependent |Operator|,
    J is a canonical Poisson matrix, h is a (possibly) time-dependent
    vector-like |Operator|, and u_0 the initial data.
    The right-hand side of the Hamiltonian equation is J times the
    gradient of the Hamiltonian

        Ham(u, t, μ) = 1/2* u * H_op(t, μ) * u + u * h(t, μ).

    The `phase_space` is assumed to be a |BlockVectorSpace|. If required, the arguments `H_op`, `h`
    and the `initial_data` are casted to operate on a |BlockVectorSpace|. With this construction,
    the solution u(t, μ) is based on a |BlockVectorSpace| which is required for the current
    implementation of the symplectic basis generation techniques.

    Parameters
    ----------
    T
        The final time T.
    initial_data
        The initial data `u_0`. Either a |VectorArray| of length 1 or
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
        Alternatively, the parameter nt can be specified to use the
        :class:`implicit midpoint rule <pymor.algorithms.timestepping.ImplicitMidpointTimeStepper>`.
    nt
        If time_stepper is `None` and nt is specified, the
        :class:`implicit midpoint rule <pymor.algorithms.timestepping.ImplicitMidpointTimeStepper>`
        as time_stepper.
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

    def __init__(self, T, initial_data, H_op, h=None, time_stepper=None, nt=None, num_values=None,
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
        assert h.range == H_op.range

        if (isinstance(initial_data, VectorArray)
                and isinstance(initial_data.space, NumpyVectorSpace)):

            initial_data = H_op.source.from_numpy(initial_data.to_numpy())
        elif (isinstance(initial_data, VectorOperator)
              and isinstance(initial_data.range, NumpyVectorSpace)):

            initial_data = VectorOperator(H_op.source.from_numpy(initial_data.as_range_array().to_numpy()))

        # J based on blocked phase_space
        J = CanonicalSymplecticFormOperator(H_op.source)

        super().__init__(T, initial_data, J, H_op, h, time_stepper, nt, num_values,
                         output_functional, visualizer, name)
