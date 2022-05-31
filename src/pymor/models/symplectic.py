# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from pymor.algorithms.simplify import contract, expand
from pymor.algorithms.to_matrix import to_matrix
from pymor.models.basic import InstationaryModel
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import ConcatenationOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator, NumpyVectorSpace
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
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

        if h is None:
            h = H_op.range.zeros()

        if isinstance(h, VectorArray):
            assert h in H_op.range
            h = VectorOperator(h, name='h')

        # generate correct canonical J
        if isinstance(H_op, BlockOperator):
            self.J = CanonicalSymplecticFormOperator(H_op.source)
        elif isinstance(H_op.range, NumpyVectorSpace):
            assert H_op.range.dim % 2 == 0, 'H_op.range has to be even dimensional'
            half_dim = H_op.range.dim // 2
            phase_space = BlockVectorSpace([NumpyVectorSpace(half_dim)] * 2)
            J = CanonicalSymplecticFormOperator(phase_space)
            self.J = NumpyMatrixOperator(to_matrix(J).toarray())
        else:
            raise NotImplementedError('Canonical Poisson matrix not implemented for this case.')

        # the contract expand structure is mainly relevant for reduced quadratic Hamiltonian systems
        operator = contract(expand(ConcatenationOperator([self.J, H_op])))
        rhs = ConcatenationOperator([self.J, h])
        super().__init__(T, initial_data, operator, rhs,
                         time_stepper=time_stepper,
                         num_values=num_values,
                         output_functional=output_functional,
                         visualizer=visualizer,
                         name=name)
        self.__auto_init(locals())

    def eval_hamiltonian(self, u, mu=None):
        """Evaluate a quadratic Hamiltonian function

        Ham(u, t, μ) = 1/2 * u * H_op(t, μ) * u + u * h(t, μ).
        """
        # compute linear part
        ham_h = self.h.apply_adjoint(u, mu=mu)
        # compute quadratic part
        ham_H = ham_h.space.make_array(self.H_op.pairwise_apply2(u, u, mu=mu)[:, np.newaxis])

        return 1/2 * ham_H + ham_h
