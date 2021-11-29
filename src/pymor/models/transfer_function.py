# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.models.iosys import InputOutputModel
from pymor.parameters.base import Mu


class TransferFunction(InputOutputModel):
    """Class for systems represented by a transfer function.

    This class describes input-output systems given by a transfer
    function :math:`H(s, mu)`.

    Parameters
    ----------
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
    tf
        The transfer function defined at least on the open right complex half-plane.
        `tf(s, mu)` is a |NumPy array| of shape `(p, m)`.
    dtf
        The complex derivative of `H` with respect to `s` (optional).
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    name
        Name of the system.

    Attributes
    ----------
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
    tf
        The transfer function.
    dtf
        The complex derivative of the transfer function.
    """

    def __init__(self, dim_input, dim_output, tf, dtf=None, parameters={}, cont_time=True, name=None):
        super().__init__(dim_input, dim_output, cont_time=cont_time, name=name)
        self.parameters_own = parameters
        self.__auto_init(locals())

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of inputs:  {self.dim_input}\n'
            f'    number of outputs: {self.dim_output}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )

    def eval_tf(self, s, mu=None):
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        if not self.parametric:
            return self.tf(s)
        else:
            return self.tf(s, mu=mu)

    def eval_dtf(self, s, mu=None):
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        if not self.parametric:
            return self.dtf(s)
        else:
            return self.dtf(s, mu=mu)

    def __add__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.dim_input == other.dim_input
        assert self.dim_output == other.dim_output

        tf = lambda s, mu=None: self.eval_tf(s, mu=mu) + other.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: self.eval_dtf(s, mu=mu) + other.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.dim_input == other.dim_input
        assert self.dim_output == other.dim_output

        tf = lambda s, mu=None: other.eval_tf(s, mu=mu) - self.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: other.eval_dtf(s, mu=mu) - self.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

    def __neg__(self):
        tf = lambda s, mu=None: -self.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: -self.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

    def __mul__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.dim_input == other.dim_input

        tf = lambda s, mu=None: self.eval_tf(s, mu=mu) @ other.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: (self.eval_dtf(s, mu=mu) @ other.eval_tf(s, mu=mu)
                                  + self.eval_tf(s, mu=mu) @ other.eval_dtf(s, mu=mu))
        return self.with_(tf=tf, dtf=dtf)

    def __rmul__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.dim_output == other.dim_input

        tf = lambda s, mu=None: other.eval_tf(s, mu=mu) @ self.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: (other.eval_dtf(s, mu=mu) @ self.eval_tf(s, mu=mu)
                                  + other.eval_tf(s, mu=mu) @ self.eval_dtf(s, mu=mu))
        return self.with_(tf=tf, dtf=dtf)
